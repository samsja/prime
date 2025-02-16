from dataclasses import dataclass
import os
import time
from typing import TYPE_CHECKING, Iterator, Literal

import torch
import torch.distributed as dist
import wandb

from zeroband.data import TEST_VOCAB_SIZE, DataConfig, get_dataloader
from zeroband.lr_scheduler import get_scheduler
from zeroband.models.llama import get_model
from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils import (
    FakeTokenizer,
    PerfCounter,
    get_peak_flops,
    get_num_params,
    get_num_flop_per_token,
    apply_ac_ckpt,
)
from zeroband.logger import get_logger

from transformers import AutoTokenizer
from pydantic_config import BaseConfig, parse_argv
import torch.nn.functional as F

from zeroband.world_info import get_world_info


class AdamConfig(BaseConfig):
    type: Literal["adam"] = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.1
    betas1: float = 0.9
    betas2: float = 0.95


class OptimConfig(BaseConfig):
    optim: AdamConfig = AdamConfig()
    sched_type: Literal["cosine", "linear", "wsd-sqrt"] = "cosine"
    warmup_steps: int = 1000
    stable_steps: int = 80_000
    total_steps: int = 88_000
    batch_size: int = 512


class TrainConfig(BaseConfig):
    micro_bs: int = 1
    ac_ckpt: bool | int = False
    reshard_after_forward: bool = True  # old shard grad op True mean full shard
    torch_compile: bool = True


class Config(BaseConfig):
    name_model: Literal["debugmodel", "70M", "150M", "271M", "1B", "7B", "10B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3"] = "llama3"

    project: str = "prime_simple"
    wandb: bool = True

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    svd_low_rank: int | None = None
    svd_warmup_steps: int = 0


@dataclass
class TrainingProgress:
    total_tokens: int
    outer_step: int
    step: int


class PowerSGD:
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        rank=1,
        min_compression_rate=2,
        use_error_feedback=True,
        orthogonalization_epsilon=0,
        random_seed=0,
        start_powersgd_iter=10,  # Wait this many iterations before starting compression
    ):
        self.params = list(params)  # Convert iterator to list to reuse
        self.rank = rank
        self.min_compression_rate = min_compression_rate
        self.use_error_feedback = use_error_feedback
        self.orthogonalization_epsilon = orthogonalization_epsilon
        self.start_powersgd_iter = start_powersgd_iter
        self.current_iter = 0

        # For error feedback - one per parameter
        self.errors = [None for _ in self.params]

        # For warm start - one pair per parameter
        self.p_memories = [None for _ in self.params]
        self.q_memories = [None for _ in self.params]

        # RNG for consistent random initialization across processes
        self.rng = torch.Generator(device="cuda")  # Change device to cuda
        self.rng.manual_seed(random_seed)

        # Stats tracking
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0

    def should_compress(self, tensor):
        """Determine if tensor should be compressed based on compression rate."""
        # During warmup, don't compress
        if self.current_iter < self.start_powersgd_iter:
            return False

        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        rank = min(n, m, self.rank)

        uncompressed_size = n * m
        compressed_size = (n + m) * rank

        return compressed_size * self.min_compression_rate < uncompressed_size

    def orthogonalize(self, matrix):
        """Apply Gram-Schmidt orthogonalization."""
        num_cols = matrix.shape[2]
        for i in range(num_cols):
            # Normalize the i'th column
            col = matrix[:, :, i : i + 1]
            col.div_(torch.norm(col, dim=1, keepdim=True) + self.orthogonalization_epsilon)

            # Project it on the rest and remove it
            if i + 1 < num_cols:
                rest = matrix[:, :, i + 1 :]
                rest.sub_(torch.sum(col * rest, dim=1, keepdim=True) * col)

    def compress_decompress(self, tensor, index):
        """Compress and decompress a tensor using PowerSGD."""
        original_shape = tensor.shape
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        rank = min(n, m, self.rank)

        # Update compression stats
        self.total_numel_before_compression += n * m
        self.total_numel_after_compression += (n + m) * rank

        # Handle error feedback
        if self.use_error_feedback and self.errors[index] is not None:
            matrix += self.errors[index].view(n, m)

        # Keep copy for error feedback
        original_matrix = matrix.clone() if self.use_error_feedback else None

        # Initialize Q if not warm starting
        if self.q_memories[index] is None:
            self.q_memories[index] = torch.randn(
                (1, m, rank), generator=self.rng, device=tensor.device, dtype=tensor.dtype
            ).to(tensor.device)

        # Initialize P
        if self.p_memories[index] is None:
            self.p_memories[index] = torch.empty((1, n, rank), device=tensor.device, dtype=tensor.dtype).to(
                tensor.device
            )

        # Orthogonalize Q
        self.orthogonalize(self.q_memories[index])

        # Compute P = MQ
        torch.bmm(matrix.unsqueeze(0), self.q_memories[index], out=self.p_memories[index])

        # Allreduce P
        dist.all_reduce(self.p_memories[index])

        # Orthogonalize P
        self.orthogonalize(self.p_memories[index])

        # Compute Q = M^TP
        torch.bmm(matrix.t().unsqueeze(0), self.p_memories[index], out=self.q_memories[index])

        # Allreduce Q
        dist.all_reduce(self.q_memories[index])
        self.q_memories[index].div_(dist.get_world_size())

        # Compute M ≈ PQ^T
        torch.bmm(self.p_memories[index], self.q_memories[index].transpose(1, 2), out=matrix.unsqueeze(0))
        matrix = matrix.squeeze(0)

        # Update error for error feedback
        if self.use_error_feedback:
            self.errors[index] = original_matrix - matrix

        return matrix.view(original_shape)

    def step(self):
        """Compress gradients of all parameters and perform allreduce."""
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    if self.should_compress(param.grad):
                        param.grad.data = self.compress_decompress(param.grad.data, i)
                    else:
                        # For small tensors or during warmup, just do regular allreduce
                        dist.all_reduce(param.grad.data)
                        param.grad.data.div_(dist.get_world_size())

        # Increment iteration counter
        self.current_iter += 1

        # Print compression stats periodically
        if self.current_iter % 100 == 0 and self.current_iter >= self.start_powersgd_iter:
            compression_rate = (
                self.total_numel_before_compression / self.total_numel_after_compression
                if self.total_numel_after_compression > 0
                else 0
            )
            print(f"Iteration {self.current_iter}")
            print(f"Compression rate: {compression_rate:.2f}x")
            print(f"Total elements before compression: {self.total_numel_before_compression}")
            print(f"Total elements after compression: {self.total_numel_after_compression}")


def train(config: Config):
    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0, (
        f"The micro batch size ({config.train.micro_bs}) must divide the number of samples on each GPU ({batch_size})."
    )
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    # Load tokenizer
    if config.data.fake and config.name_model == "debugmodel":
        tokenizer = FakeTokenizer()
    elif config.type_model == "llama2":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    elif config.type_model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size,
        rank=world_info.rank,
        batch_size=config.train.micro_bs,
        data_config=config.data,
    )
    train_dataloader_iterator = iter(train_dataloader)

    model, model_config = get_model(
        type_model=config.type_model,
        name_model=config.name_model,
        seq_length=config.data.seq_length,
        vocab_size=len(tokenizer) if config.name_model != "debugmodel" or not config.data.fake else TEST_VOCAB_SIZE,
    )
    model = model.to(world_info.local_rank)

    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    num_params = get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")
    num_flop_per_token = get_num_flop_per_token(
        num_params,
        model_config,
        config.data.seq_length,
    )

    if config.train.ac_ckpt:
        num = 1 if isinstance(config.train.ac_ckpt, bool) else config.train.ac_ckpt
        apply_ac_ckpt(model, num)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.optim.optim.lr,
        weight_decay=config.optim.optim.weight_decay,
        betas=(config.optim.optim.betas1, config.optim.optim.betas2),
    )

    scheduler = get_scheduler(
        sched_type=config.optim.sched_type,
        optimizer=optimizer,
        num_warmup_steps=config.optim.warmup_steps,
        num_stable_steps=config.optim.stable_steps,
        num_training_steps=config.optim.total_steps,
    )

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    perf_counter = PerfCounter(window_size=10)

    power_sgd = (
        PowerSGD(model.parameters(), config.svd_low_rank, start_powersgd_iter=config.svd_warmup_steps)
        if config.svd_low_rank is not None
        else None
    )

    while True:
        loss_batch = 0

        for grad_acc_step in range(gradient_accumulation_steps):
            # is_accumulating = grad_acc_step < gradient_accumulation_steps - 1

            batch = next(train_dataloader_iterator)
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            seqlens = [seqlen.to("cuda") for seqlen in batch["seqlens"]]
            block_mask = create_block_mask_from_seqlens(seqlens) if seqlens is not None else None

            logits = model(tokens=input_ids, block_mask=block_mask).contiguous()
            flatten_logits = logits.reshape(-1, logits.size(-1))  # b seq vocab -> (b * seq) vocab
            flatten_labels = labels.reshape(-1)  # b seq -> (b * seq)

            ce_loss = F.cross_entropy(flatten_logits, flatten_labels)

            del logits
            del flatten_logits
            del flatten_labels

            loss = ce_loss / gradient_accumulation_steps
            loss.backward()
            loss_batch += loss.detach().clone()

            # Launch both allreduces at the same time to hide latency
            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG)

        power_sgd.step()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore (is a dtensor)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        # logging
        training_progress.step += 1
        inner_lr = [group["lr"] for group in optimizer.param_groups][0]

        # syncing loss across all data parallel rank within a nodes
        new_tokens = config.data.seq_length * config.optim.batch_size
        perf_counter.count_tokens(new_tokens)
        training_progress.total_tokens += new_tokens

        metrics = {
            "Loss": loss_batch.item(),
            "step": training_progress.step,
            "inner_lr": inner_lr,
            "Perplexity": torch.exp(loss_batch).item(),
            "total_tokens": training_progress.total_tokens,
            "time": time.time(),
            "grad_norm": grad_norm.item(),
        }

        log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

        tokens_per_second = perf_counter.get_tokens_per_second()
        if tokens_per_second is not None:
            metrics["tokens_per_second"] = tokens_per_second
            metrics["mfu"] = 100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
            log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

        if world_info.rank == 0 and config.wandb:
            wandb.log(metrics)

        logger.info(log)

        if training_progress.step > config.optim.total_steps:
            break

    logger.info("Training finished, exiting ...")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ  # type: ignore
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    config = Config(**parse_argv())  # type: ignore
    world_info = get_world_info()
    logger = get_logger()

    torch.cuda.set_device(world_info.local_rank)
    dist.init_process_group(backend="nccl")

    train(config)
