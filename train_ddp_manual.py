from dataclasses import dataclass
import os
import time
from typing import TYPE_CHECKING, Literal

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


@dataclass
class TrainingProgress:
    total_tokens: int
    outer_step: int
    step: int


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend="newtonschulz5", backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            zeropower_backend = zeropower_backends[group["backend"]]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group["backend_steps"]) for g1 in g.split(g.size(1))])
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_backend(g, steps=group["backend_steps"])
                    scale = max(g.size(0), g.size(1)) ** 0.5  # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)


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

    hidden_matrix_params = [p for n, p in model.layers.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.output.weight]

    # init the optimizer(s)
    adam_params = [
        dict(params=head_params, lr=0.008),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=config.optim.optim.lr,
        momentum=0.95,
        nesterov=True,
        backend="svd",
    )

    optimizers = [optimizer2, optimizer1]

    schedulers = [
        get_scheduler(
            sched_type=config.optim.sched_type,
            optimizer=optimizer,
            num_warmup_steps=config.optim.warmup_steps,
            num_stable_steps=config.optim.stable_steps,
            num_training_steps=config.optim.total_steps,
        )
        for optimizer in optimizers
    ]

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    if world_info.rank == 0 and config.wandb:
        wandb.init(project=config.project, config=config.model_dump())

    if config.train.torch_compile:
        model = torch.compile(model) if not TYPE_CHECKING else model

    perf_counter = PerfCounter(window_size=10)

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

        jobs = []
        for param in model.parameters():
            jobs.append(dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=True))

        for job in jobs:
            job.wait()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore (is a dtensor)

        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        # logging
        training_progress.step += 1
        inner_lr = [group["lr"] for group in optimizers[0].param_groups][0]

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
