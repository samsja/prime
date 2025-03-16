# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823

from typing import Generator
from collections import deque

import torch
from torch.optim.optimizer import ParamsT
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed import gather, scatter

@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz(G, steps=10, eps=1e-7, f_iter=nsloop_torch):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
        # DTensor will NaN for sharded compute on Shard(1)
        if isinstance(X, DTensor):
            p = [Shard(0) if isinstance(p, Shard) else p for p in X._spec.placements]
            X = X.redistribute(placements=p)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)  # ensure top singular value <= 1
    X = f_iter(X, steps)
    return X if G.size(-2) <= G.size(-1) else X.mT


def paramst_to_groups(params: ParamsT) -> list[dict]:
    if all(isinstance(p, dict) for p in params):
        return params
    if all(isinstance(p, torch.nn.Parameter) for p in params):
        return [dict(params=params)]
    if all(isinstance(p, list) for p in params):
        return [dict(params=p) for p in params]
    raise ValueError(f"Invalid paramst_to_groups input: {params}")


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    This is a pruned implementation which uses the following hardcoded behaviors:
    * assumed used of 2D+ DTensor parameters, which will always be true if you use FSDP2.
    * nestrov momentum (on the input to NS)
    * EMA momentum (unlike the original Muon, which uses .mul_(beta))

    Arguments:
        params: Params/groups to be optimized.

    Default arguments (used for params with no specific group lr/wd/beta):
        lr: Learning rate.
        wd: Weight decay.
        beta: momentum buffer decay rate.
    """

    def __init__(
        self, params: ParamsT, *, lr: float | None = None, wd: float = 0.0, beta: float = 0.95, ns_steps: int = 5
    ):
        # setup torch optimizer
        defaults = dict(lr=lr, wd=wd, beta=beta, ns_steps=ns_steps)
        groups = paramst_to_groups(list(params))
        super().__init__(groups, defaults)
        # init buffers ahead of time
        for pg in self.param_groups:
            for p in pg["params"]:
                assert isinstance(p, DTensor), "We only support DTensor. Use FSDP2."
                self.mesh = p._spec.device_mesh
                # TODO: figure out how to store optim step state without exploding DCP
                self.state[p] = dict(m=torch.zeros_like(p))
                if p.ndim < 2:
                    raise ValueError(f"0/1D parameters are banned from Muon; user provided {p.shape=}")
                if p.ndim > 2:
                    print(f"WARNING: muon used for {p.shape=}")
            # todo: also declare tensorlists for foreach
            ...

    def filter_group(self, group: dict) -> Generator[tuple[DTensor, DTensor, DTensor, int], None, None]:
        pg, lr, wd, beta = group["params"], group["lr"], group["wd"], group["beta"]
        pg = [p for p in pg if p.grad is not None]
        list_p = [p.data for p in pg]
        list_g = [p.grad.flatten(1) for p in pg]
        list_m = [self.state[p]["m"] for p in pg]
        torch._foreach_lerp_(list_m, list_g, 1 - beta)  # EMA momentum
        torch._foreach_lerp_(list_g, list_m, beta)  # nestrov momentum (for NS input)
        torch._foreach_mul_(list_p, 1 - lr * wd)  # weight decay
        yield from zip(list_p, list_g, list_m)

    @torch.no_grad()
    def step(self, *, prefetch_factor: int = 8):  # <-- changeme to 1 if you have numerical bugs
        # fsdp sharding mesh dim is always last
        r, ws = self.mesh.get_local_rank(-1), self.mesh.size(-1)

        dq = deque()

        def deferred_work(p, g, g_full_block, spec, lr, src_rank, rank):
            if rank == src_rank:
                chunks = list(g_full_block.chunk(ws, dim=0))
                scatter(g.to_local(), chunks, src=src_rank, async_op=True)
            else:
                scatter(g.to_local(), None, src=src_rank, async_op=True) 

       
            # update parameter with NS'd grad
            lr_scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
            p.add_(g, alpha=-lr * lr_scale)

        i = 0
        for group in self.param_groups:
            for p, g, m in self.filter_group(group):
                spec = g._spec
                dest_rank = i  % ws
                if dest_rank == r:
                    gather_lists = [torch.zeros_like(g.to_local()) for _ in range(ws)]
                    gather(g.to_local(), gather_lists, dst=dest_rank, async_op=True) 
                    g_full_block = torch.cat(gather_lists, dim=0)
                    g_full_block.copy_(zeropower_via_newtonschulz(g_full_block, steps=group["ns_steps"]))
                    g_full_block = g_full_block.view_as(p).type_as(p)
                else:
                    
                    g_local = g.to_local()
                    gather(g_local, None, dst=dest_rank, async_op=True)
                    g_full_block = None
                    
                dq.append([p, g, g_full_block, spec, group["lr"], dest_rank, r])
                if len(dq) > prefetch_factor:
                    deferred_work(*dq.popleft())
                i += 1
        for ls in dq:
            deferred_work(*ls)
