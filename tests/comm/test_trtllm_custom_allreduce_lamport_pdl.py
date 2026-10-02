"""Ordering test for the lamport one-shot all-reduce + RMSNorm kernel under PDL.

``lamport_style_one_shot_all_reduce_norm_kernel`` in
``include/flashinfer/comm/trtllm_allreduce.cuh`` is selected by the
``trtllm_custom_all_reduce`` binding for ONESHOT + RESIDUAL_RMS_NORM with fp16
or bf16, at most 16 tokens and a hidden size of at least 256. With
``launch_with_pdl=True`` it may start before the kernel that produces
``residual`` (and ``bias`` / ``weight``) has finished, so every read of those
buffers has to be issued after ``cudaGridDependencySynchronize()``.

The Python wrapper for this binding was removed in #5218, but the TVM-FFI export
is still compiled into the ``trtllm_comm`` module, so the test calls it through
the raw JIT module.

The producer kernel widens the race window on purpose: it triggers the
dependent launch first, spins, and only then writes residual, bias and weight.
A read issued before the sync therefore observes the NaN the buffer held
beforehand, which then shows up in the output.
"""

import multiprocessing as mp
import pathlib
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

import flashinfer.comm as comm
from flashinfer.comm.torch_symmetric_memory import _alloc_symm_buffer_bytes
from flashinfer.jit.comm import gen_trtllm_comm_module
from flashinfer.utils import get_compute_capability, round_up

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda:0"))[0] not in (9, 10, 12),
    reason="trtllm_comm kernels support SM90/SM100/SM12x only",
)

# Lamport dispatch bound, mirroring kLamportTokenNumThreshold in the kernel header.
LAMPORT_TOKEN_NUM_THRESHOLD = 16
MAX_ALL_REDUCE_BLOCKS = 24
TEST_LOOP = 20
EPS = 1e-6
# Long enough that the fused kernel is resident and past its prologue before the
# producer writes anything, short enough to keep the test quick.
SPIN_CYCLES = 4_000_000

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FLASHINFER_INCLUDE = str(_REPO_ROOT / "include")
_SPDLOG_INCLUDE = str(_REPO_ROOT / "3rdparty" / "spdlog" / "include")
_CUDA_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

_CPP_SOURCE = r"""
void pdl_delayed_producer(at::Tensor out, at::Tensor value, int64_t spin_cycles);
"""

# The copy is done on raw 16-bit words, so one kernel serves both float16 and
# bfloat16 without a dtype dispatch.
_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

__global__ void trigger_then_write_kernel(uint16_t* out, const uint16_t* value, int64_t n,
                                          int64_t spin_cycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Let the dependent kernel start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = value[i];
  }
}

void pdl_delayed_producer(at::Tensor out, at::Tensor value, int64_t spin_cycles) {
  TORCH_CHECK(out.is_cuda() && value.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(out.is_contiguous() && value.is_contiguous(), "expected contiguous tensors");
  TORCH_CHECK(out.scalar_type() == value.scalar_type(), "dtype mismatch");
  TORCH_CHECK(out.element_size() == 2, "expected a 16-bit dtype");
  TORCH_CHECK(out.numel() == value.numel(), "size mismatch");
  trigger_then_write_kernel<<<64, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<uint16_t*>(out.data_ptr()), static_cast<const uint16_t*>(value.data_ptr()),
      out.numel(), spin_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _load_producer():
    major, minor = torch.cuda.get_device_capability()
    gencode = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
    return load_inline(
        name="test_trtllm_custom_allreduce_lamport_pdl_producer",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        extra_include_paths=[_FLASHINFER_INCLUDE, _SPDLOG_INCLUDE],
        extra_cuda_cflags=[*_CUDA_FLAGS, gencode],
        functions=["pdl_delayed_producer"],
        verbose=False,
    )


def _alloc_workspace(world_size, max_token_num, hidden_dim, group):
    """Allocate the 7 IPC buffers the custom all-reduce binding expects.

    Same layout as the removed ``trtllm_create_ipc_workspace_for_all_reduce``:
    [comm, comm, barrier_in, barrier_out, lamport_0, lamport_1, lamport_2].
    Returns the per-buffer peer pointers, the allocations to keep alive, and the
    lamport buffer size in fp16 elements.
    """
    buffer_size = world_size * max_token_num * hidden_dim * 4
    flag_size = (MAX_ALL_REDUCE_BLOCKS + 1) * 4 * world_size * 2
    lamport_size = (
        world_size * LAMPORT_TOKEN_NUM_THRESHOLD * world_size * hidden_dim * 2
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    refs, ptrs = [], []
    for size, dtype in [
        (buffer_size, torch.float32),
        (buffer_size, torch.float32),
        (flag_size, torch.int32),
        (flag_size, torch.int32),
        (lamport_size, torch.float16),
        (lamport_size, torch.float16),
        (lamport_size, torch.float16),
    ]:
        peer_ptrs, tensor, handle = _alloc_symm_buffer_bytes(
            round_up(size, 16), world_size, dtype, device, group.group_name
        )
        refs.append((tensor, handle))
        ptrs.append(peer_ptrs)
    return ptrs, refs, lamport_size // 2


def _reference(allreduce_sum, residual, bias, weight, token_num, hidden_dim):
    inter = allreduce_sum + residual.float() + bias.float().repeat(token_num)
    x = inter.view(token_num, hidden_dim)
    norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * weight.float()
    return inter, norm.view(-1)


def _run_lamport_pdl_worker(
    world_size,
    rank,
    dtype,
    hidden_dim,
    distributed_init_port,
    launch_with_pdl=True,
    gpu_offset=0,
):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    producer = _load_producer()
    module = gen_trtllm_comm_module().build_and_load()

    refs = None
    try:
        ptrs, refs, lamport_numel = _alloc_workspace(
            world_size, LAMPORT_TOKEN_NUM_THRESHOLD, hidden_dim, group
        )
        ptr_tensors = [torch.tensor(p, dtype=torch.int64) for p in ptrs]
        tol = 1e-2 if dtype == torch.float16 else 8e-2

        # A failure is recorded rather than raised: the race can be lost on one
        # rank and not another, and leaving the loop early would strand the
        # other ranks in the lamport spin-wait, turning a regression into a hang
        # instead of a failure.
        failures = []
        for config_code in (0, comm.AllReduceStrategyConfig.PUSH_MODE):
            for token_num in (1, LAMPORT_TOKEN_NUM_THRESHOLD):
                message_size = token_num * hidden_dim
                # The lamport buffers must start as negative zero and the flag
                # must advance by one per call, so every case starts clean.
                torch.cuda.synchronize()
                dist.barrier(group=group)
                comm.trtllm_lamport_initialize_all(
                    ptrs[4][rank],
                    ptrs[5][rank],
                    ptrs[6][rank],
                    lamport_numel,
                    torch.float16,
                )
                torch.cuda.synchronize()
                dist.barrier(group=group)

                inp = torch.randn(message_size, dtype=dtype, device=device)
                # The lamport protocol uses -0.0 as its "not yet written" marker.
                inp = torch.where(inp == 0, torch.zeros_like(inp), inp)
                ar_sum = inp.float()
                dist.all_reduce(ar_sum, group=group)

                # residual, bias and weight share one buffer so the delayed
                # producer writes all three upstream-produced inputs at once.
                fused_value = torch.randn(
                    message_size + 2 * hidden_dim, dtype=dtype, device=device
                )
                fused = torch.empty_like(fused_value)
                residual = fused[:message_size]
                bias = fused[message_size : message_size + hidden_dim]
                weight = fused[message_size + hidden_dim :]
                ref_inter, ref_norm = _reference(
                    ar_sum,
                    fused_value[:message_size],
                    fused_value[message_size : message_size + hidden_dim],
                    fused_value[message_size + hidden_dim :],
                    token_num,
                    hidden_dim,
                )
                out = torch.empty(message_size, dtype=dtype, device=device)
                inter = torch.empty(message_size, dtype=dtype, device=device)

                stale = mismatch = 0
                flag_value = 1
                for _ in range(TEST_LOOP):
                    fused.fill_(float("nan"))
                    out.zero_()
                    inter.zero_()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)

                    producer.pdl_delayed_producer(fused, fused_value, SPIN_CYCLES)
                    module.trtllm_custom_all_reduce(
                        inp,
                        out,
                        world_size,
                        rank,
                        token_num,
                        comm.AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        comm.AllReduceStrategyType.ONESHOT,
                        config_code,
                        launch_with_pdl,
                        flag_value,
                        ptr_tensors[0],
                        ptr_tensors[2],
                        ptr_tensors[3],
                        bias,
                        residual,
                        weight,
                        None,
                        EPS,
                        inter,
                        ptr_tensors[4],
                        ptr_tensors[5],
                        ptr_tensors[6],
                    )
                    flag_value += 1
                    torch.cuda.synchronize()

                    if not (torch.isfinite(out).all() and torch.isfinite(inter).all()):
                        stale += 1
                    elif not (
                        torch.allclose(inter.float(), ref_inter, atol=tol, rtol=3e-2)
                        and torch.allclose(out.float(), ref_norm, atol=tol, rtol=3e-2)
                    ):
                        mismatch += 1

                if stale or mismatch:
                    failures.append(
                        f"config={config_code} token_num={token_num}: "
                        f"stale={stale} mismatch={mismatch} of {TEST_LOOP}"
                    )
                dist.barrier(group=group)

        assert not failures, (
            f"rank {rank}: lamport one-shot all-reduce+RMSNorm read upstream data "
            f"before the PDL grid dependency sync (launch_with_pdl={launch_with_pdl})"
            " or produced wrong output in " + ", ".join(failures)
        )
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        del refs
        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple = (),
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (
            (world_size, i, dtype, hidden_dim, distributed_init_port)
            + target_args
            + (gpu_offset,)
        )
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


# Run as: pytest tests/comm/test_trtllm_custom_allreduce_lamport_pdl.py
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_dim", [1024, 4096])
@pytest.mark.parametrize("launch_with_pdl", [True, False])
def test_trtllm_custom_allreduce_lamport_pdl_ordering(
    world_size, dtype, hidden_dim, launch_with_pdl
):
    """residual/bias/weight must not be read before cudaGridDependencySynchronize().

    launch_with_pdl=False is the control arm: plain stream ordering makes the
    producer's writes visible regardless, so it passes either way and shows the
    harness is not simply reporting NaN unconditionally.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    # Warm both extension caches in the parent so the workers do not all compile.
    _load_producer()
    gen_trtllm_comm_module().build_and_load()

    multi_process_parallel(
        world_size,
        dtype,
        hidden_dim,
        _run_lamport_pdl_worker,
        target_args=(launch_with_pdl,),
    )
    print(
        f"lamport pdl ordering tp={world_size} dtype={dtype} hidden={hidden_dim} "
        f"launch_with_pdl={launch_with_pdl}: OK"
    )


if __name__ == "__main__":
    test_trtllm_custom_allreduce_lamport_pdl_ordering(2, torch.bfloat16, 4096, True)
