"""
Reproduce NaN bug in FlashInfer's trtllm_fp4_block_scale_moe kernel.

The kernel intermittently produces NaN when all of the following are true:
  1. MXFP8 activation quantization is enabled (W4A8 mode)
  2. intermediate_size is not a multiple of 1024 (e.g. 1536)
  3. Running on NVIDIA Driver 570.x with SM100 (Blackwell) cubins

Padding intermediate_size to 1024 (e.g. 2048) eliminates the bug.
The bug is also absent on Driver 580.x.

This script loads real MoE weights, TP-shards them, and calls the kernel
directly — no inference engine needed.

Requirements:
    pip install flashinfer safetensors huggingface_hub

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_kernel_nan_real_weights.py \
        --model-dir /path/to/gpt-oss-120b
"""

import argparse, glob, os, subprocess, sys, time
import torch

assert torch.cuda.is_available(), "CUDA not available"

from flashinfer import (
    mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a,
    trtllm_fp4_block_scale_moe,
)

SF_VEC = 32           # MXFP scale-factor block size
TILE_M = 128          # epilogue tile M for shuffle
TOP_K = 4
N_EXPERTS = 16        # subset of 128 experts (saves GPU memory)
LAYERS = [2, 8]
SIZES = {1536: "BUGGY (round_up(1440, 256))",
         2048: "FIXED (round_up(1440, 1024))"}

pad = torch.nn.functional.pad


def env_info():
    gpu = torch.cuda.get_device_name()
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().split("\n")[0]
    except Exception:
        drv = "unknown"
    return gpu, drv, __import__("flashinfer").__version__


def load_tensor(model_dir, key):
    from safetensors import safe_open
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as sf:
            if key in sf.keys():
                return sf.get_tensor(key)
    return None


def swap_halves(x, axis):
    """Swap gate/up halves -- TRT-LLM expects [up, gate] ordering."""
    if axis < 0: axis += x.ndim
    shape = list(x.shape); shape[axis] //= 2; shape.insert(axis + 1, 2)
    return x.reshape(shape).flip(axis + 1).reshape(x.shape)


def shuffle_expert(w, ws, b):
    """Shuffle one expert's weight/scale/bias for transposed MMA layout."""
    return (shuffle_matrix_a(w.view(torch.uint8), TILE_M),
            shuffle_matrix_sf_a(ws.view(torch.uint8), TILE_M),
            shuffle_matrix_a(b.reshape(-1, 1), TILE_M))


def load_layer(model_dir, layer_idx, inter_size, tp_rank=0, tp_size=2):
    """Load one MoE layer: TP-shard, pad, shuffle for the kernel."""
    t0 = time.time()
    pfx = f"model.layers.{layer_idx}.mlp.experts"
    w13b = load_tensor(model_dir, f"{pfx}.gate_up_proj_blocks")
    w13s = load_tensor(model_dir, f"{pfx}.gate_up_proj_scales")
    w13bi = load_tensor(model_dir, f"{pfx}.gate_up_proj_bias")
    w2b = load_tensor(model_dir, f"{pfx}.down_proj_blocks")
    w2s = load_tensor(model_dir, f"{pfx}.down_proj_scales")
    w2bi = load_tensor(model_dir, f"{pfx}.down_proj_bias")
    assert w13b is not None, f"Weights not found for layer {layer_idx}"

    E, full_I, H = w13b.shape[0], w13b.shape[1] // 2, w13b.shape[2] * SF_VEC
    pH = ((H + 255) // 256) * 256  # pad hidden: 2880 -> 3072
    hp = (pH - H) // 2             # packed FP4 padding

    w13f = w13b.reshape(E, full_I * 2, -1)
    w2f = w2b.reshape(E, H, -1)
    if hp > 0:
        w13f = pad(w13f, (0, hp))
        w13s = pad(w13s, (0, (pH - H) // SF_VEC))
        w2f = pad(w2f, (0, 0, 0, pH - H))
        w2s = pad(w2s, (0, 0, 0, pH - H))

    Ir = full_I // tp_size
    s, e, p = tp_rank * Ir, (tp_rank + 1) * Ir, inter_size - full_I // tp_size
    w13 = torch.cat([pad(w13f[:, :full_I][:, s:e], (0, 0, 0, p)),
                     pad(w13f[:, full_I:][:, s:e], (0, 0, 0, p))], dim=1)
    w13sc = torch.cat([pad(w13s[:, :full_I][:, s:e], (0, 0, 0, p)),
                       pad(w13s[:, full_I:][:, s:e], (0, 0, 0, p))], dim=1)
    gr = Ir // SF_VEC
    w2 = pad(w2f[:, :, s // 2:e // 2], (0, p // 2))
    w2sc = pad(w2s[:, :, tp_rank * gr:tp_rank * gr + gr], (0, p // SF_VEC))
    if w13bi is not None:
        b13 = torch.cat([pad(w13bi[:, :full_I][:, s:e], (0, p)),
                         pad(w13bi[:, full_I:][:, s:e], (0, p))], dim=1).float()
    else:
        b13 = torch.zeros(E, 2 * inter_size, dtype=torch.float32)
    b2 = (pad(w2bi.float(), (0, pH - H)) if w2bi is not None
           else torch.zeros(E, pH, dtype=torch.float32))

    n = N_EXPERTS
    w13, w13sc, b13 = w13[:n].cuda(), w13sc[:n].cuda(), b13[:n].cuda()
    w2, w2sc, b2 = w2[:n].cuda(), w2sc[:n].cuda(), b2[:n].cuda()
    w13, w13sc, b13 = swap_halves(w13, -2), swap_halves(w13sc, -2), swap_halves(b13, -1)

    w13_sh, w13s_sh, b13_sh, w2_sh, w2s_sh, b2_sh = [], [], [], [], [], []
    for i in range(n):
        a, b_, c = shuffle_expert(w13[i], w13sc[i], b13[i])
        w13_sh.append(a); w13s_sh.append(b_); b13_sh.append(c)
        a, b_, c = shuffle_expert(w2[i], w2sc[i], b2[i])
        w2_sh.append(a); w2s_sh.append(b_); b2_sh.append(c)

    P = inter_size
    print(f"  Layer {layer_idx}, I={inter_size}: loaded in {time.time()-t0:.1f}s")
    return dict(
        w13=torch.stack(w13_sh),
        w13s=torch.stack(w13s_sh).reshape(n, 2*P, pH//SF_VEC).view(torch.float8_e4m3fn),
        b13=torch.stack(b13_sh).reshape(n, -1),
        w2=torch.stack(w2_sh),
        w2s=torch.stack(w2s_sh).reshape(n, pH, P//SF_VEC).view(torch.float8_e4m3fn),
        b2=torch.stack(b2_sh).reshape(n, -1),
        g1a=torch.ones(n, dtype=torch.float32, device="cuda"),
        g1b=torch.zeros(n, dtype=torch.float32, device="cuda"),
        g1c=torch.full((n,), 1e4, dtype=torch.float32, device="cuda"),
        E=n, H=pH)


def run_test(w, inter_size, num_iters):
    """Call trtllm_fp4_block_scale_moe repeatedly, count NaN outputs."""
    nan_count = 0
    for i in range(num_iters):
        n = [1, 2, 4, 8, 16, 32, 64][i % 7]
        x = torch.randn(n, w["H"], dtype=torch.bfloat16, device="cuda") * 0.1
        xq, xs = mxfp8_quantize(x, False, alignment=w["H"])
        xs = xs.view(torch.float8_e4m3fn).reshape(n, -1)
        out = trtllm_fp4_block_scale_moe(
            torch.randn(n, w["E"], dtype=torch.bfloat16, device="cuda"),
            None, xq, xs,
            w["w13"], w["w13s"], w["b13"], w["g1a"], w["g1b"], w["g1c"],
            w["w2"], w["w2s"], w["b2"], None, None, None,
            w["E"], TOP_K, None, None, inter_size, 0, w["E"],
            None, 1, True,
            output=torch.empty(n, w["H"], dtype=torch.bfloat16, device="cuda"),
            tune_max_num_tokens=max(n, 8))[0]
        if out.isnan().any().item():
            nan_count += 1
            if nan_count <= 3:
                print(f"    NaN at iter {i}, tokens={n}, "
                      f"{out.isnan().float().mean().item()*100:.1f}% elements")
    return nan_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir",
                    default=os.environ.get("MODEL_DIR", "openai/gpt-oss-120b"))
    ap.add_argument("--num-iters", type=int, default=10000)
    args = ap.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_dir)

    gpu, drv, fi = env_info()
    print(f"Model: {model_dir}\nGPU: {gpu}\nDriver: {drv}\nFlashInfer: {fi}")
    print(f"Config: {args.num_iters} iters, {N_EXPERTS} experts, top_k={TOP_K}\n")

    results = []
    for layer in LAYERS:
        for isize, desc in SIZES.items():
            print(f"{'='*60}\nLayer {layer} | I={isize} | {desc}\n{'='*60}")
            w = load_layer(model_dir, layer, isize)
            t0 = time.time()
            nans = run_test(w, isize, args.num_iters)
            tag = "FAIL" if nans > 0 else "PASS"
            print(f"  NaN={nans}/{args.num_iters} "
                  f"({nans/args.num_iters*100:.4f}%) [{tag}] ({time.time()-t0:.0f}s)\n")
            results.append((layer, isize, nans, args.num_iters, tag))
            del w; torch.cuda.empty_cache()

    print(f"{'='*60}\nSUMMARY\n{'='*60}")
    for layer, isize, nans, total, tag in results:
        print(f"  Layer {layer}, I={isize}: NaN={nans}/{total} "
              f"({nans/total*100:.4f}%) [{tag}]")


if __name__ == "__main__":
    main()
