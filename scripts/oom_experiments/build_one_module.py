#!/usr/bin/env python3
"""Build exactly one JitSpec module, in isolation, for memory profiling."""

import argparse
import importlib
import json
import os
import shutil
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "spec",
        help="dotted_path:func_name, e.g. flashinfer.jit.fused_moe:gen_cutlass_fused_moe_sm100_module",
    )
    p.add_argument(
        "--arches",
        default=None,
        help='Override FLASHINFER_CUDA_ARCH_LIST, e.g. "10.0a" or "9.0a 10.0a 10.3a 11.0a 12.0f"',
    )
    p.add_argument(
        "--manifest", default=None, help="Optional path to dump a manifest JSON"
    )
    args = p.parse_args()

    if args.arches:
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = args.arches
    os.environ["FLASHINFER_JIT_VERBOSE"] = "1"

    # Import AFTER setting env so arch list is read fresh
    from flashinfer.jit import env as jit_env  # noqa

    mod_path, func_name = args.spec.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, func_name)

    print(
        f"[exp] arches: {os.environ.get('FLASHINFER_CUDA_ARCH_LIST', '<default>')}",
        flush=True,
    )
    print(f"[exp] calling {args.spec}() ...", flush=True)

    spec = fn()
    print(f"[exp] JitSpec name: {spec.name}", flush=True)
    print(f"[exp] sources: {len(spec.sources)} files", flush=True)
    print(f"[exp] build_dir: {spec.build_dir}", flush=True)

    # Wipe ONLY this module's cache so we force a full recompile of these TUs
    if Path(spec.build_dir).exists():
        shutil.rmtree(spec.build_dir)
        print(f"[exp] wiped {spec.build_dir}", flush=True)

    t0 = time.time()
    spec.build(verbose=True, need_lock=False)
    t1 = time.time()
    print(f"[exp] build wall time: {t1 - t0:.1f}s", flush=True)

    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(
                {
                    "spec": args.spec,
                    "module_name": spec.name,
                    "n_sources": len(spec.sources),
                    "sources": [str(s) for s in spec.sources],
                    "arches": os.environ.get("FLASHINFER_CUDA_ARCH_LIST"),
                    "max_jobs": os.environ.get("MAX_JOBS"),
                    "wall_seconds": t1 - t0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
