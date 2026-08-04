.. _autotuning:

Autotuning
==========

FlashInfer includes an autotuner that selects the best kernel implementation
(runner and tactic) for each operation and input shape by profiling at runtime.

What Is Autotuning?
-------------------

Several FlashInfer operations -- GEMM and MoE -- support multiple backend
implementations (runners).  Each runner may also expose several low-level
tactics (e.g. tile sizes, pipeline stages).  The best choice depends on the
hardware, data types, and input shapes of your workload.

Without autotuning, FlashInfer picks a default (fallback) tactic.  With
autotuning enabled, the autotuner profiles every candidate for a given shape and
automatically selects the fastest one.

Enabling Autotuning
-------------------

Wrap the portion of your code that you want to tune inside the
``flashinfer.autotune`` context manager:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune():
        # All FlashInfer ops executed here will be profiled.
        output = flashinfer.gemm.bmm_fp8(A, B, A_scale, B_scale, dtype=out_dtype)

The first time an operation runs inside the context, the autotuner benchmarks
all available runners and tactics for that ``(operation, backend, shape)``
combination.  Subsequent calls with the same shape reuse the cached result
without re-profiling.

You can also pass ``tune_mode=True`` explicitly (the default):

.. code-block:: python

    with flashinfer.autotune(True):
        output = flashinfer.gemm.bmm_fp8(A, B, A_scale, B_scale, dtype=out_dtype)

Note that ``flashinfer.autotune()``, ``flashinfer.autotune(True)``, and ``flashinfer.autotune(tune_mode=True)`` are all equivalent.

When ``tune_mode=False``, the context manager enters a no-profiling mode that
only uses previously cached or loaded configs. This is equivalent to ``flashinfer.autotune(False)``:

.. code-block:: python

    with flashinfer.autotune(False):
        # No profiling -- uses default/fallback tactic if nothing is cached.
        model(inputs)

Autotuning in the Benchmark Harness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FlashInfer benchmark harness supports autotuning via the ``--autotune``
flag:

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune

Config Lookup Priority
----------------------

When a FlashInfer operation executes, the autotuner resolves the best
``(runner, tactic)`` by searching these sources in order:

1. **In-memory profiling cache** — results from live autotuning in the current
   process.
2. **User-loaded file configs** — loaded via ``load_configs()`` or
   ``autotune(cache=...)``.
3. **Managed store entries** — the on-disk store used by
   ``autotune(v2_opt_in=True)``, consulted once per key per process and then
   memoized. Only present when a store is attached; see
   :ref:`autotuner-v2` below.
4. **Bundled package configs** — legacy ``.py`` config files shipped with
   FlashInfer (only when the ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1``
   environment variable is set and tuning mode is off).
5. **Fallback tactic (−1)** — a safe default that every runner must implement.

Notably, user-loaded file configs (level 2) are
**always consulted, even during tuning mode**, so that already-tuned shapes
from a cache file are never re-profiled.

Config Caching
--------------

.. note::

   Config caching (the ``cache`` parameter) is **experimental**.
   Single-process and multi-threaded use is fully supported.
   Multi-process and multi-node use is best-effort: concurrent writes to
   a shared cache file may result in lost updates from race conditions.

   This caveat applies to the **legacy file cache** described in this
   section. The managed store (``autotune(v2_opt_in=True)``, see
   :ref:`autotuner-v2`) publishes one atomic file per tuned operation and
   is safe for concurrent multi-rank tuning.

By default, autotuning results live only in memory and are lost when the
process exits.  The ``cache`` parameter on ``flashinfer.autotune`` lets you
persist results to a JSON file and load them back in future runs, avoiding
repeated profiling.

Saving Tuned Configs
^^^^^^^^^^^^^^^^^^^^

Pass a file path to ``cache`` when tuning.  On exit, all profiled configs are
written to that file:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune(True, cache="my_configs.json"):
        model(inputs)
    # On exit, tuned configs are saved to my_configs.json.

Loading Cached Configs
^^^^^^^^^^^^^^^^^^^^^^

To reuse previously tuned configs without profiling, pass ``tune_mode=False``
with the same cache path:

.. code-block:: python

    import flashinfer

    with flashinfer.autotune(False, cache="my_configs.json"):
        # Configs are loaded on entry.  No profiling occurs.
        model(inputs)

Incremental Tuning
^^^^^^^^^^^^^^^^^^

Cache files support incremental updates.  When ``autotune(True, cache=path)``
exits, ``save_configs`` performs the following merge:

1. Previously loaded configs (from the file read on entry) are used as a base.
2. Newly profiled configs are overlaid (new results take priority for duplicate
   keys).
3. The file on disk is re-read and merged, so that configs saved by other
   sessions since entry are also preserved (in-memory results still win on
   overlap).
4. The merged result is atomically written back to the same file.

This means you can run multiple tuning sessions -- for example different batch
sizes or sequence lengths -- and accumulate all configs in a single file:

.. code-block:: python

    # Session 1: tune with batch_size=1
    with flashinfer.autotune(True, cache="configs.json"):
        run_model(batch_size=1)

    # Session 2: tune with batch_size=32 (configs.json now has both)
    with flashinfer.autotune(True, cache="configs.json"):
        run_model(batch_size=32)

Cache Hit Behavior During Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``autotune(True, cache=path)`` is active and a matching config is found in
the cache file, the autotuner uses it directly **without re-profiling**.  This
means:

- Shapes that were already tuned are skipped, saving time.
- Only truly new shapes trigger profiling.
- A log message is printed once per ``(operation, runner)`` pair when a cache
  hit is detected.

Caching in the Benchmark Harness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The benchmark harness supports config caching via the ``--autotune_cache`` flag.

**Tune and save during benchmarking:**

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune --autotune_cache my_configs.json

**Run with cached configs (no profiling):**

.. code-block:: bash

    python3 flashinfer_benchmark.py \
        --routine mm_fp4 --m 4 --n 7168 --k 4608 \
        --out_dtype bfloat16 --backends cudnn cutlass trtllm \
        --use_128x4_sf_layout --use_nvfp4 \
        --autotune_cache my_configs.json

Cache File Format
^^^^^^^^^^^^^^^^^

The cache file is a plain JSON dictionary.  Each key is a string representation
of ``(custom_op, runner_class_name, optimization_profile)`` and each value is
``[runner_class_name, tactic]``:

.. code-block:: json

    {
      "_metadata": {
        "flashinfer_version": "0.6.3",
        "cuda_version": "13.0",
        "cublas_version": "13.2.1",
        "cudnn_version": "91900",
        "gpu": "NVIDIA B200"
      },
      "('fp4_gemm', 'CudnnFp4GemmRunner', ((4, 7168), (7168, 4608), ...))": [
        "CudnnFp4GemmRunner",
        3
      ],
      "('flashinfer::trtllm_fp4_block_scale_moe', 'MoERunner', ((1, 7168), (1, 256), (1, 8), (1, 8), (1, 3584), (1, 448)))": [
        "MoERunner",
        [
          8,
          34
        ]
      ]
    }

The ``_metadata`` key records the environment that created the cache file
(FlashInfer version, CUDA, cuBLAS, cuDNN, and GPU).

On load, ``_metadata`` is compared against the current environment.  If any
field differs (e.g. different GPU, FlashInfer version, or cuBLAS version),
the **entire cache is skipped** — no configs are loaded.  This prevents
silently using invalid tactics.  A warning is logged once per process with
the mismatch details; repeats for further cache files are logged at DEBUG.

What happens to the file on the next save depends on the kind of mismatch:

* The saved value is **definite** (e.g. ``"cudnn_version": "91900"`` vs
  current ``92101``): the file belongs to a different environment sharing
  this cache path.  It is never overwritten — the autotuner behaves as if
  ``cache=None`` — so configs tuned for that environment survive.  Use a
  different cache path for the current environment, or delete the file to
  re-prime it.
* The saved value is **indeterminate** (``"unknown"``, recorded when the
  writer could not detect e.g. its cuDNN version, or a field missing from
  files written by older FlashInfer versions): the file's configs cannot
  be attributed to any environment, so the next tuned save **replaces**
  the file with freshly tuned configs stamped with the current
  environment's metadata.  The cache heals itself instead of forcing a
  full re-tune on every startup forever.

Advanced users can bypass individual checks by manually editing the JSON file
and setting a metadata field to ``"*"``.  For example, setting
``"cudnn_version": "*"`` in ``_metadata`` will skip the cuDNN version check
while still enforcing all other fields.

Tactics are typically integers, but some runners use compound tactics (e.g.
``(tile_size, gemm1_tactic, gemm2_tactic)``).  These are serialized as nested
JSON arrays and restored to tuples on load.

The file is human-readable but not portable. Config ordering is not guaranteed to be
stable across FlashInfer, CUDA, cuDNN, or cuBLAS versions.

.. _autotuner-v2:

Autotuner v2 — Managed Persistence (Experimental)
-------------------------------------------------

.. note::

   **Experimental and opt-in.** ``v2_opt_in`` is a transitional argument: it
   exists so callers migrate explicitly rather than by surprise, and it is
   expected to be removed once the default flips. Do not build long-term
   configuration around it. Design rationale, including what is *not* yet
   decided about v1's future, is in ``docs/design_docs/autotuner_v2.md``.

With the legacy cache, the *caller* owns persistence: the filename is the cache
identity, so every consumer reimplements hashing, atomic writes, and staleness
handling. Autotuner v2 moves that ownership into FlashInfer. Opt in with
``v2_opt_in=True``:

.. code-block:: python

    import flashinfer

    # Warmup: tune uncovered shapes and publish each winner atomically.
    with flashinfer.autotune(True, v2_opt_in=True):
        model(dummy_inputs)

    # Serving: no context needed -- the store stays attached for the process.
    model(inputs)

A later process in the same environment hydrates instead of profiling:

.. code-block:: python

    with flashinfer.autotune(False, v2_opt_in=True):
        pass                       # hydrate only
    model(inputs)                  # reuses tuned winners

``tune_mode`` keeps its meaning: ``True`` profiles and publishes, ``False``
replays already-tuned winners without profiling.

What it changes
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Legacy (``cache=<path>``)
     - v2 (``v2_opt_in=True``)
   * - Cache identity
     - The filename you choose
     - An environment hash FlashInfer computes
   * - Unit of write
     - One JSON file, rewritten on exit
     - One atomic file per tuned operation, published as it is measured
   * - Concurrent ranks
     - Best-effort; writes can be lost
     - Safe -- redundant work, last valid write wins
   * - Crash mid-tuning
     - Nothing is saved
     - Every winner measured so far is already on disk
   * - Wrong environment
     - Whole file rejected; you pick a new path
     - A different directory; nothing to invalidate by hand

Cache location
^^^^^^^^^^^^^^

The store lives under ``FLASHINFER_CACHE_DIR/autotune``, overridable with the
``FLASHINFER_AUTOTUNE_CACHE_DIR`` environment variable or the ``cache_root``
argument. ``cache_root`` is a **directory** and is placement only -- schema and
environment namespaces live below it, so no choice of root can mix incompatible
entries::

    <root>/v2/<environment_hash>/
        manifest.json                # the environment, human-readable
        entries/<operation_hash>.json

Entries do not survive a FlashInfer upgrade: the version is part of the
environment hash, so a new version simply starts a new directory and the old
one is left alone. Autotune data is a disposable optimization -- re-tuning once
after an upgrade is the intended cost.

Measurement policy
^^^^^^^^^^^^^^^^^^

Tuning measures the way you deploy. The decisive question is whether per-call
**host** cost counts: under CUDA-graph serving it is paid once at capture, but
under eager serving it is paid on every call.

.. code-block:: python

    from flashinfer import MeasurementPolicy

    with flashinfer.autotune(
        True, v2_opt_in=True,
        measure=MeasurementPolicy(execution_mode="cuda_graph"),
    ):
        model(dummy_inputs)

``execution_mode`` accepts ``"auto"`` (default, today's behavior),
``"cuda_graph"``, or ``"eager"``. The policy is part of the store's environment
identity, so entries tuned under different policies never overwrite each other.

Distributed use
^^^^^^^^^^^^^^^

With a shared filesystem, every rank may tune simultaneously: publishes are
atomic and redundant work is harmless. Ranks may still hold different
locally-measured winners in memory until they re-read the store; after a
post-tuning barrier, call:

.. code-block:: python

    flashinfer.autotune_v2_reload()

so every rank serves byte-identical tactics.

API Reference
-------------

``flashinfer.autotune``
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    flashinfer.autotune(tune_mode: bool = True, cache: str | None = None)

Context manager for autotuning with optional file-based caching.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``tune_mode``
     - ``bool``
     - If ``True``, profile uncovered shapes during execution.
       If ``False``, only use cached/loaded configs (no profiling).
   * - ``cache``
     - ``str | None``
     - Optional path to a JSON config file.
       On entry, configs are loaded from this file if it exists.
       On exit, configs are saved back to this file when ``tune_mode=True``.
       Legacy implementation only; cannot be combined with ``v2_opt_in=True``.
   * - ``v2_opt_in``
     - ``bool``
     - Opt into the v2 implementation (managed store). ``False`` (default) is
       byte-identical to before this argument existed. **Transitional** --
       see :ref:`autotuner-v2`.
   * - ``persistent_cache``
     - ``bool``
     - Whether v2 may touch disk. ``True`` (default) attaches the managed
       store for the remainder of the process. Requires ``v2_opt_in=True``.
   * - ``cache_root``
     - ``str | None``
     - Root **directory** for the managed store (placement only). Defaults to
       ``FLASHINFER_AUTOTUNE_CACHE_DIR`` or ``FLASHINFER_CACHE_DIR/autotune``.
       Requires ``v2_opt_in=True``.
   * - ``measure``
     - ``MeasurementPolicy | None``
     - How tactics are timed during profiling. Requires ``v2_opt_in=True``.

**Behavior matrix:**

.. list-table::
   :header-rows: 1
   :widths: 15 10 20 20 35

   * - ``tune_mode``
     - ``cache``
     - Load on entry?
     - Save on exit?
     - Use case
   * - ``True``
     - path
     - Yes (if file exists)
     - Yes (incremental)
     - Cache hits skip profiling; misses are tuned and merged back
   * - ``True``
     - ``None``
     - No
     - No
     - Tune in-memory only (results lost on exit)
   * - ``False``
     - path
     - Yes (if file exists)
     - No
     - Inference with pre-tuned configs
   * - ``False``
     - ``None``
     - No
     - No
     - No-op (default behavior)


``flashinfer.MeasurementPolicy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    flashinfer.MeasurementPolicy(execution_mode: str = "auto",
                                 cold_l2: bool | None = None)

How tuning measurements are taken. ``execution_mode`` is ``"auto"`` (today's
behavior), ``"cuda_graph"`` (host cost excluded, as graph serving pays it), or
``"eager"`` (per-call host cost included). ``cold_l2`` forces cold- or hot-L2
profiling; ``None`` inherits each operation's default. Part of the store's
environment identity -- see :ref:`autotuner-v2`.

``flashinfer.autotune_v2_reload``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    flashinfer.autotune_v2_reload() -> None

Drops in-process tuned winners so later lookups re-read the managed store.
Call on every rank *after* a post-tuning barrier so all ranks serve
byte-identical tactics.

``flashinfer.autotune_v2``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    flashinfer.autotune_v2(mode: str = "tune", persistent_cache: bool = True, ...)

The v2 implementation, callable directly. ``autotune(v2_opt_in=True)``
dispatches to it. ``mode="tune"`` / ``"replay"`` corresponds to
``tune_mode=True`` / ``False``. Prefer the ``autotune()`` spelling: this name
is scheduled for removal.

Multi-Thread / Multi-Process Considerations
--------------------------------------------

Quick Reference
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Environment
     - Safe?
     - Notes
   * - Single process
     - Yes
     - Fully safe.
   * - Multi-threaded (single process)
     - Yes
     - All state is lock-protected.
   * - Multi-process, each with its own cache file
     - Yes
     - No shared state.
   * - Multi-process, shared file, **reading only**
     - Yes
     - Readers never see partial files.
   * - Multi-process, shared file, **all writing**
     - Best-effort
     - Works under low contention. Under high contention the last writer
       can overwrite another's results. Use per-rank files for guaranteed
       correctness.

Thread Safety
^^^^^^^^^^^^^

The ``AutoTuner`` singleton is protected by a reentrant lock
(``threading.RLock``).  All state-mutating operations -- ``search_cache``,
``choose_one``, ``save_configs``, ``load_configs``, ``clear_cache``, and the
mode-flag save/restore in ``autotune()`` -- acquire this lock, so multiple
threads can safely share the same autotuner instance.

During tuning mode, the lock also serializes GPU profiling per process, which is the
correct behavior since concurrent kernel measurements would interfere with each
other.

Multi-Process
^^^^^^^^^^^^^

Each process has its own ``AutoTuner`` singleton (separate address space), so
in-memory state is fully isolated.  The only shared resource is the cache
**file** on disk.

- **Reads are safe.**  Writes use ``os.replace`` (atomic on local filesystems),
  so a concurrent reader always sees either the old or new complete file, never
  a partial one.
- **Concurrent writes are best-effort.**  Before writing, ``save_configs``
  re-reads the file from disk and merges any new entries from other processes
  (in-memory results win on overlap).  This significantly reduces the
  lost-update window.  However, the read-merge-write sequence is not itself
  atomic, so two truly simultaneous writers can still race::

      Process A                          Process B
      ─────────                          ─────────
      1. Read file {X, Y}
                                         2. Read file {X, Y}
      3. Merge → {X, Y, Z}
      4. Write {X, Y, Z}
                                         5. Merge → {X, Y, W}
                                            (stale: doesn't see Z)
                                         6. Write {X, Y, W}
                                            ← Z is lost

If you are tuning with multiple processes (e.g. multi-GPU
with ``torchrun``), you could use separate output files per rank and merge them afterwards:

.. code-block:: python

    import json

    merged = {}
    for path in ["configs_rank0.json", "configs_rank1.json"]:
        with open(path) as f:
            merged.update(json.load(f))

    with open("configs_merged.json", "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

.. note::

   Atomic file writes rely on ``os.replace()``, which maps to the POSIX
   ``rename()`` syscall.  This is atomic on all local filesystems and is
   expected to be atomic on most network filesystems (NFS, Lustre) per POSIX
   semantics.  FlashInfer's cubin caching also relies on this guarantee.
