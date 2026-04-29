#!/usr/bin/env python3
"""Aggregate per-TU and per-experiment data into a single analysis summary."""

import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    root = Path("aot-memory-reports")
    if not root.exists():
        raise SystemExit(f"no such dir: {root}")

    # exp_id -> {manifest, per_tu_rows}
    experiments = {}
    for exp_dir in sorted(root.glob("exp_*")):
        manifest_p = exp_dir / "manifest.json"
        tsv_p = exp_dir / "nvcc_per_tu.tsv"
        if not manifest_p.exists() or not tsv_p.exists():
            continue
        manifest = json.loads(manifest_p.read_text())
        rows = []
        with tsv_p.open() as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                try:
                    row["peak_rss_kb"] = int(row["peak_rss_kb"])
                    row["wall_s"] = float(row["wall_s"])
                except ValueError:
                    continue
                rows.append(row)
        experiments[exp_dir.name] = {"manifest": manifest, "rows": rows}

    if not experiments:
        raise SystemExit("no experiments found")

    # ─────────────────────────────────────────────────────────────────
    # Table 1: per-cell summary
    # P_worst_VmHWM is ORDINAL — shared-page inflation makes it unreliable as
    # an absolute physical number. cgroup.peak is the cardinal aggregate.
    # ─────────────────────────────────────────────────────────────────
    print("# AOT OOM Experiment — Analysis Summary\n")
    print("## 1. Per-cell summary\n")
    print(
        "`P_worst_VmHWM` is ordinal (ranking-only, has shared-page inflation). "
        "`cgroup.peak` is cardinal (kernel-accounted physical peak — drives the divisor).\n"
    )
    print(
        "| Experiment | Module | Arches | n_arch | MAX_JOBS | n_TUs | P_worst_VmHWM (GiB)* | cgroup.peak (GiB) | wall (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")

    for eid, exp in experiments.items():
        m = exp["manifest"]
        rows = exp["rows"]
        if not rows:
            continue
        peaks_kb = sorted(r["peak_rss_kb"] for r in rows)
        n = len(peaks_kb)
        p_worst = peaks_kb[-1] if peaks_kb else 0
        cgroup_kb = m.get("cgroup_peak_kb", 0)
        wall = m.get("wall_seconds", 0)
        arches = m.get("arches") or "default"
        n_arches = len(arches.split()) if arches != "default" else 0

        spec_short = (
            m.get("spec", "?").split(":")[-1].replace("gen_", "").replace("_module", "")
        )
        print(
            f"| {eid} | {spec_short} | {arches} | {n_arches} | {m.get('max_jobs', '?')} | "
            f"{n} | {p_worst / 1024 / 1024:.2f}* | {cgroup_kb / 1024 / 1024:.2f} | {wall:.0f} |"
        )
    print("\n*Ordinal only — do not sum across rows. See §8 of handoff for why.\n")

    # ─────────────────────────────────────────────────────────────────
    # Table 2: heavy-tail TUs — feeds JOB_POOLS source list
    # VmHWM ranking is robust because shared-page bias is roughly
    # constant across TUs of the same module.
    # ─────────────────────────────────────────────────────────────────
    print("## 2. Heavy-tail TUs — top 30 by VmHWM (feeds Ninja JOB_POOLS list)\n")
    print("| Rank | Source | VmHWM (GiB) | Wall (s) | Experiment |")
    print("|---|---|---|---|---|")
    flat = []
    for eid, exp in experiments.items():
        for r in exp["rows"]:
            flat.append((r["peak_rss_kb"], r["wall_s"], r["source"], eid))
    flat.sort(reverse=True)
    for i, (kb, wall, src, eid) in enumerate(flat[:30], 1):
        print(f"| {i} | `{src}` | {kb / 1024 / 1024:.2f} | {wall:.1f} | {eid} |")

    # ─────────────────────────────────────────────────────────────────
    # Table 3: P_eff regression — cgroup.peak vs MAX_JOBS (per module × arch_list)
    # Linear model: cgroup.peak(j) = K + P_eff × j
    #   P_eff = effective additional physical RAM per concurrent nvcc → divisor
    #   K     = base overhead (Python, ninja, page cache, daemon)
    # Two-point fit (j=1 and j=auto cells) is enough for the slope.
    # ─────────────────────────────────────────────────────────────────
    print("\n## 3. P_eff regression — drives the calibrated MAX_JOBS divisor\n")
    print(
        "Linear fit: `cgroup.peak(j) = K + P_eff × j`. "
        "`P_eff` (GiB/job) is the measured replacement for the hand-picked `8 GiB/job` divisor. "
        "`K` (GiB) is base overhead — subtract from `MemAvailable` before dividing.\n"
    )

    # Group by (module × arches), each group should have ≥2 j-values
    by_cell = defaultdict(list)
    for eid, exp in experiments.items():
        m = exp["manifest"]
        if m.get("cgroup_peak_kb", 0) <= 0:
            continue
        spec = m.get("spec", "?")
        arches = m.get("arches") or "default"
        j = int(m.get("max_jobs", 0) or 0)
        peak_gib = m["cgroup_peak_kb"] / 1024 / 1024
        if j > 0:
            by_cell[(spec, arches)].append((j, peak_gib, eid))

    print(
        "| Module | Arches | Points (j, cgroup.peak GiB) | K (GiB) | P_eff (GiB/job) | Calibrated divisor* |"
    )
    print("|---|---|---|---|---|---|")
    for (spec, arches), pts in by_cell.items():
        if len(pts) < 2:
            continue
        pts.sort()
        # Two-point linear fit (use lowest and highest j)
        j0, p0, _ = pts[0]
        j1, p1, _ = pts[-1]
        if j1 == j0:
            continue
        p_eff = (p1 - p0) / (j1 - j0)
        K = p0 - p_eff * j0
        # Suggest divisor: P_eff with safety margin (round up to nearest 0.5 GiB)
        suggested = max(1.0, round((p_eff * 1.15) * 2) / 2)
        pts_str = ", ".join(f"({j},{p:.1f})" for j, p, _ in pts)
        print(
            f"| {spec.split(':')[-1]} | {arches} | {pts_str} | {K:.2f} | {p_eff:.2f} | **{suggested:.1f}** |"
        )
    print(
        "\n*Includes 15% safety margin, rounded to nearest 0.5 GiB. "
        "Final divisor should be `max(P_eff)` across the cells you care about.\n"
    )

    # Sanity-check column: arch-count effect (single-arch j=1 vs full-arch j=1)
    print("## 4. Arch-count effect on `cgroup.peak` (sanity check)\n")
    print("| Module | Arches | n_arch | cgroup.peak (GiB) at j=1 |")
    print("|---|---|---|---|")
    for _eid, exp in experiments.items():
        m = exp["manifest"]
        if int(m.get("max_jobs", 0) or 0) != 1:
            continue
        arches = m.get("arches") or "default"
        n_arches = len(arches.split()) if arches != "default" else 0
        peak_gib = m.get("cgroup_peak_kb", 0) / 1024 / 1024
        spec_short = m.get("spec", "?").split(":")[-1]
        print(f"| {spec_short} | {arches} | {n_arches} | {peak_gib:.2f} |")
    print(
        "\nLinear scaling ⇒ arch-trimming is a viable lever for AOT smoke. "
        "Sub-linear ⇒ less so.\n"
    )

    print("Done.")


if __name__ == "__main__":
    main()
