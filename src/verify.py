"""Cross-check every PDF deliverable against what landed in the repo.

Exit code 0 if every required artefact is present and every notebook has
executed outputs. Exit 1 otherwise. Also prints a human-readable checklist.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def nb_exec_status(path: pathlib.Path) -> tuple[int, int, int]:
    """Return (total_code_cells, executed_code_cells, cells_with_output)."""
    if not path.exists():
        return (0, 0, 0)
    nb = json.loads(path.read_text())
    total = executed = with_output = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        total += 1
        if c.get("execution_count") is not None:
            executed += 1
        if c.get("outputs"):
            with_output += 1
    return total, executed, with_output


def exists(p: str | pathlib.Path) -> bool:
    return pathlib.Path(ROOT / p).exists()


CHECKS = {
    "repo root files": [
        ("part1.ipynb", True),
        ("part2.ipynb", True),
        ("part3.ipynb", True),
        ("part4.ipynb", True),
        ("part5.ipynb", True),
        ("pipeline.py", True),
        ("requirements.txt", True),
        ("README.md", True),
        (".gitignore", True),
    ],
    "Part 1 outputs": [
        ("outputs/eval_probs_baseline.npy", True),
        ("outputs/part1_threshold_sweep.csv", True),
        ("outputs/part1_roc_pr.png", True),
        ("outputs/chosen_threshold.json", True),
        ("models/baseline/model.safetensors", True),
        ("data/train_100k.parquet", True),
        ("data/eval_20k.parquet", True),
    ],
    "Part 2 outputs": [
        ("outputs/part2_bias_metrics.csv", True),
        ("outputs/part2_grouped_bar.png", True),
        ("outputs/part2_cms.png", True),
        ("outputs/part2_summary.json", True),
    ],
    "Part 3 outputs": [
        ("outputs/part3_attack1_asr.csv", True),
        ("outputs/part3_attack2_comparison.csv", True),
        ("outputs/eval_probs_poisoned.npy", True),
        ("models/poisoned/model.safetensors", True),
    ],
    "Part 4 outputs": [
        ("outputs/part4_comparison.csv", True),
        ("outputs/part4_pareto.csv", True),
        ("outputs/part4_pareto.png", True),
        ("outputs/best_mitigated.json", True),
        ("outputs/eval_probs_reweighed.npy", True),
        ("outputs/eval_probs_oversampled.npy", True),
        ("models/reweighed/model.safetensors", True),
        ("models/oversampled/model.safetensors", True),
    ],
    "Part 5 outputs": [
        ("outputs/part5_layer_counts.csv", True),
        ("outputs/part5_layer_distribution.png", True),
        ("outputs/part5_review_breakdown.png", True),
        ("outputs/part5_threshold_sensitivity.csv", True),
        ("models/calibrator.pkl", True),
    ],
}


def main() -> int:
    ok = True
    for group, items in CHECKS.items():
        print(f"\n=== {group} ===")
        for path, required in items:
            present = exists(path)
            flag = "OK " if present else ("MISSING" if required else "skip")
            if required and not present:
                ok = False
            print(f"  [{flag}] {path}")

    print("\n=== notebook execution state ===")
    for nb in ["part1.ipynb", "part2.ipynb", "part3.ipynb", "part4.ipynb", "part5.ipynb"]:
        t, e, o = nb_exec_status(ROOT / nb)
        ratio = (e / t * 100) if t else 0
        flag = "OK" if (t == e and e == o) else "INCOMPLETE"
        print(f"  [{flag}] {nb}: {e}/{t} executed, {o} with output ({ratio:.0f}% executed)")
        if not (t == e == o):
            ok = False

    # regex pattern-count check from pipeline
    sys.path.insert(0, str(ROOT))
    try:
        from pipeline import BLOCKLIST  # type: ignore
        requirements = {
            "direct_threat": 5,
            "self_harm_directed": 4,
            "doxxing_stalking": 4,
            "dehumanization": 4,
            "coordinated_harassment": 3,
        }
        print("\n=== pipeline.py blocklist ===")
        for cat, req in requirements.items():
            count = len(BLOCKLIST.get(cat, []))
            flag = "OK " if count >= req else "SHORT"
            if count < req:
                ok = False
            print(f"  [{flag}] {cat}: {count} patterns (>= {req} required)")
        total = sum(len(v) for v in BLOCKLIST.values())
        print(f"  total: {total} (>= 20 required) {'OK' if total >= 20 else 'SHORT'}")
        if total < 20:
            ok = False
    except Exception as e:
        print(f"  [MISSING] pipeline.BLOCKLIST not importable: {e}")
        ok = False

    print("\n=== verdict ===")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
