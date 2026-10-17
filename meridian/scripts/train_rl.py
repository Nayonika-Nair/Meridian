"""Train and evaluate the offline advisory bandit. Deterministic for a given --seed.

    python scripts/train_rl.py --seed 0 --train 1500 --eval 400 --out data/rl_model.json --report data/rl_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rl import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--eval", type=int, default=400)
    ap.add_argument("--out", default="data/rl_model.json")
    ap.add_argument("--report", default="data/rl_report.json")
    a = ap.parse_args()
    report, model = run_experiment(a.train, a.eval, a.seed)
    model.save(a.out, meta={"seed": a.seed, "train_sessions": a.train, "synthetic": True})
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{'policy':40s} {'reward':>8s} {'util':>6s} {'cost':>7s} {'block%':>7s} {'review%':>8s} {'viol':>5s}  95% CI")
    for n, r in report["results"].items():
        lo, hi = r["reward_ci95_over_sessions"]
        print(f"{n:40s} {r['mean_reward']:8.3f} {r['mean_utility']:6.3f} {r['mean_cost']:7.2f} {100*r['decisions']['BLOCK']:7.1f} {100*r['decisions']['REVIEW']:8.1f} {r['constraint_violations']:5d}  [{lo:.3f}, {hi:.3f}]")
    print(report["verdict"])


if __name__ == "__main__":
    main()
