#!/usr/bin/env python3
import json
from pathlib import Path


def fold_weak(folds: list[dict], n: int = 3) -> list[dict]:
  return sorted(
    [{"project_id": f["project_id"], "accuracy": f["accuracy"]} for f in folds],
    key=lambda x: x["accuracy"],
  )[:n]


def main() -> None:
  rows: list[dict] = []
  base = Path("outputs/eval_xproject.json")
  if base.exists():
    r = json.loads(base.read_text(encoding="utf-8"))
    rows.append(
      {
        "name": "hotchpotch/static-embedding-japanese",
        "truncate_dim": r.get("truncate_dim"),
        "text_prefix": r.get("text_prefix", ""),
        "max_seq_length": r.get("max_seq_length"),
        "micro_accuracy": r["micro_accuracy"],
        "macro_accuracy": r["macro_accuracy"],
        "micro_log_loss": r["micro_log_loss"],
        "source": str(base),
        "weak_folds": fold_weak(r.get("folds", [])),
      }
    )

  out = Path("outputs/eval_xproject")
  for path in sorted(out.glob("*.json")):
    if path.name == "summary.json":
      continue
    r = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
      {
        "name": r["embedding_model"],
        "truncate_dim": r.get("truncate_dim"),
        "text_prefix": r.get("text_prefix", ""),
        "max_seq_length": r.get("max_seq_length"),
        "micro_accuracy": r["micro_accuracy"],
        "macro_accuracy": r["macro_accuracy"],
        "micro_log_loss": r["micro_log_loss"],
        "source": str(path),
        "weak_folds": fold_weak(r["folds"]),
      }
    )

  by = {row["name"]: row for row in rows}
  rows = sorted(by.values(), key=lambda x: -x["micro_accuracy"])
  print(f"{'model':48s} {'micro':>7} {'macro':>7} {'lloss':>7}")
  for row in rows:
    print(
      f"{row['name']:48s} {row['micro_accuracy']:7.4f} "
      f"{row['macro_accuracy']:7.4f} {row['micro_log_loss']:7.4f}"
    )
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(
    json.dumps({"models": rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print("wrote outputs/eval_xproject/summary.json")


if __name__ == "__main__":
  main()
