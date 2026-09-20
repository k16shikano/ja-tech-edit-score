#!/usr/bin/env python3
"""editor-as-distribution Phase 0 共通。"""
from __future__ import annotations

import difflib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

POSITION_RANK = {"a": 0, "b": 1, "c": 2, "d": 3}
LABEL_ORDER = ("a", "b", "c", "d")


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def ead_out() -> Path:
  return repo_root() / "outputs" / "ead"


def ead_reports() -> Path:
  return ead_out() / "reports"


def ead_work() -> Path:
  return ead_out() / "work"


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def write_json(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tau_level_work_path() -> Path:
  return ead_work() / "tau_level.json"


def checkpoint_identifier(path: Path, *, root: Path | None = None) -> str:
  resolved = path.resolve()
  base = (root or repo_root()).resolve()
  try:
    return str(resolved.relative_to(base))
  except ValueError:
    return str(resolved)


def write_tau_level_record(
  *,
  tau_level: float,
  gate_level_checkpoint: Path | str,
  root: Path | None = None,
  den_adapter_checkpoint: Path | str | None = None,
  calibration_sets: list[str] | None = None,
  method: str | None = None,
  metrics: dict[str, Any] | None = None,
) -> Path:
  base = root or repo_root()
  record: dict[str, Any] = {
    "tau_level": float(tau_level),
    "gate_level_checkpoint": checkpoint_identifier(Path(gate_level_checkpoint), root=base),
  }
  if den_adapter_checkpoint is not None:
    record["den_adapter_checkpoint"] = checkpoint_identifier(Path(den_adapter_checkpoint), root=base)
  if calibration_sets is not None:
    record["calibration_sets"] = list(calibration_sets)
  if method is not None:
    record["calibration_method"] = method
  if metrics is not None:
    record["calibration_metrics"] = metrics
  out = tau_level_work_path()
  write_json(out, record)
  return out


def load_tau_level_record() -> dict[str, Any] | None:
  path = tau_level_work_path()
  if not path.is_file():
    return None
  return json.loads(path.read_text(encoding="utf-8"))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n <= 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1.0 + z * z / n
  centre = p + z * z / (2.0 * n)
  margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
  return ((centre - margin) / denom, (centre + margin) / denom)


def rate_summary(k: int, n: int) -> dict[str, Any]:
  lo, hi = wilson_ci(k, n)
  return {
    "k": k,
    "n": n,
    "rate": (k / n) if n else None,
    "wilson_95_lower": lo,
    "wilson_95_upper": hi,
  }


def id_from_row(row: dict) -> str | None:
  if row.get("id"):
    return str(row["id"])
  if row.get("item_id"):
    return str(row["item_id"])
  meta = row.get("meta") or {}
  if meta.get("id"):
    return str(meta["id"])
  if meta.get("item_id"):
    return str(meta["item_id"])
  return None


def draft_from_row(row: dict) -> str | None:
  for key in ("draft", "source_text", "context_draft"):
    val = row.get(key)
    if val is not None and str(val):
      return str(val)
  messages = row.get("messages")
  if isinstance(messages, list) and messages:
    content = str(messages[0].get("content") or "")
    for prefix in (
      "次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ。",
    ):
      if content.startswith(prefix):
        parts = content.split("\n\n", 1)
        if len(parts) == 2:
          return parts[1]
    if "\n\n" in content:
      return content.split("\n\n", 1)[1]
    return content
  return None


def ids_from_edit_sft(rows: list[dict]) -> set[str]:
  out: set[str] = set()
  for row in rows:
    item_id = id_from_row(row)
    if item_id:
      out.add(item_id)
  return out


def drafts_from_pref(rows: list[dict]) -> set[str]:
  out: set[str] = set()
  for row in rows:
    draft = draft_from_row(row)
    if draft:
      out.add(draft)
  return out


def schema_stats(rows: list[dict], *, name: str) -> dict[str, Any]:
  if not rows:
    return {"name": name, "n": 0, "keys": [], "char_len": {}}
  keys = sorted({k for row in rows for k in row.keys()})
  draft_lens: list[int] = []
  cand_lens: list[int] = []
  for row in rows:
    draft = draft_from_row(row)
    if draft is not None:
      draft_lens.append(len(draft))
    for key in ("y", "gold", "edited_text", "candidate_a", "candidate_b", "generated"):
      val = row.get(key)
      if val is not None:
        cand_lens.append(len(str(val)))
        break
  def summarize(vals: list[int]) -> dict[str, float | int | None]:
    if not vals:
      return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
      "n": len(vals),
      "min": min(vals),
      "max": max(vals),
      "mean": statistics.mean(vals),
      "median": statistics.median(vals),
    }
  return {
    "name": name,
    "n": len(rows),
    "keys": keys,
    "char_len": {
      "draft": summarize(draft_lens),
      "candidate": summarize(cand_lens),
    },
  }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
  ]
  for row in rows:
    cells = [str(c) for c in row]
    lines.append("| " + " | ".join(cells) + " |")
  return "\n".join(lines)


def fmt_rate(stat: dict[str, Any]) -> str:
  if not stat.get("n"):
    return "—"
  rate = stat.get("rate")
  lo = stat.get("wilson_95_lower")
  hi = stat.get("wilson_95_upper")
  if rate is None:
    return "—"
  return f"{stat['k']}/{stat['n']} = {rate:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_pct(x: float | None) -> str:
  if x is None or math.isnan(x):
    return "—"
  return f"{x:.3f}"


def rps_ordinal(y_true: np.ndarray, probs: np.ndarray) -> float:
  """4値順序ラベル（0..K-1）の Ranked Probability Score。K-1 個の累積確率を事例ごとに比較する。"""
  import numpy as _np

  y = _np.asarray(y_true, dtype=_np.int64)
  p = _np.asarray(probs, dtype=_np.float64)
  if y.size == 0:
    return float("nan")
  n, k = p.shape
  if k < 2:
    return float("nan")
  scores: list[float] = []
  for i in range(n):
    cum_pred = _np.cumsum(p[i])[: k - 1]
    cum_obs = _np.array([1.0 if int(y[i]) <= j else 0.0 for j in range(k - 1)])
    scores.append(float(_np.mean((cum_pred - cum_obs) ** 2)))
  return float(_np.mean(scores))


def rps_sanity_baselines(y_valid: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
  import numpy as _np

  k = int(max(int(y_valid.max(initial=0)), int(y_train.max(initial=0))) + 1)
  uniform = _np.full((len(y_valid), k), 1.0 / k)
  counts = _np.bincount(_np.asarray(y_train, dtype=_np.int64), minlength=k)
  marginal = counts / max(int(counts.sum()), 1)
  marginal_probs = _np.tile(marginal, (len(y_valid), 1))
  return {
    "uniform": rps_ordinal(y_valid, uniform),
    "marginal_train": rps_ordinal(y_valid, marginal_probs),
  }


def normalized_levenshtein(a: str, b: str) -> float:
  if a == b:
    return 0.0
  if not a or not b:
    return 1.0
  return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def split_sentences(text: str) -> list[str]:
  parts = re.split(r"[。！？\n]+", text)
  return [p for p in parts if p.strip()]


def char_class_rates(text: str) -> tuple[float, float, float]:
  if not text:
    return (0.0, 0.0, 0.0)
  n = len(text)
  kanji = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
  hira = sum(1 for ch in text if "\u3040" <= ch <= "\u309f")
  punct = sum(1 for ch in text if ch in "、。，．！？…「」『』（）()[]【】")
  return (kanji / n, hira / n, punct / n)


def type_token_ratio(text: str) -> float:
  if not text:
    return 0.0
  chars = list(text)
  return len(set(chars)) / len(chars)


def surface_features(draft: str, cand: str) -> dict[str, float]:
  delta = len(cand) - len(draft)
  ratio = len(cand) / max(len(draft), 1)
  dsents = split_sentences(draft)
  csents = split_sentences(cand)
  d_lens = [len(s) for s in dsents] or [0]
  c_lens = [len(s) for s in csents] or [0]
  d_kanji, d_hira, d_punct = char_class_rates(draft)
  c_kanji, c_hira, c_punct = char_class_rates(cand)
  return {
    "delta_chars": float(delta),
    "abs_delta_chars": float(abs(delta)),
    "char_ratio": float(ratio),
    "edit_distance_norm": float(normalized_levenshtein(draft, cand)),
    "sent_count_diff": float(len(csents) - len(dsents)),
    "avg_sent_len_diff": float(statistics.mean(c_lens) - statistics.mean(d_lens)),
    "std_sent_len_diff": float(
      (statistics.pstdev(c_lens) if len(c_lens) > 1 else 0.0)
      - (statistics.pstdev(d_lens) if len(d_lens) > 1 else 0.0)
    ),
    "ttr_diff": float(type_token_ratio(cand) - type_token_ratio(draft)),
    "punct_density_diff": float(c_punct - d_punct),
    "kanji_rate_diff": float(c_kanji - d_kanji),
    "hiragana_rate_diff": float(c_hira - d_hira),
  }


def resolve_d_model_dir(cv_dir: Path, fold: int = 0) -> Path:
  best = cv_dir / f"fold{fold}" / "best"
  if not best.is_dir():
    raise FileNotFoundError(best)
  if (best / "meta.json").is_file() or (best / "config.json").is_file():
    return best
  raise FileNotFoundError(best)


FEATURE_NAMES = [
  "delta_chars",
  "abs_delta_chars",
  "char_ratio",
  "edit_distance_norm",
  "sent_count_diff",
  "avg_sent_len_diff",
  "std_sent_len_diff",
  "ttr_diff",
  "punct_density_diff",
  "kanji_rate_diff",
  "hiragana_rate_diff",
]
