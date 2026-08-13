#!/usr/bin/env python3
"""生成の機械計測と整合性判定。

用途:
1. モジュール: 構成要素（行頭 # 見出し、【図】【コード】）の保持判定。
   ブラインドペア構築の選抜前フィルタが import する。
2. CLI: 生成 jsonl（draft / generated / gold を持つ行）を受け取り、
   無編集率・構成要素保持・文中断率を表と JSON で出す。
   下書き↔生成の類似度は無編集の補助指標。下書き↔gold は記述統計（目標値・選好の代理ではない）。
   チェックポイント選定と、人手判定前の機械ゲートの両方に使う。

判定に使うのは数の保存（見出しの本数、プレースホルダの個数）だけ。
文言の変更や位置の移動は推敲の範囲なので違反にしない。
文体（ですます・である）はヒューリスティックでしか測れないため、
フィルタには使わず、CLI の参考列にだけ出す。
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path

HEADING_RE = re.compile(r"^\s*#{1,6}\s")
PLACEHOLDERS = ("【コード】", "【図】")


def heading_count(text: str) -> int:
  return sum(1 for line in text.splitlines() if HEADING_RE.match(line))


def placeholders_preserved(draft: str, generated: str) -> bool:
  return all(generated.count(p) >= draft.count(p) for p in PLACEHOLDERS)


def headings_preserved(draft: str, generated: str) -> bool:
  return heading_count(generated) >= heading_count(draft)


def structure_preserved(draft: str, generated: str) -> bool:
  """選抜前フィルタの判定。見出し本数とプレースホルダ個数を保っているか。"""
  return headings_preserved(draft, generated) and placeholders_preserved(draft, generated)


def _polite_ratio(text: str) -> float | None:
  sents = [s.strip() for s in re.split(r"[。！？]\s*", text) if s.strip()]
  if not sents:
    return None
  pol = sum(
    1 for s in sents if re.search(r"(です|ます|ません|でした|ましょう|ください)$", s)
  )
  return pol / len(sents)


def register_flip_suspect(draft: str, generated: str) -> bool:
  """文体反転の疑い（参考値）。文末表現の割合による推定で、誤検出がある。"""
  rd, rg = _polite_ratio(draft), _polite_ratio(generated)
  if rd is None or rg is None:
    return False
  return (rd >= 0.5) != (rg >= 0.5)


def truncation_suspect(generated: str) -> bool:
  """文の途中で切れている疑い。生成上限に当たった出力の検出用。"""
  g = generated.rstrip()
  if not g:
    return False
  last = g.splitlines()[-1].strip()
  end_ok = (
    g.endswith(("。", "」", "】", ")", "）", "！", "？", "…", "．"))
    or last.startswith("#")
    or last.endswith(("|", "```", "---"))
  )
  return not end_ok


def similarity(a: str, b: str) -> float:
  return difflib.SequenceMatcher(None, a, b).ratio()


def summarize_file(path: Path) -> dict:
  rows = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  if not rows:
    return {"file": str(path), "n": 0}

  sims = []
  gold_sims = []
  copies = 0
  head_ng = 0
  ph_ng = 0
  flip = 0
  trunc = 0
  len_ratios = []
  for r in rows:
    d, g = r["draft"], r["generated"]
    s = similarity(d, g)
    sims.append(s)
    if g.strip() == d.strip():
      copies += 1
    if not headings_preserved(d, g):
      head_ng += 1
    if not placeholders_preserved(d, g):
      ph_ng += 1
    if register_flip_suspect(d, g):
      flip += 1
    if truncation_suspect(g):
      trunc += 1
    if d:
      len_ratios.append(len(g) / len(d))
    if r.get("gold"):
      gold_sims.append(similarity(d, r["gold"]))

  n = len(rows)
  out = {
    "file": str(path),
    "n": n,
    "copy_rate": copies / n,
    "sim_median": statistics.median(sims),
    "sim_ge_095": sum(1 for s in sims if s >= 0.95) / n,
    "heading_loss_rate": head_ng / n,
    "placeholder_loss_rate": ph_ng / n,
    "truncation_suspect_rate": trunc / n,
    "register_flip_suspect_rate": flip / n,
    "len_ratio_median": statistics.median(len_ratios) if len_ratios else None,
  }
  if gold_sims:
    out["gold_sim_median"] = statistics.median(gold_sims)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("files", nargs="+", help="生成 jsonl（draft/generated/gold）")
  parser.add_argument("--out", default="", help="サマリ JSON の出力先")
  args = parser.parse_args()

  results = [summarize_file(Path(p)) for p in args.files]

  cols = [
    ("n", "件数"),
    ("copy_rate", "無編集率"),
    ("sim_median", "下書き↔生成 類似度中央値（高=ほぼ無編集）"),
    ("sim_ge_095", "下書き↔生成 類似度>=0.95"),
    ("heading_loss_rate", "見出し欠落率"),
    ("placeholder_loss_rate", "図/コード欠落率"),
    ("truncation_suspect_rate", "文中断率"),
    ("register_flip_suspect_rate", "文体反転疑い(参考)"),
    ("gold_sim_median", "下書き↔gold 類似度中央値（参照・目標値ではない）"),
  ]
  for r in results:
    print(f"== {r['file']}")
    for key, label in cols:
      v = r.get(key)
      if v is None:
        continue
      if isinstance(v, float):
        print(f"  {label}: {v:.3f}")
      else:
        print(f"  {label}: {v}")

  if args.out:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
      json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
