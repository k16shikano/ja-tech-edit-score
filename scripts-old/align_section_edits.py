#!/usr/bin/env python3
"""節ペアの文アラインメントと編集プロファイルを計算する。"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from build_hard_eval_v2 import paragraphs, split_heading_body

ALIGN_THRESHOLD = 0.6
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL | re.MULTILINE)
CLOSING_BRACKETS = "」』)]"
SENTENCE_END_RE = re.compile(r"[。！？][" + re.escape(CLOSING_BRACKETS) + r"]*")


@dataclass(frozen=True)
class SentenceRef:
  global_idx: int
  para_idx: int
  sent_idx: int
  text: str


def load_rows(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def split_prose_sentences(text: str) -> list[str]:
  text = text.strip()
  if not text:
    return []
  sentences: list[str] = []
  start = 0
  for match in SENTENCE_END_RE.finditer(text):
    end = match.end()
    chunk = text[start:end].strip()
    if chunk:
      sentences.append(chunk)
    start = end
  tail = text[start:].strip()
  if tail:
    sentences.append(tail)
  return sentences


def paragraph_sentence_units(para: str) -> list[str]:
  """段落を文単位に分割する。コードブロック内は分割しない。"""
  units: list[str] = []
  pos = 0
  for match in CODE_FENCE_RE.finditer(para):
    if match.start() > pos:
      units.extend(split_prose_sentences(para[pos : match.start()]))
    units.append(match.group(0).strip())
    pos = match.end()
  if pos < len(para):
    units.extend(split_prose_sentences(para[pos:]))
  return [u for u in units if u.strip()]


def text_to_sentences(text: str) -> list[SentenceRef]:
  _heading, body = split_heading_body(text)
  refs: list[SentenceRef] = []
  global_idx = 0
  for para_idx, para in enumerate(paragraphs(body)):
    for sent_idx, sent in enumerate(paragraph_sentence_units(para)):
      refs.append(
        SentenceRef(
          global_idx=global_idx,
          para_idx=para_idx,
          sent_idx=sent_idx,
          text=sent,
        )
      )
      global_idx += 1
  return refs


def greedy_align(
  source: list[SentenceRef],
  edited: list[SentenceRef],
  *,
  threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
  candidates: list[tuple[float, int, int]] = []
  for i, s in enumerate(source):
    for j, e in enumerate(edited):
      ratio = SequenceMatcher(None, s.text, e.text).ratio()
      if ratio >= threshold:
        candidates.append((ratio, i, j))
  candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

  used_s: set[int] = set()
  used_e: set[int] = set()
  matches: list[tuple[int, int, float]] = []
  for ratio, i, j in candidates:
    if i in used_s or j in used_e:
      continue
    used_s.add(i)
    used_e.add(j)
    matches.append((i, j, ratio))
  return matches, used_s, used_e


def inversion_rate(source_indices: list[int], edit_indices: list[int]) -> float:
  if len(source_indices) < 2:
    return 0.0
  ordered = sorted(zip(source_indices, edit_indices, strict=True), key=lambda x: x[0])
  edit_order = [e for _, e in ordered]
  inv = 0
  for i in range(len(edit_order)):
    for j in range(i + 1, len(edit_order)):
      if edit_order[i] > edit_order[j]:
        inv += 1
  max_inv = len(edit_order) * (len(edit_order) - 1) // 2
  return inv / max_inv if max_inv else 0.0


def grouping_change_rate(
  source: list[SentenceRef],
  edited: list[SentenceRef],
  matches: list[tuple[int, int, float]],
) -> float:
  if len(matches) < 2:
    return 0.0
  changed = 0
  total = 0
  for a in range(len(matches)):
    si, ei, _ = matches[a]
    for b in range(a + 1, len(matches)):
      sj, ej, _ = matches[b]
      same_src = source[si].para_idx == source[sj].para_idx
      same_edt = edited[ei].para_idx == edited[ej].para_idx
      total += 1
      if same_src != same_edt:
        changed += 1
  return changed / total if total else 0.0


def char_count(refs: list[SentenceRef]) -> int:
  return sum(len(r.text) for r in refs)


def profile_pair(row: dict, *, dataset: str) -> dict:
  source_sents = text_to_sentences(row["source_text"])
  edit_sents = text_to_sentences(row["edited_text"])
  matches, used_s, used_e = greedy_align(source_sents, edit_sents, threshold=ALIGN_THRESHOLD)

  unmatched_source = [i for i in range(len(source_sents)) if i not in used_s]
  unmatched_edit = [i for i in range(len(edit_sents)) if i not in used_e]

  similarities = [ratio for _, _, ratio in matches]
  mean_similarity = statistics.mean(similarities) if similarities else 0.0
  matched_expr = 1.0 - mean_similarity

  src_chars = char_count(source_sents)
  edt_chars = char_count(edit_sents)
  unmatched_src_chars = sum(len(source_sents[i].text) for i in unmatched_source)
  unmatched_edt_chars = sum(len(edit_sents[i].text) for i in unmatched_edit)
  total_chars = max(src_chars + edt_chars, 1)
  unmatched_char_ratio = (unmatched_src_chars + unmatched_edt_chars) / total_chars

  matched_src_chars = sum(len(source_sents[i].text) for i in used_s)
  matched_edt_chars = sum(len(edit_sents[i].text) for i in used_e)
  source_coverage = matched_src_chars / max(src_chars, 1)
  edit_coverage = matched_edt_chars / max(edt_chars, 1)

  # 未対応文は完全な表現変化（1.0）として文字数で重み付け
  expression_change_total = (
    matched_expr * (matched_src_chars + matched_edt_chars)
    + (unmatched_src_chars + unmatched_edt_chars)
  ) / max(src_chars + edt_chars, 1)

  src_para_count = len({r.para_idx for r in source_sents}) if source_sents else 0
  edt_para_count = len({r.para_idx for r in edit_sents}) if edit_sents else 0
  para_den = max(src_para_count, edt_para_count, 1)
  paragraph_count_delta_rate = abs(edt_para_count - src_para_count) / para_den

  order_rate = inversion_rate(
    [m[0] for m in matches],
    [m[1] for m in matches],
  )
  grouping_rate = grouping_change_rate(source_sents, edit_sents, matches)
  structure_overall = max(order_rate, grouping_rate, paragraph_count_delta_rate)

  mapping = [
    {
      "source_global": source_sents[si].global_idx,
      "edit_global": edit_sents[ej].global_idx,
      "source_para": source_sents[si].para_idx,
      "edit_para": edit_sents[ej].para_idx,
      "similarity": round(ratio, 4),
    }
    for si, ej, ratio in sorted(matches, key=lambda x: source_sents[x[0]].global_idx)
  ]

  meta = row.get("meta", {})
  return {
    "id": row["id"],
    "project_id": row.get("project_id"),
    "dataset": dataset,
    "source_text": row["source_text"],
    "edited_text": row["edited_text"],
    "meta": meta,
    "alignment": {
      "matched_count": len(matches),
      "added_count": len(unmatched_edit),
      "deleted_count": len(unmatched_source),
      "mean_similarity": round(mean_similarity, 4),
      "coverage": {
        "source": round(source_coverage, 4),
        "edited": round(edit_coverage, 4),
        "min": round(min(source_coverage, edit_coverage), 4),
      },
      "mapping": mapping,
    },
    "expression_change": {
      "matched": round(matched_expr, 4),
      "total": round(expression_change_total, 4),
      "unmatched_char_ratio": round(unmatched_char_ratio, 4),
    },
    "structure_change": {
      "order_inversion_rate": round(order_rate, 4),
      "grouping_change_rate": round(grouping_rate, 4),
      "paragraph_count_delta_rate": round(paragraph_count_delta_rate, 4),
      "overall": round(structure_overall, 4),
    },
    "counts": {
      "source_sentences": len(source_sents),
      "edited_sentences": len(edit_sents),
      "source_paragraphs": src_para_count,
      "edited_paragraphs": edt_para_count,
    },
  }


def histogram_table(values: list[float], *, bins: list[float], title: str) -> list[str]:
  counts = [0] * (len(bins) - 1)
  for value in values:
    placed = False
    for i in range(len(bins) - 1):
      lo, hi = bins[i], bins[i + 1]
      if i == len(bins) - 2:
        if lo <= value <= hi:
          counts[i] += 1
          placed = True
          break
      elif lo <= value < hi:
        counts[i] += 1
        placed = True
        break
    if not placed and values:
      counts[-1] += 1
  lines = [f"### {title}", "", "| 区間 | 件数 |", "|---|---:|"]
  for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    label = f"[{lo:.2f}, {hi:.2f}]" if i < len(bins) - 2 else f"[{lo:.2f}, {hi:.2f}]"
    lines.append(f"| {label} | {counts[i]} |")
  lines.append("")
  return lines


def structure_signal(profile: dict) -> float:
  sc = profile["structure_change"]
  return max(
    sc["order_inversion_rate"],
    sc["grouping_change_rate"],
    sc["paragraph_count_delta_rate"],
  )


def preview_examples(profiles: list[dict], *, n: int = 5) -> list[str]:
  eligible = [
    p
    for p in profiles
    if p["alignment"]["matched_count"] >= 2
    and p["expression_change"]["matched"] < 0.15
    and p["alignment"]["coverage"]["source"] >= 0.7
    and p["alignment"]["coverage"]["edited"] >= 0.7
  ]
  ranked = sorted(
    eligible,
    key=lambda p: (structure_signal(p), -p["expression_change"]["matched"]),
    reverse=True,
  )
  lines = [
    "## 代表例（構成変化大・表現変化小）",
    "",
    "expression_change.matched < 0.15、coverage ≥ 0.7、structure_signal が大きい順。",
    "",
  ]
  seen: set[str] = set()
  shown = 0
  for profile in ranked:
    if profile["id"] in seen:
      continue
    seen.add(profile["id"])
    shown += 1
    if shown > n:
      break
    lines.extend(
      [
        f"### {profile['id']}",
        "",
        f"- dataset: {profile['dataset']}",
        f"- expression_change.matched: {profile['expression_change']['matched']:.3f}",
        f"- expression_change.total: {profile['expression_change']['total']:.3f}",
        f"- coverage (src/ed/min): "
        f"{profile['alignment']['coverage']['source']:.3f}/"
        f"{profile['alignment']['coverage']['edited']:.3f}/"
        f"{profile['alignment']['coverage']['min']:.3f}",
        f"- structure_change.overall: {profile['structure_change']['overall']:.3f}",
      f"- structure_signal: {structure_signal(profile):.3f} "
      f"(order: {profile['structure_change']['order_inversion_rate']:.3f}, "
      f"grouping: {profile['structure_change']['grouping_change_rate']:.3f}, "
      f"para_delta: {profile['structure_change']['paragraph_count_delta_rate']:.3f})",
        f"- 対応文数: {profile['alignment']['matched_count']} "
        f"(追加 {profile['alignment']['added_count']} / 削除 {profile['alignment']['deleted_count']})",
      ]
    )
    mapping = profile["alignment"]["mapping"]
    src_seq = [m["source_global"] for m in mapping]
    edt_seq = [m["edit_global"] for m in mapping]
    lines.append(f"- 下書き文番号列 → 編集後文番号列: `{src_seq}` → `{edt_seq}`")
    lines.append("")
  return lines


def write_preview(profiles: list[dict], path: Path) -> None:
  expr_vals = [p["expression_change"]["matched"] for p in profiles]
  struct_vals = [p["structure_change"]["overall"] for p in profiles]
  sim_vals = [p["alignment"]["mean_similarity"] for p in profiles if p["alignment"]["matched_count"]]

  lines = [
    "# 節編集プロファイル preview",
    "",
    f"総件数: {len(profiles)}",
    "",
  ]
  lines.extend(histogram_table(expr_vals, bins=[0, 0.05, 0.10, 0.15, 0.25, 0.40, 1.0], title="expression_change.matched"))
  expr_total_vals = [p["expression_change"]["total"] for p in profiles]
  lines.extend(histogram_table(expr_total_vals, bins=[0, 0.05, 0.15, 0.25, 0.40, 0.60, 1.0], title="expression_change.total"))
  cov_vals = [p["alignment"]["coverage"]["min"] for p in profiles if p["alignment"]["matched_count"]]
  if cov_vals:
    lines.extend(histogram_table(cov_vals, bins=[0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0], title="alignment coverage (min)"))
  lines.extend(histogram_table(struct_vals, bins=[0, 0.05, 0.10, 0.20, 0.35, 0.55, 1.0], title="structure_change.overall"))
  if sim_vals:
    lines.extend(histogram_table(sim_vals, bins=[0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0], title="mean_similarity（対応文）"))
  lines.extend(preview_examples(profiles))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--inputs",
    nargs="+",
    default=[
      "data/examples.section.heldout.jsonl",
      "data/examples.section.raw.jsonl",
    ],
    help="節ペア JSONL（複数可）",
  )
  parser.add_argument("--out", default="data/section_edit_profile.jsonl")
  parser.add_argument("--preview", default="data/section_edit_profile_preview.md")
  args = parser.parse_args()

  root = Path(__file__).resolve().parent.parent
  out_path = Path(args.out)
  if not out_path.is_absolute():
    out_path = root / out_path
  preview_path = Path(args.preview)
  if not preview_path.is_absolute():
    preview_path = root / preview_path

  profiles: list[dict] = []
  for input_arg in args.inputs:
    input_path = Path(input_arg)
    if not input_path.is_absolute():
      input_path = root / input_path
    if not input_path.exists():
      print(f"[skip] missing: {input_path}", file=sys.stderr)
      continue
    dataset = "heldout" if "heldout" in input_path.name else "raw"
    rows = load_rows(input_path)
    for row in rows:
      profiles.append(profile_pair(row, dataset=dataset))
    print(f"{input_path.name}: {len(rows)} rows ({dataset})")

  if not profiles:
    raise SystemExit("no profiles")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as handle:
    for profile in profiles:
      handle.write(json.dumps(profile, ensure_ascii=False) + "\n")
  write_preview(profiles, preview_path)

  print("-" * 64)
  print(f"wrote {out_path} ({len(profiles)} profiles)")
  print(f"wrote {preview_path}")


if __name__ == "__main__":
  main()
