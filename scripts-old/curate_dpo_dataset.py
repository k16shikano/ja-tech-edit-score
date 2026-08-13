#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import re
from pathlib import Path


JP_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ONLY_RULE_RE = re.compile(r"^[=\-~^`_*:#.\s]+$")
DIRECTIVE_RE = re.compile(r"^\.\.\s+\S+")
PLACEHOLDER_RE = re.compile(
  r"(要確認|原著者に確認|著者に確認|真意を.*確認|念の為.*確認|と思われるが|確認する必要|確認が必要)"
)
AUTHOR_DATE_RE = re.compile(r"^[A-Z][A-Za-z .'\-]+,\s+[A-Z][A-Za-z]+\s+\d{4}$")
CITATION_NUM_RE = re.compile(r"\[\d+\]")
BIB_KEY_RE = re.compile(r"\[@(?:book|article|inproceedings|misc):[^\]]+\]")


def normalize(text: str) -> str:
  return re.sub(r"\s+", "", text)


def looks_usable(text: str, min_chars: int, max_chars: int) -> bool:
  stripped = text.strip()
  if len(stripped) < min_chars or len(stripped) > max_chars:
    return False
  return True


def strip_reference_block(text: str) -> str:
  marker = "\n\n[参照]\n"
  if marker in text:
    return text.split(marker, 1)[0].strip()
  return text.strip()


def has_natural_language(text: str) -> bool:
  stripped = strip_reference_block(text)
  if JP_RE.search(stripped):
    return True
  return len(EN_WORD_RE.findall(stripped)) >= 4


def is_directive_only(text: str) -> bool:
  lines = [line.strip() for line in strip_reference_block(text).splitlines() if line.strip()]
  if not lines:
    return True
  for line in lines:
    if ONLY_RULE_RE.fullmatch(line):
      continue
    if DIRECTIVE_RE.match(line):
      continue
    return False
  return True


def is_short_fragment(text: str) -> bool:
  stripped = strip_reference_block(text)
  lines = [line.strip() for line in stripped.splitlines() if line.strip()]
  if len(lines) != 1:
    return False
  line = lines[0]
  if AUTHOR_DATE_RE.fullmatch(line):
    return True
  if line.startswith(("-", "*")) and not JP_RE.search(line):
    return True
  if not JP_RE.search(line) and len(EN_WORD_RE.findall(line)) <= 3 and len(normalize(line)) < 40:
    return True
  return len(normalize(line)) < 12


def strip_trivial_markup(text: str) -> str:
  stripped = strip_reference_block(text)
  stripped = CITATION_NUM_RE.sub("", stripped)
  stripped = BIB_KEY_RE.sub("", stripped)
  return normalize(stripped)


def is_citation_or_bib_only_diff(a: str, b: str) -> bool:
  if normalize(a) == normalize(b):
    return False
  return strip_trivial_markup(a) == strip_trivial_markup(b)


def bad_length_ratio(inp: str, chosen: str, rejected: str) -> bool:
  src_len = max(len(normalize(strip_reference_block(inp))), 1)
  chosen_len = max(len(normalize(chosen)), 1)
  rejected_len = max(len(normalize(rejected)), 1)
  ratios = [chosen_len / src_len, rejected_len / src_len]
  return any(r > 6.0 or r < 0.15 for r in ratios)


def reject_reason(
  rec: dict,
  min_chars: int,
  max_chars: int,
  exclude_res: list[re.Pattern[str]],
  drop_citation_only: bool = False,
) -> str | None:
  meta = rec.get("meta", {})
  inp = rec.get("input", "")
  chosen = rec.get("chosen", "")
  rejected = rec.get("rejected", "")

  if not all(looks_usable(text, min_chars, max_chars) for text in (inp, chosen, rejected)):
    return "bad_length"

  if normalize(chosen) == normalize(rejected):
    return "same_choice"

  if drop_citation_only and is_citation_or_bib_only_diff(chosen, rejected):
    return "citation_or_bib_only"

  blob = "\n".join([inp, chosen, rejected, meta.get("source_reference", "")])
  if any(pattern.search(blob) for pattern in exclude_res):
    return "user_exclude_pattern"

  if PLACEHOLDER_RE.search(blob):
    return "placeholder_note"

  if is_directive_only(inp) or is_directive_only(chosen) or is_directive_only(rejected):
    return "directive_only"

  if is_short_fragment(inp) or is_short_fragment(chosen) or is_short_fragment(rejected):
    return "short_fragment"

  if not JP_RE.search(chosen) and not JP_RE.search(rejected):
    return "no_japanese_target"

  if not has_natural_language(inp) or not has_natural_language(chosen) or not has_natural_language(rejected):
    return "non_paragraph"

  if bad_length_ratio(inp, chosen, rejected):
    return "bad_length_ratio"

  return None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, help="input DPO jsonl")
  parser.add_argument("--out", required=True, help="output curated DPO jsonl")
  parser.add_argument("--project-id", default="", help="optional project id filter")
  parser.add_argument("--min-chars", type=int, default=5)
  parser.add_argument("--max-chars", type=int, default=2500)
  parser.add_argument(
    "--exclude-pattern",
    action="append",
    default=[],
    help="regex pattern to exclude if matched in input/chosen/rejected",
  )
  parser.add_argument(
    "--report",
    default="",
    help="optional JSON report path for kept/skipped counts by reason",
  )
  parser.add_argument(
    "--drop-citation-only",
    action="store_true",
    help="drop pairs where chosen/rejected differ only by citation numbers or bib keys",
  )
  args = parser.parse_args()

  in_path = Path(args.input)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  exclude_res = [re.compile(p) for p in args.exclude_pattern]
  seen = set()
  kept = 0
  skipped = 0
  reasons = Counter()

  with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
    for line in src:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      meta = rec.get("meta", {})

      if args.project_id and meta.get("project_id") != args.project_id:
        skipped += 1
        reasons["project_id_mismatch"] += 1
        continue

      reason = reject_reason(
        rec,
        args.min_chars,
        args.max_chars,
        exclude_res,
        drop_citation_only=args.drop_citation_only,
      )
      if reason:
        skipped += 1
        reasons[reason] += 1
        continue

      inp = rec.get("input", "")
      chosen = rec.get("chosen", "")
      rejected = rec.get("rejected", "")
      key = (normalize(inp), normalize(chosen), normalize(rejected))
      if key in seen:
        skipped += 1
        reasons["duplicate"] += 1
        continue
      seen.add(key)

      dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
      kept += 1

  print(f"kept: {kept}")
  print(f"skipped: {skipped}")
  for reason, count in sorted(reasons.items()):
    print(f"{reason}: {count}")

  if args.report:
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
      "input": str(in_path),
      "output": str(out_path),
      "kept": kept,
      "skipped": skipped,
      "reasons": dict(sorted(reasons.items())),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
