#!/usr/bin/env python3
"""難試験 v2（節単位・構成軸）の項目を held-out 節ペアから生成する。

方針（ハイブリッド C）:
  - base_text  = 下書き節（またはその段落ウィンドウ）
  - human 候補 = 編集者の実編集（同ウィンドウ）
  - base 候補  = 下書きのまま（identity / copy）
  - deg-* 候補 = human に制御された構成改悪を加えたもの

改悪の種類:
  deg-join    段落境界（空行）を全部落とし、一塊にする
  deg-split   一文ごとに段落化し、過剰分割する
  deg-reverse 本文段落の順序を逆転する

順位は定義により固定（人手並べ替え不要）:
  human > deg-join > deg-split > base > deg-reverse

CE の max_length=512 に収まるよう、全候補について
tokenizer で (base_text, candidate) のトークン長を実測し、
溢れる項目は落とす。節全体が長い場合は、段落数が一致する
節から連続 N 段落のウィンドウを切り出す。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HEADING_RE = re.compile(r"^#{1,6}\s")
MARKERS = ("```", "~~~", "|--", "af://", "★", "<!--", "TODO", "FIXME")

DEGRADATIONS = []  # filled after deg_* defs
GOLD_RANK = ["human", "deg-join", "deg-split", "base", "deg-reverse"]


def load_rows(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def split_heading_body(text: str) -> tuple[str, str]:
  lines = text.splitlines()
  if lines and HEADING_RE.match(lines[0]):
    return lines[0], "\n".join(lines[1:]).strip("\n")
  return "", text.strip("\n")


def paragraphs(body: str) -> list[str]:
  # markdown_sections.paragraph_count と同じ分割（\n\n）。
  # \s* を含む正規表現は空マッチで文字単位に割れる危険があるので使わない。
  return [p.strip("\n") for p in body.split("\n\n") if p.strip()]


def join_with_heading(heading: str, body: str) -> str:
  if heading:
    return heading + "\n\n" + body.strip("\n") + "\n"
  return body.strip("\n") + "\n"


def deg_join(text: str) -> str:
  heading, body = split_heading_body(text)
  return join_with_heading(heading, "\n".join(paragraphs(body)))


def deg_split(text: str) -> str:
  heading, body = split_heading_body(text)
  lines = [line for line in body.splitlines() if line.strip()]
  return join_with_heading(heading, "\n\n".join(lines))


def deg_reverse(text: str) -> str:
  heading, body = split_heading_body(text)
  return join_with_heading(heading, "\n\n".join(reversed(paragraphs(body))))


DEGRADATIONS = [
  ("deg-join", deg_join),
  ("deg-split", deg_split),
  ("deg-reverse", deg_reverse),
]


def has_markers(text: str) -> bool:
  return any(m in text for m in MARKERS)


def make_window(heading: str, paras: list[str], start: int, n: int) -> str:
  return join_with_heading(heading, "\n\n".join(paras[start : start + n]))


def candidate_map(base_text: str, human_text: str) -> dict[str, dict] | None:
  by_id = {
    "human": {
      "id": "human",
      "text": human_text,
      "generator": "human",
      "prompt_tag": "real-edit",
    },
    "base": {
      "id": "base",
      "text": base_text,
      "generator": "copy",
      "prompt_tag": "identity",
    },
  }
  seen = {human_text.strip(), base_text.strip()}
  for deg_id, fn in DEGRADATIONS:
    deg_text = fn(human_text)
    if deg_text.strip() in seen:
      return None
    seen.add(deg_text.strip())
    by_id[deg_id] = {
      "id": deg_id,
      "text": deg_text,
      "generator": "script",
      "prompt_tag": deg_id,
    }
  return by_id


def fits_tokens(
  tokenizer,
  base_text: str,
  by_id: dict[str, dict],
  *,
  max_tokens: int,
) -> bool:
  for cand in by_id.values():
    n = len(
      tokenizer(
        base_text,
        cand["text"],
        truncation=False,
        add_special_tokens=True,
      )["input_ids"]
    )
    if n > max_tokens:
      return False
  return True


def build_item_from_texts(
  *,
  idx: int,
  row: dict,
  base_text: str,
  human_text: str,
  window_meta: dict,
) -> dict | None:
  by_id = candidate_map(base_text, human_text)
  if by_id is None:
    return None
  candidates = [by_id[cid] for cid in GOLD_RANK]
  meta = row["meta"]
  hs, _ = split_heading_body(human_text)
  bs, _ = split_heading_body(base_text)
  return {
    "id": f"he-v2-{idx:02d}",
    "seed_text": "",
    "seed_meta": {
      "project_id": row["project_id"],
      "path": meta["path"],
      "section_key": meta["section_key"],
      "merged_branch": meta.get("merged_branch", ""),
      "source_reference": row["source_reference"],
      "paragraph_count_source": len(paragraphs(split_heading_body(base_text)[1])),
      "paragraph_count_edited": len(paragraphs(split_heading_body(human_text)[1])),
      "section_paragraph_count_source": meta["paragraph_count_source"],
      "section_paragraph_count_edited": meta["paragraph_count_edited"],
      "note": "held-out real edit; token-budgeted window or section",
      "gold_rank_rule": "human > deg-join > deg-split > base > deg-reverse",
      **window_meta,
    },
    "base_text": base_text,
    "base_generator": "draft",
    "candidates": candidates,
    "human": {
      "best_id": GOLD_RANK[0],
      "rank": list(GOLD_RANK),
      "notes": "definitional rank for controlled degradations; no human reorder",
    },
    "status": "labeled",
  }


def expand_candidates(
  row: dict,
  *,
  window_paragraphs: int,
) -> list[tuple[str, str, dict]]:
  """(base_text, human_text, window_meta) の候補を列挙。"""
  src = row["source_text"].strip("\n") + "\n"
  edt = row["edited_text"].strip("\n") + "\n"
  if has_markers(src) or has_markers(edt):
    return []

  out: list[tuple[str, str, dict]] = []
  out.append(
    (
      src,
      edt,
      {
        "unit": "section",
        "window_start": 0,
        "window_paragraphs": row["meta"]["paragraph_count_edited"],
      },
    )
  )

  hs, src_body = split_heading_body(src)
  he, edt_body = split_heading_body(edt)
  sp = paragraphs(src_body)
  ep = paragraphs(edt_body)
  # 文字単位に割けた場合のガード（平均段落長が極端に短い）
  if not sp or not ep:
    return out
  if sum(len(p) for p in sp) / len(sp) < 20 or sum(len(p) for p in ep) / len(ep) < 20:
    return out
  max_start = min(len(sp), len(ep)) - window_paragraphs
  if max_start >= 0:
    for start in range(0, max_start + 1):
      out.append(
        (
          make_window(hs, sp, start, window_paragraphs),
          make_window(he, ep, start, window_paragraphs),
          {
            "unit": "window",
            "window_start": start,
            "window_paragraphs": window_paragraphs,
            "aligned_equal_count": len(sp) == len(ep),
          },
        )
      )
  return out


def selection_key(item: dict) -> tuple:
  meta = item["seed_meta"]
  # 段落数が変わる節由来を優先。ウィンドウより節全体を優先。
  section_delta = abs(
    meta["section_paragraph_count_edited"] - meta["section_paragraph_count_source"]
  )
  unit_rank = 0 if meta.get("unit") == "section" else 1
  return (-section_delta, unit_rank, len(item["base_text"]))


def write_preview(items: list[dict], path: Path) -> None:
  lines: list[str] = [
    "# Hard Eval v2 candidates preview",
    "",
    "順位は定義により固定: `human > deg-join > deg-split > base > deg-reverse`",
    "全候補は CE `max_length` に収まるようトークン実測で選抜済み。",
    "",
  ]
  for item in items:
    seed_meta = item["seed_meta"]
    unit = seed_meta.get("unit", "section")
    lines.append(f"## {item['id']}")
    lines.append("")
    lines.append(
      f"（{seed_meta['project_id']} / {seed_meta['section_key']} / "
      f"{unit} 段落 {seed_meta['paragraph_count_source']}→"
      f"{seed_meta['paragraph_count_edited']}）"
    )
    lines.append("")
    for cand in item["candidates"]:
      lines.append(f"### Candidate: `{cand['id']}` ({cand.get('generator', '?')})")
      lines.append("")
      lines.append(cand["text"].strip())
      lines.append("")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="held-out 節ペア JSONL")
  parser.add_argument("--out", required=True, help="labeled JSONL の出力先")
  parser.add_argument("--preview", required=True, help="参照用 Markdown の出力先")
  parser.add_argument("--n-items", type=int, default=24)
  parser.add_argument("--min-paragraphs", type=int, default=3)
  parser.add_argument("--window-paragraphs", type=int, default=3)
  parser.add_argument("--max-per-project", type=int, default=8)
  parser.add_argument(
    "--max-tokens",
    type=int,
    default=512,
    help="(base, candidate) の最大トークン長（CE max_length に合わせる）",
  )
  parser.add_argument(
    "--tokenizer",
    default="outputs/pref-ce-beyond-para/model",
    help="トークン長実測用の tokenizer ディレクトリまたは HF id",
  )
  args = parser.parse_args()

  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  rows = load_rows(Path(args.input))
  # 同一節の重複は段落変化が大きいものを残す
  by_section: dict[tuple, dict] = {}
  for row in rows:
    if has_markers(row["source_text"]) or has_markers(row["edited_text"]):
      continue
    if row["meta"]["paragraph_count_edited"] < args.min_paragraphs:
      continue
    if row["meta"]["paragraph_count_source"] < args.min_paragraphs:
      continue
    key = (row["project_id"], row["meta"]["path"], row["meta"]["section_key"])
    prev = by_section.get(key)
    if prev is None:
      by_section[key] = row
      continue
    prev_delta = abs(
      prev["meta"]["paragraph_count_edited"] - prev["meta"]["paragraph_count_source"]
    )
    cur_delta = abs(
      row["meta"]["paragraph_count_edited"] - row["meta"]["paragraph_count_source"]
    )
    if cur_delta > prev_delta or (
      cur_delta == prev_delta and len(row["edited_text"]) < len(prev["edited_text"])
    ):
      by_section[key] = row

  pool: list[dict] = []
  n_overflow = 0
  for row in by_section.values():
    for base_text, human_text, window_meta in expand_candidates(
      row, window_paragraphs=args.window_paragraphs
    ):
      if len(paragraphs(split_heading_body(human_text)[1])) < args.min_paragraphs:
        continue
      by_id = candidate_map(base_text, human_text)
      if by_id is None:
        continue
      if not fits_tokens(tokenizer, base_text, by_id, max_tokens=args.max_tokens):
        n_overflow += 1
        continue
      item = build_item_from_texts(
        idx=0,
        row=row,
        base_text=base_text,
        human_text=human_text,
        window_meta=window_meta,
      )
      if item is not None:
        pool.append(item)

  ranked = sorted(pool, key=selection_key)
  picked: list[dict] = []
  per_project: dict[str, int] = defaultdict(int)
  seen_windows: set[tuple] = set()
  for item in ranked:
    if len(picked) >= args.n_items:
      break
    pid = item["seed_meta"]["project_id"]
    if per_project[pid] >= args.max_per_project:
      continue
    # 同一節から複数ウィンドウを取りすぎない（開始位置の重複抑制）
    wkey = (
      pid,
      item["seed_meta"]["section_key"],
      item["seed_meta"].get("window_start"),
      item["seed_meta"].get("window_paragraphs"),
    )
    if wkey in seen_windows:
      continue
    # 同じ節の隣接ウィンドウは開始位置が近すぎるとほぼ同じなので stride 風に間引く
    too_close = False
    for prev in picked:
      if (
        prev["seed_meta"]["project_id"] == pid
        and prev["seed_meta"]["section_key"] == item["seed_meta"]["section_key"]
        and prev["seed_meta"].get("unit") == "window"
        and item["seed_meta"].get("unit") == "window"
        and abs(
          prev["seed_meta"]["window_start"] - item["seed_meta"]["window_start"]
        )
        < args.window_paragraphs
      ):
        too_close = True
        break
    if too_close:
      continue
    seen_windows.add(wkey)
    item["id"] = f"he-v2-{len(picked) + 1:02d}"
    picked.append(item)
    per_project[pid] += 1

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for item in picked:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")
  write_preview(picked, Path(args.preview))

  n_section = sum(1 for it in picked if it["seed_meta"].get("unit") == "section")
  print(f"pool after token filter: {len(pool)} (overflow skipped: {n_overflow})")
  print(f"items: {len(picked)} (whole sections: {n_section}, windows: {len(picked) - n_section})")
  print("per project:", dict(per_project))
  print(f"max_tokens: {args.max_tokens}")
  print(f"wrote {out_path}")
  print(f"wrote {args.preview}")


if __name__ == "__main__":
  main()
