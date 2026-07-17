#!/usr/bin/env python3
"""難試験 v2（節単位・構成軸）の項目を held-out 節ペアから生成する。

方針（ハイブリッド C）:
  - base_text  = 下書き節（実編集の source 側）
  - human 候補 = 編集者の実編集（gold の想定だが、順位付けは人手で行う）
  - base 候補  = 下書きのまま（identity）
  - deg-* 候補 = 実編集テキストに制御された構成改悪を加えたもの。
                 文レベルの質は gold と同一なので、構成の識別力だけを試す。

改悪の種類:
  deg-join    段落境界（空行）を全部落とし、一塊にする
  deg-split   一文ごとに段落化し、過剰分割する
  deg-reverse 本文段落の順序を逆転する（トピックの流れを壊す）

モデル推敲候補は本脚本では作らない（必要なら v1 と同じ手順で後から追記する）。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HEADING_RE = re.compile(r"^#{1,6}\s")


def load_rows(path: Path) -> list[dict]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def split_heading_body(text: str) -> tuple[str, str]:
  """節テキストを（見出し行, 本文）に分ける。見出しが無ければ空文字。"""
  lines = text.splitlines()
  if lines and HEADING_RE.match(lines[0]):
    return lines[0], "\n".join(lines[1:]).strip("\n")
  return "", text.strip("\n")


def paragraphs(body: str) -> list[str]:
  return [p.strip("\n") for p in re.split(r"\n\s*\n", body) if p.strip()]


def join_with_heading(heading: str, body: str) -> str:
  if heading:
    return heading + "\n\n" + body.strip("\n") + "\n"
  return body.strip("\n") + "\n"


def deg_join(text: str) -> str:
  """段落境界を消して一塊にする。"""
  heading, body = split_heading_body(text)
  paras = paragraphs(body)
  merged = "\n".join(p for p in paras)
  return join_with_heading(heading, merged)


def deg_split(text: str) -> str:
  """一文（一行）ごとに段落化する。"""
  heading, body = split_heading_body(text)
  lines = [line for line in body.splitlines() if line.strip()]
  return join_with_heading(heading, "\n\n".join(lines))


def deg_reverse(text: str) -> str:
  """本文段落の順序を逆転する。"""
  heading, body = split_heading_body(text)
  paras = paragraphs(body)
  return join_with_heading(heading, "\n\n".join(reversed(paras)))


DEGRADATIONS = [
  ("deg-join", deg_join),
  ("deg-split", deg_split),
  ("deg-reverse", deg_reverse),
]


def is_eligible(row: dict, *, max_chars: int, min_paragraphs: int) -> bool:
  src, edt = row["source_text"], row["edited_text"]
  meta = row["meta"]
  if len(src) > max_chars or len(edt) > max_chars:
    return False
  if meta["paragraph_count_source"] < min_paragraphs:
    return False
  if meta["paragraph_count_edited"] < min_paragraphs:
    return False
  # 作業メモやコードブロックが残る節は除外する。
  # base 側にメモがあると「メモを消した」だけで human を識別でき、構成の試験にならない。
  for marker in ("```", "~~~", "|--", "af://", "★", "<!--", "TODO", "FIXME"):
    if marker in src or marker in edt:
      return False
  return True


def selection_key(row: dict) -> tuple:
  """段落数変化が大きいものを優先し、次に短いものを優先する。"""
  meta = row["meta"]
  para_delta = abs(meta["paragraph_count_edited"] - meta["paragraph_count_source"])
  return (-para_delta, len(row["source_text"]))


def build_item(idx: int, row: dict) -> dict | None:
  base_text = row["source_text"].strip("\n") + "\n"
  human_text = row["edited_text"].strip("\n") + "\n"

  candidates = [
    {"id": "human", "text": human_text, "generator": "human", "prompt_tag": "real-edit"},
    {"id": "base", "text": base_text, "generator": "copy", "prompt_tag": "identity"},
  ]
  seen = {human_text.strip(), base_text.strip()}
  for deg_id, fn in DEGRADATIONS:
    deg_text = fn(human_text)
    if deg_text.strip() in seen:
      return None
    seen.add(deg_text.strip())
    candidates.append(
      {"id": deg_id, "text": deg_text, "generator": "script", "prompt_tag": deg_id}
    )

  meta = row["meta"]
  return {
    "id": f"he-v2-{idx:02d}",
    "seed_text": "",
    "seed_meta": {
      "project_id": row["project_id"],
      "path": meta["path"],
      "section_key": meta["section_key"],
      "merged_branch": meta.get("merged_branch", ""),
      "source_reference": row["source_reference"],
      "paragraph_count_source": meta["paragraph_count_source"],
      "paragraph_count_edited": meta["paragraph_count_edited"],
      "note": "held-out real edit; draft section as base",
    },
    "base_text": base_text,
    "base_generator": "draft",
    "candidates": candidates,
    "human": {},
    "status": "pending",
  }


def write_preview(items: list[dict], path: Path) -> None:
  lines: list[str] = [
    "# Hard Eval v2 candidates preview",
    "",
    "各 `## he-v2-XX` 内の `###` サブセクションを**良い順**に並べ替える。",
    "見出しと本文は一緒に動かし、本文は編集しない。",
    "並べ替え後: `make hard-eval-label PREVIEW=... SOURCE=... OUTPUT=...`",
    "",
  ]
  for item in items:
    seed_meta = item["seed_meta"]
    lines.append(f"## {item['id']}")
    lines.append("")
    lines.append(
      f"（{seed_meta['project_id']} / {seed_meta['section_key']} / "
      f"段落 {seed_meta['paragraph_count_source']}→{seed_meta['paragraph_count_edited']}）"
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
  parser.add_argument("--out", required=True, help="pending JSONL の出力先")
  parser.add_argument("--preview", required=True, help="並べ替え用 Markdown の出力先")
  parser.add_argument("--n-items", type=int, default=24)
  parser.add_argument("--max-chars", type=int, default=2600)
  parser.add_argument("--min-paragraphs", type=int, default=3)
  parser.add_argument("--max-per-project", type=int, default=8)
  args = parser.parse_args()

  rows = load_rows(Path(args.input))
  eligible = [
    r for r in rows
    if is_eligible(r, max_chars=args.max_chars, min_paragraphs=args.min_paragraphs)
  ]

  # 同一節が複数ブランチから重複採掘されるので、段落変化が最大のものだけ残す
  by_section: dict[tuple, dict] = {}
  for row in eligible:
    key = (row["project_id"], row["meta"]["path"], row["meta"]["section_key"])
    prev = by_section.get(key)
    if prev is None or selection_key(row) < selection_key(prev):
      by_section[key] = row

  ranked = sorted(by_section.values(), key=selection_key)
  picked: list[dict] = []
  per_project: dict[str, int] = defaultdict(int)
  for row in ranked:
    if len(picked) >= args.n_items:
      break
    if per_project[row["project_id"]] >= args.max_per_project:
      continue
    picked.append(row)
    per_project[row["project_id"]] += 1

  items: list[dict] = []
  for idx, row in enumerate(picked, start=1):
    item = build_item(idx, row)
    if item is not None:
      items.append(item)

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for item in items:
      f.write(json.dumps(item, ensure_ascii=False) + "\n")
  write_preview(items, Path(args.preview))

  n_changed = sum(
    1 for it in items
    if it["seed_meta"]["paragraph_count_source"] != it["seed_meta"]["paragraph_count_edited"]
  )
  print(f"eligible sections: {len(eligible)} (dedup: {len(by_section)})")
  print(f"items: {len(items)} (paragraph-count changed: {n_changed})")
  print("per project:", dict(per_project))
  print(f"wrote {out_path}")
  print(f"wrote {args.preview}")


if __name__ == "__main__":
  main()
