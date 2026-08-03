#!/usr/bin/env python3
"""段落推敲以外の差分を揃えつつ、revised（正解側）の地の文は落とさない。

編集 SFT の学習対象は「意味を保った段落の推敲」である。
revised は人間推敲後の正解本文なので、段落を削って再構成してはならない。

やること:
- revised の地の文は順序どおりすべて残す
- コード / 画像 / aside だけ、対応する draft 側の形に差し替える
  （フェンス属性・function 殻・note 囲みなどは推敲の軸ではない）

draft 側:
- 地の文は元のまま（削られた段落も「削除された」信号として残してよい）
- 構造ブロックは revised に出したのと同じ形にそろえる
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

MEDIA_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)")
STRUCTURAL_MATCH_MIN = 0.25


@dataclass
class Block:
  kind: str  # prose | code | aside | media
  text: str

  @property
  def norm(self) -> str:
    if self.kind == "code":
      return normalize_code_body(self.text)
    if self.kind == "aside":
      return strip_blockquote_markers(self.text)
    return re.sub(r"\s+", " ", self.text).strip()


def strip_blockquote_markers(text: str) -> str:
  lines = []
  for ln in text.splitlines():
    if ln.startswith("> "):
      lines.append(ln[2:])
    elif ln.startswith(">"):
      lines.append(ln[1:])
    else:
      lines.append(ln)
  return "\n".join(lines).strip()


def normalize_code_body(text: str) -> str:
  lines = text.splitlines()
  if not lines:
    return ""
  body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
  out: list[str] = []
  for ln in body:
    s = ln.strip()
    if not s or s in {"…（省略）…", "...（省略）...", "…", "..."}:
      continue
    if s.startswith("<!--") and s.endswith("-->"):
      continue
    out.append(re.sub(r"\s+", " ", s))
  return "\n".join(out)


def is_media_block(text: str) -> bool:
  lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
  if not lines:
    return False
  return all(MEDIA_RE.match(ln) or ln.startswith("{#") for ln in lines)


def parse_blocks(text: str) -> list[Block]:
  lines = text.splitlines()
  blocks: list[Block] = []
  i = 0
  n = len(lines)

  while i < n:
    ln = lines[i]
    if ln.strip().startswith("```"):
      j = i + 1
      while j < n and not lines[j].strip().startswith("```"):
        j += 1
      if j < n:
        j += 1
      blocks.append(Block("code", "\n".join(lines[i:j])))
      i = j
      continue

    if ln.startswith(">"):
      j = i
      while j < n and (lines[j].startswith(">") or lines[j].strip() == ""):
        if lines[j].strip() == "" and j + 1 < n and not lines[j + 1].startswith(">"):
          break
        j += 1
      blocks.append(Block("aside", "\n".join(lines[i:j])))
      i = j
      continue

    if not ln.strip():
      i += 1
      continue

    if re.fullmatch(r"<!--.*?-->", ln.strip()):
      i += 1
      continue

    buf = [ln]
    i += 1
    while i < n:
      cur = lines[i]
      if not cur.strip():
        break
      if cur.strip().startswith("```") or cur.startswith(">"):
        break
      if re.fullmatch(r"<!--.*?-->", cur.strip()):
        i += 1
        continue
      buf.append(cur)
      i += 1
    raw = "\n".join(buf).strip("\n")
    if not raw.strip():
      continue
    if is_media_block(raw):
      blocks.append(Block("media", raw))
    else:
      blocks.append(Block("prose", raw))

  return blocks


def unwrap_answer_asides(blocks: list[Block]) -> list[Block]:
  """答え / ans 囲みは地の文へ。note は aside のまま。"""
  out: list[Block] = []
  for b in blocks:
    if b.kind != "aside":
      out.append(b)
      continue
    inner = strip_blockquote_markers(b.text)
    if re.match(r"^note\b", inner, re.I):
      out.append(b)
      continue
    # ans / 答え / ヒント → 地の文
    lines = inner.splitlines()
    if lines and re.match(
      r"^(ans\{[^}]*\}|ヒント[:：]?|warning|警告)\s*$",
      lines[0].strip(),
      re.I,
    ):
      lines = lines[1:]
      inner = "\n".join(lines).strip()
    if inner:
      # aside 内にコードがあれば分解
      out.extend(parse_blocks(inner))
  return out


def best_structural_match(
  target: Block,
  candidates: list[Block],
  used: set[int],
) -> int | None:
  best_i = None
  best_s = -1.0
  for i, c in enumerate(candidates):
    if i in used or c.kind != target.kind:
      continue
    s = difflib.SequenceMatcher(None, target.norm, c.norm).ratio()
    # 同種でまだ未使用なら、類似が低くても順番候補にする
    if best_i is None:
      best_i, best_s = i, s
    elif s > best_s:
      best_i, best_s = i, s
  if best_i is None:
    return None
  if best_s < STRUCTURAL_MATCH_MIN and target.norm and candidates[best_i].norm:
    # 中身がまったく違う同種ブロックは対応とみなさない
    return None
  return best_i


def align_nonprose_to_draft(draft: str, revised: str) -> tuple[str, str, dict]:
  """revised の地の文は全残し。構造だけ draft 形に寄せる。"""
  d_blocks = unwrap_answer_asides(parse_blocks(draft))
  r_blocks = unwrap_answer_asides(parse_blocks(revised))

  d_struct = [(i, b) for i, b in enumerate(d_blocks) if b.kind in {"code", "media", "aside"}]
  used_d: set[int] = set()

  out_r: list[str] = []
  # revised 上で使った draft 構造の置換結果（draft 側の同位置にも使う）
  replaced_from_draft: dict[int, str] = {}

  stats = {
    "prose_kept_revised": 0,
    "struct_from_draft": 0,
    "struct_kept_revised": 0,
    "draft_aside_unwrapped": 0,
  }
  for b in parse_blocks(draft):
    if b.kind == "aside" and not re.match(r"^>\s*note\b", b.text, re.I | re.M):
      stats["draft_aside_unwrapped"] += 1

  # revised を主軸に出力
  di_struct = 0
  for rb in r_blocks:
    if rb.kind == "prose":
      out_r.append(rb.text)
      stats["prose_kept_revised"] += 1
      continue

    # 構造: draft の未使用同種から最良を取る（なければ revised のまま）
    # まず順方向の同種も試す
    match_i = None
    # 順方向優先
    while di_struct < len(d_struct) and d_struct[di_struct][0] in used_d:
      di_struct += 1
    if di_struct < len(d_struct) and d_struct[di_struct][1].kind == rb.kind:
      cand_i, cand_b = d_struct[di_struct]
      s = difflib.SequenceMatcher(None, rb.norm, cand_b.norm).ratio()
      if s >= STRUCTURAL_MATCH_MIN or not rb.norm or not cand_b.norm:
        match_i = cand_i
        di_struct += 1

    if match_i is None:
      match_i = best_structural_match(rb, d_blocks, used_d)

    if match_i is not None:
      used_d.add(match_i)
      text = d_blocks[match_i].text
      out_r.append(text)
      replaced_from_draft[match_i] = text
      stats["struct_from_draft"] += 1
    else:
      out_r.append(rb.text)
      stats["struct_kept_revised"] += 1

  # draft 側: 地の文は維持。構造は revised で採用した draft 形にそろえる
  out_d: list[str] = []
  for i, db in enumerate(d_blocks):
    if db.kind == "prose":
      out_d.append(db.text)
      continue
    if i in replaced_from_draft:
      out_d.append(replaced_from_draft[i])
    else:
      out_d.append(db.text)

  new_draft = "\n\n".join(out_d).strip() + "\n"
  new_revised = "\n\n".join(out_r).strip() + "\n"
  new_draft = re.sub(r"\n{3,}", "\n\n", new_draft)
  new_revised = re.sub(r"\n{3,}", "\n\n", new_revised)
  return new_draft, new_revised, stats


def apply_to_chat_jsonl(path: Path, out_path: Path | None = None) -> dict:
  n = 0
  changed = 0
  empty = 0
  agg: dict[str, int] = {}
  rows_out: list[dict] = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      n += 1
      msgs = obj.get("messages") or []
      if len(msgs) < 2:
        rows_out.append(obj)
        continue
      user = str(msgs[0].get("content") or "")
      rev = str(msgs[1].get("content") or "")
      if "\n\n" in user:
        head, draft = user.split("\n\n", 1)
        prefix = head + "\n\n"
      else:
        prefix, draft = "", user
      new_d, new_r, st = align_nonprose_to_draft(draft, rev)
      if not new_d.strip() or not new_r.strip():
        empty += 1
        rows_out.append(obj)
        continue
      if new_d != draft or new_r != rev:
        changed += 1
      for k, v in st.items():
        agg[k] = agg.get(k, 0) + int(v)
      msgs[0]["content"] = prefix + new_d
      msgs[1]["content"] = new_r
      obj.setdefault("meta", {})["nonprose_aligned"] = True
      rows_out.append(obj)

  dest = out_path or path
  with dest.open("w", encoding="utf-8") as f:
    for obj in rows_out:
      f.write(json.dumps(obj, ensure_ascii=False) + "\n")
  return {"n": n, "changed": changed, "skipped_empty": empty, **agg, "out": str(dest)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--draft", type=Path)
  parser.add_argument("--revised", type=Path)
  parser.add_argument("--jsonl", type=Path)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()
  if args.jsonl:
    print(json.dumps(apply_to_chat_jsonl(args.jsonl, args.out), ensure_ascii=False, indent=2))
    return
  if args.draft and args.revised:
    nd, nr, st = align_nonprose_to_draft(
      args.draft.read_text(encoding="utf-8"),
      args.revised.read_text(encoding="utf-8"),
    )
    print("STATS", st)
    print("=== DRAFT ===")
    print(nd)
    print("=== REVISED ===")
    print(nr)
    return
  raise SystemExit("need --jsonl or --draft/--revised")


if __name__ == "__main__":
  main()
