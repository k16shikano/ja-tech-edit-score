#!/usr/bin/env python3
"""曖昧性解消の予備測定（A1 のみ、学習なし）。"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ead.common import ead_reports, fmt_rate, load_jsonl, md_table, rate_summary, repo_root, write_json

try:
  import ginza  # noqa: F401
  import spacy
except ImportError as exc:
  raise SystemExit(f"ginza/ja-ginza required: {exc}") from exc

try:
  from sudachipy import Dictionary
except ImportError as exc:
  raise SystemExit(f"sudachipy required: {exc}") from exc

SENT_SPLIT = re.compile(r"(?<=[。！？\n])")
T2_RE = re.compile(r"([^、。！？\n]{1,80}?)(と|や)([^、。！？\n]{1,80}?)の([^、。！？\n]{1,80})")
T7_RE = re.compile(r"(?:[^\s、。！？\n]+の){3,}[^\s、。！？\n]+")
T8_QUANT = re.compile(r"(すべて|必ず|全(?:て)?|いつも|常に)")
T8_NEG = re.compile(r"(ない|ません|ず|ぬ|非)")
GA_PARTICLE = re.compile(r"が")
NUM_POS = frozenset({"NUM", "NOUN"})


@dataclass
class Site:
  type_id: str
  sent_idx: int
  start: int
  end: int
  snippet: str
  meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SiteOutcome:
  site: Site
  outcome: str  # resolved | kept | vanished
  edited_snippet: str
  unchanged_sentence: bool
  delta_sign: str


def split_sentences(text: str) -> list[str]:
  parts = SENT_SPLIT.split(text.strip())
  return [p for p in parts if p.strip()]


def sudachi_surfaces(text: str, tokenizer) -> list[str]:
  return [m.surface() for m in tokenizer.tokenize(text)]


def align_sentences(draft_sents: list[str], edit_sents: list[str], tokenizer) -> list[tuple[int | None, float]]:
  """draft index -> (edit index or None, score)."""
  edit_toks = [sudachi_surfaces(s, tokenizer) for s in edit_sents]
  draft_toks = [sudachi_surfaces(s, tokenizer) for s in draft_sents]
  mapping: list[tuple[int | None, float]] = []
  used_edit: set[int] = set()
  for di, dt in enumerate(draft_toks):
    best_j: int | None = None
    best_score = 0.0
    for ej, et in enumerate(edit_toks):
      if ej in used_edit:
        continue
      if not dt and not et:
        score = 1.0
      elif not dt or not et:
        score = 0.0
      else:
        score = difflib.SequenceMatcher(None, dt, et).ratio()
      if score > best_score:
        best_score = score
        best_j = ej
    if best_j is not None and best_score >= 0.45:
      used_edit.add(best_j)
      mapping.append((best_j, best_score))
    else:
      mapping.append((None, best_score))
  return mapping


def char_map_between(a: str, b: str) -> tuple[int, int] | None:
  if a == b:
    return (0, len(b))
  matcher = difflib.SequenceMatcher(None, a, b)
  blocks = matcher.get_matching_blocks()
  if not blocks:
    return None
  first = blocks[0]
  return (first.b, first.b + first.size) if first.size > 0 else None


def tokens_in_span(sent, start: int, end: int):
  return [t for t in sent if t.idx >= start and t.idx < end]


# --- enumerators (draft side, permissive) ---

def enum_t1(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  toks = list(sent)
  for i, tok in enumerate(toks):
    if tok.dep_ != "acl":
      continue
    j = i + 1
    nouns: list[Any] = []
    while j < len(toks) and toks[j].pos_ in {"NOUN", "PROPN", "NUM"}:
      nouns.append(toks[j])
      j += 1
    if len(nouns) >= 2:
      start = tok.idx
      end = nouns[-1].idx + len(nouns[-1].text)
      sites.append(
        Site("T1", sent_idx, start, end, sent.text[start:end], {"acl": tok.text, "nouns": [n.text for n in nouns]})
      )
  return sites


def enum_t2(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  for m in T2_RE.finditer(sent.text):
    sites.append(Site("T2", sent_idx, m.start(), m.end(), m.group(0), {"conj": m.group(2)}))
  return sites


def enum_t3(sent, sent_idx: int) -> list[Site]:
  ga_positions = [t.idx for t in sent if t.text == "が" and t.pos_ == "ADP"]
  if len(ga_positions) < 2:
    return []
  return [Site("T3", sent_idx, ga_positions[0], ga_positions[-1] + 1, sent.text[ga_positions[0] : ga_positions[-1] + 1], {"ga_n": len(ga_positions)})]


def enum_t4(sent, sent_idx: int) -> list[Site]:
  has_pred = any(t.pos_ in {"VERB", "AUX"} and t.dep_ in {"ROOT", "ccomp", "advcl", "acl"} for t in sent)
  has_nsubj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in sent)
  if has_pred and not has_nsubj:
    root = [t for t in sent if t.dep_ == "ROOT"]
    if root:
      t = root[0]
      return [Site("T4", sent_idx, t.idx, t.idx + len(t.text), sent.text.strip(), {"root": t.text})]
  return []


def enum_t5(sent, sent_idx: int) -> list[Site]:
  advcls = [t for t in sent if t.dep_ == "advcl"]
  if len(advcls) < 2:
    return []
  advcls.sort(key=lambda t: t.i)
  run = [advcls[0]]
  sites: list[Site] = []
  for t in advcls[1:]:
    if t.i - run[-1].i <= 2:
      run.append(t)
    else:
      if len(run) >= 2:
        start = run[0].idx
        end = run[-1].idx + len(run[-1].text)
        sites.append(Site("T5", sent_idx, start, end, sent.text[start:end], {"n": len(run)}))
      run = [t]
  if len(run) >= 2:
    start = run[0].idx
    end = run[-1].idx + len(run[-1].text)
    sites.append(Site("T5", sent_idx, start, end, sent.text[start:end], {"n": len(run)}))
  return sites


def enum_t6(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  for tok in sent:
    if tok.pos_ != "NUM" and not (tok.text.isdigit() or re.match(r"^[一二三四五六七八九十百千万億]+", tok.text)):
      continue
    head = tok.head
    if head.i == tok.i:
      continue
    if abs(head.i - tok.i) > 1 and head.pos_ in {"NOUN", "PROPN"}:
      start = min(tok.idx, head.idx)
      end = max(tok.idx + len(tok.text), head.idx + len(head.text))
      sites.append(Site("T6", sent_idx, start, end, sent.text[start:end], {"num": tok.text, "head": head.text}))
  return sites


def enum_t7(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  for m in T7_RE.finditer(sent.text):
    sites.append(Site("T7", sent_idx, m.start(), m.end(), m.group(0), {"no_count": m.group(0).count("の")}))
  return sites


def enum_t8(sent, sent_idx: int) -> list[Site]:
  if not T8_QUANT.search(sent.text) or not T8_NEG.search(sent.text):
    return []
  q = T8_QUANT.search(sent.text)
  n = T8_NEG.search(sent.text)
  if not q or not n:
    return []
  start = min(q.start(), n.start())
  end = max(q.end(), n.end())
  return [Site("T8", sent_idx, start, end, sent.text[start:end], {"quant": q.group(0), "neg": n.group(0)})]


def enum_t9(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  for tok in sent:
    if tok.dep_ != "nsubjpass" and "pass" not in str(tok.morph):
      if tok.pos_ == "VERB" and any(c.text in {"れる", "られる", "された", "れ"} for c in tok.children):
        pass
    if tok.dep_ in {"ROOT", "acl"} and any(c.dep_ == "auxpass" or c.lemma_ in {"れる", "られる"} for c in tok.subtree):
      has_agent = any(c.dep_ in {"obl", "nmod"} and c.text not in {"に", "を"} for c in tok.children)
      if not has_agent and any(c.text == "に" for c in tok.children):
        sites.append(Site("T9", sent_idx, tok.idx, tok.idx + len(tok.text), sent.text[tok.idx : tok.idx + len(tok.text)], {}))
  return sites


def enum_t10(sent, sent_idx: int) -> list[Site]:
  sites: list[Site] = []
  toks = list(sent)
  for i, tok in enumerate(toks):
    if tok.pos_ not in {"NOUN", "PROPN"}:
      continue
    if i + 1 < len(toks) and toks[i + 1].text == "、" and (i == 0 or toks[i - 1].pos_ != "ADP"):
      sites.append(Site("T10", sent_idx, tok.idx, toks[i + 1].idx + 1, sent.text[tok.idx : toks[i + 1].idx + 1], {}))
  return sites


ENUMERATORS: dict[str, Callable] = {
  "T1": enum_t1,
  "T2": enum_t2,
  "T3": enum_t3,
  "T4": enum_t4,
  "T5": enum_t5,
  "T6": enum_t6,
  "T7": enum_t7,
  "T8": enum_t8,
  "T9": enum_t9,
  "T10": enum_t10,
}

TYPE_LABELS = {
  "T1": "連体修飾のスコープ",
  "T2": "並列のスコープ",
  "T3": "「が」の多義",
  "T4": "主語の省略",
  "T5": "連用中止の連鎖",
  "T6": "数量詞の遊離",
  "T7": "「の」の連鎖",
  "T8": "否定のスコープ",
  "T9": "受動態の動作主省略",
  "T10": "格助詞の省略",
}


def resolved_t1(draft_site: Site, edit_sent) -> bool:
  if enum_t1(edit_sent, 0):
    return False
  toks = list(edit_sent)
  for i, tok in enumerate(toks):
    if tok.dep_ == "acl":
      j = i + 1
      nouns = 0
      while j < len(toks) and toks[j].pos_ in {"NOUN", "PROPN", "NUM"}:
        nouns += 1
        j += 1
      if nouns >= 2:
        return False
  return True


def resolved_t2(draft_site: Site, edit_sent) -> bool:
  return not bool(T2_RE.search(edit_sent.text))


def resolved_t3(draft_site: Site, edit_sent) -> bool:
  ga_n = sum(1 for t in edit_sent if t.text == "が" and t.pos_ == "ADP")
  if ga_n < 2:
    return True
  if "けれど" in edit_sent.text or "しかし" in edit_sent.text:
    return True
  if len(split_sentences(edit_sent.text)) > 1:
    return True
  return False


def resolved_t4(_: Site, edit_sent) -> bool:
  return any(t.dep_ in {"nsubj", "nsubjpass"} for t in edit_sent)


def resolved_t5(_: Site, edit_sent) -> bool:
  advcls = [t for t in edit_sent if t.dep_ == "advcl"]
  return len(advcls) < 2


def resolved_t6(_: Site, edit_sent) -> bool:
  return len(enum_t6(edit_sent, 0)) == 0


def resolved_t7(_: Site, edit_sent) -> bool:
  return not bool(T7_RE.search(edit_sent.text))


def resolved_t8(draft_site: Site, edit_sent) -> bool:
  q = T8_QUANT.search(edit_sent.text)
  n = T8_NEG.search(edit_sent.text)
  if not q or not n:
    return True
  draft_q = draft_site.meta.get("quant", "")
  draft_n = draft_site.meta.get("neg", "")
  dq = T8_QUANT.search(draft_site.snippet)
  dn = T8_NEG.search(draft_site.snippet)
  if dq and dn and q and n:
    return (q.start() != dq.start()) or (n.start() != dn.start())
  return False


def resolved_t9(_: Site, edit_sent) -> bool:
  return len(enum_t9(edit_sent, 0)) == 0 or any(
    c.dep_ in {"obl", "nsubj"} and c.pos_ in {"NOUN", "PROPN"} for t in edit_sent for c in t.children
  )


def resolved_t10(_: Site, edit_sent) -> bool:
  return len(enum_t10(edit_sent, 0)) == 0


RESOLVERS: dict[str, Callable[[Site, Any], bool]] = {
  "T1": resolved_t1,
  "T2": resolved_t2,
  "T3": resolved_t3,
  "T4": resolved_t4,
  "T5": resolved_t5,
  "T6": resolved_t6,
  "T7": resolved_t7,
  "T8": resolved_t8,
  "T9": resolved_t9,
  "T10": resolved_t10,
}


def parse_doc(nlp, text: str):
  text = text.strip()
  if not text:
    return None, []
  doc = nlp(text)
  sents = list(doc.sents)
  return doc, sents


def enumerate_all(sents) -> list[Site]:
  sites: list[Site] = []
  for si, sent in enumerate(sents):
    for tid, fn in ENUMERATORS.items():
      sites.extend(fn(sent, si))
  return sites


def classify_site(
  site: Site,
  draft_sent,
  edit_sent,
  *,
  vanished: bool,
  unchanged: bool,
) -> SiteOutcome:
  if vanished or edit_sent is None:
    return SiteOutcome(site, "vanished", "", unchanged, "")
  edit_snip = edit_sent.text.strip()
  if RESOLVERS[site.type_id](site, edit_sent):
    outcome = "resolved"
  else:
    outcome = "kept"
  return SiteOutcome(site, outcome, edit_snip, unchanged, "")


def process_pair(
  nlp,
  tokenizer,
  draft: str,
  edited: str,
  *,
  item_id: str,
  delta_sign: str,
  enabled_types: list[str],
) -> tuple[list[SiteOutcome], list[Site], dict[str, Any]]:
  draft_sents_raw = split_sentences(draft)
  edit_sents_raw = split_sentences(edited)
  _, draft_sents = parse_doc(nlp, draft)
  _, edit_sents = parse_doc(nlp, edited)
  if draft_sents is None:
    draft_sents = []
  if edit_sents is None:
    edit_sents = []

  mapping = align_sentences(draft_sents_raw, edit_sents_raw, tokenizer)
  align_fail = sum(1 for ej, sc in mapping if ej is None) / max(len(mapping), 1)

  outcomes: list[SiteOutcome] = []
  all_draft_sites: list[Site] = []
  for tid in enabled_types:
    for si, sent in enumerate(draft_sents):
      for site in ENUMERATORS[tid](sent, si):
        all_draft_sites.append(site)
        ej, _ = mapping[si] if si < len(mapping) else (None, 0.0)
        edit_sent = edit_sents[ej] if ej is not None and ej < len(edit_sents) else None
        unchanged = (
          si < len(draft_sents_raw)
          and ej is not None
          and ej < len(edit_sents_raw)
          and draft_sents_raw[si].strip() == edit_sents_raw[ej].strip()
        )
        vanished = ej is None
        out = classify_site(site, sent, edit_sent, vanished=vanished, unchanged=unchanged)
        out.unchanged_sentence = unchanged
        out.delta_sign = delta_sign
        outcomes.append(out)

  new_sites: list[Site] = []
  for ej, sent in enumerate(edit_sents):
    draft_si = next((di for di, (eji, _) in enumerate(mapping) if eji == ej), None)
    draft_sent = draft_sents[draft_si] if draft_si is not None and draft_si < len(draft_sents) else None
    for tid in enabled_types:
      edit_sites = ENUMERATORS[tid](sent, ej)
      draft_sites_same = ENUMERATORS[tid](draft_sent, draft_si or 0) if draft_sent is not None else []
      for es in edit_sites:
        if es.snippet in draft:
          continue
        if any(ds.type_id == tid and (es.snippet in ds.snippet or ds.snippet in es.snippet) for ds in draft_sites_same):
          continue
        new_sites.append(es)

  meta = {"item_id": item_id, "align_fail_rate": align_fail, "n_draft_sents": len(draft_sents_raw)}
  return outcomes, new_sites, meta


def aggregate_type(outcomes: list[SiteOutcome], new_sites: list[Site], type_id: str) -> dict[str, Any]:
  typed = [o for o in outcomes if o.site.type_id == type_id]
  resolved = sum(1 for o in typed if o.outcome == "resolved")
  kept = sum(1 for o in typed if o.outcome == "kept")
  vanished = sum(1 for o in typed if o.outcome == "vanished")
  denom = resolved + kept
  rho = rate_summary(resolved, denom) if denom else {"k": 0, "n": 0, "rate": None, "wilson_95_lower": None, "wilson_95_upper": None}
  new_n = sum(1 for s in new_sites if s.type_id == type_id)
  unchanged_kept = sum(1 for o in typed if o.outcome == "kept" and o.unchanged_sentence)
  changed_res = sum(1 for o in typed if o.outcome == "resolved" and not o.unchanged_sentence)
  return {
    "type_id": type_id,
    "label": TYPE_LABELS[type_id],
    "n_sites": len(typed),
    "resolved": resolved,
    "kept": kept,
    "vanished": vanished,
    "vanish_rate": vanished / len(typed) if typed else None,
    "new": new_n,
    "net_change": new_n - resolved,
    "rho": rho,
    "unchanged_kept": unchanged_kept,
    "changed_resolved": changed_res,
    "comparable": denom >= 30,
  }


def stratify_rho(outcomes: list[SiteOutcome], type_id: str, *, key: Callable[[SiteOutcome], str]) -> dict[str, dict]:
  out: dict[str, dict] = {}
  for label in sorted({key(o) for o in outcomes if o.site.type_id == type_id and key(o)}):
    sub = [o for o in outcomes if o.site.type_id == type_id and key(o) == label]
    res = sum(1 for o in sub if o.outcome == "resolved")
    kept = sum(1 for o in sub if o.outcome == "kept")
    out[label] = rate_summary(res, res + kept)
  return out


def pick_examples(outcomes: list[SiteOutcome], type_id: str, n: int = 5) -> list[dict]:
  examples: list[dict] = []
  for outcome in ("resolved", "kept", "vanished"):
    for o in outcomes:
      if o.site.type_id != type_id:
        continue
      if o.outcome != outcome:
        continue
      examples.append(
        {
          "outcome": o.outcome,
          "draft_snippet": o.site.snippet,
          "edited_snippet": o.edited_snippet[:200],
          "unchanged_sentence": o.unchanged_sentence,
        }
      )
      if sum(1 for e in examples if e["outcome"] == outcome) >= n:
        break
    if len(examples) >= n:
      break
  return examples[:n]


def build_report(payload: dict[str, Any]) -> str:
  lines = [
    "# ead-ambiguity-probe",
    "",
    "A1（段落内 1597 件）の曖昧箇所列挙と解消率。文書レベル集約なし。",
    "",
    f"解析器: GiNZA (`ja_ginza`)。文アラインメント: SudachiPy 形態素列の ratio（閾値 0.45）。",
    f"アラインメント失敗率（文単位・対応なし）: {payload['align_fail_rate_mean']:.3f}（平均）。",
    "",
    "## 型別（ρ 降順）",
    "",
    md_table(
      ["型", "箇所数", "解消", "保持", "消失", "ρ", "新規", "純変化", "比較可"],
      [
        [
          f"{r['type_id']} {r['label']}",
          str(r["n_sites"]),
          str(r["resolved"]),
          str(r["kept"]),
          str(r["vanished"]),
          fmt_rate(r["rho"]) if r["comparable"] else f"{r['rho']['k']}/{r['rho']['n']} (n<30)",
          str(r["new"]),
          str(r["net_change"]),
          "yes" if r["comparable"] else "no",
        ]
        for r in payload["types_by_rho"]
      ],
    ),
    "",
    "## 無変更文での保持",
    "",
    md_table(
      ["型", "無変更文で保持"],
      [[r["type_id"], str(r["unchanged_kept"])] for r in payload["types_by_rho"]],
    ),
    "",
    "## T8–T10",
    "",
    payload["optional_types_note"],
    "",
    "## 層別（Δ文字数符号, 比較可能型のみ抜粋）",
    "",
  ]
  for row in payload.get("strata_delta", [])[:6]:
    lines.append(f"- {row}")
  lines += ["", "## 実例", ""]
  for tid, exs in payload["examples"].items():
    lines.append(f"### {tid} {TYPE_LABELS.get(tid, tid)}")
    lines.append("")
    for i, ex in enumerate(exs, 1):
      lines.append(f"{i}. [{ex['outcome']}] 下書き: {ex['draft_snippet'][:120]}")
      if ex["edited_snippet"]:
        lines.append(f"   推敲: {ex['edited_snippet'][:120]}")
    lines.append("")
  lines += ["## 判定メモ", ""]
  for note in payload["judgment_notes"]:
    lines.append(f"- {note}")
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=repo_root())
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--skip-optional", action="store_true")
  args = parser.parse_args()

  root = args.root.resolve()
  rows = load_jsonl(root / "data/revision_corpus/keep_hunk_nopara.jsonl")
  if len(rows) != 1597 and not args.limit:
    raise SystemExit(f"expected 1597 rows, got {len(rows)}")
  if args.limit:
    rows = rows[: args.limit]

  nlp = spacy.load("ja_ginza")
  nlp.max_length = 2_000_000
  tokenizer = Dictionary().create()

  enabled = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
  optional_note = "T8–T10 を実装済み。"
  if args.skip_optional:
    optional_note = "T8–T10 は `--skip-optional` で除外。"
  else:
    enabled.extend(["T8", "T9", "T10"])

  all_outcomes: list[SiteOutcome] = []
  all_new: list[Site] = []
  align_rates: list[float] = []
  parse_fail = 0

  for i, row in enumerate(rows):
    draft = str(row.get("source_text") or "")
    edited = str(row.get("edited_text") or "")
    if not draft.strip() or not edited.strip():
      parse_fail += 1
      continue
    delta = len(edited) - len(draft)
    delta_sign = "shrink" if delta < 0 else ("expand" if delta > 0 else "neutral")
    try:
      outcomes, new_sites, meta = process_pair(
        nlp,
        tokenizer,
        draft,
        edited,
        item_id=str(row.get("id") or i),
        delta_sign=delta_sign,
        enabled_types=enabled,
      )
    except Exception:
      parse_fail += 1
      continue
    for o in outcomes:
      o.delta_sign = delta_sign
    all_outcomes.extend(outcomes)
    all_new.extend(new_sites)
    align_rates.append(meta["align_fail_rate"])
    if (i + 1) % 200 == 0:
      print(f"processed {i+1}/{len(rows)}", file=sys.stderr)

  type_stats = [aggregate_type(all_outcomes, all_new, tid) for tid in enabled]
  comparable = [t for t in type_stats if t["comparable"]]
  types_by_rho = sorted(
    type_stats,
    key=lambda t: (t["rho"]["rate"] if t["rho"]["rate"] is not None else -1),
    reverse=True,
  )

  strata_lines: list[str] = []
  for t in comparable[:4]:
    by_delta = stratify_rho(all_outcomes, t["type_id"], key=lambda o: o.delta_sign)
    parts = [f"{k}={fmt_rate(v)}" for k, v in by_delta.items()]
    strata_lines.append(f"{t['type_id']}: " + ", ".join(parts))

  examples = {tid: pick_examples(all_outcomes, tid, 5) for tid in enabled}

  judgment: list[str] = []
  if np.mean(align_rates) > 0.10:
    judgment.append(f"アラインメント失敗率 {np.mean(align_rates):.3f} が 10% 超。")
  low_n = [t["type_id"] for t in type_stats if not t["comparable"]]
  if low_n:
    judgment.append(f"比較不可（n<30）: {', '.join(low_n)}")
  top = [t for t in comparable if (t["rho"]["rate"] or 0) > 0.55]
  if top:
    judgment.append("ρ>0.55 の型: " + ", ".join(f"{t['type_id']}={t['rho']['rate']:.3f}" for t in top))
  else:
    judgment.append("どの型も ρ が 0.55 を明確に超えない。")
  high_vanish = [t for t in type_stats if (t["vanish_rate"] or 0) > 0.4]
  if high_vanish:
    judgment.append("消失率>0.4: " + ", ".join(t["type_id"] for t in high_vanish))

  payload = {
    "corpus": "A1 keep_hunk_nopara",
    "n_items": len(rows),
    "parse_fail": parse_fail,
    "align_fail_rate_mean": float(np.mean(align_rates)) if align_rates else None,
    "enabled_types": enabled,
    "optional_types_note": optional_note,
    "types_by_rho": types_by_rho,
    "type_stats": {t["type_id"]: t for t in type_stats},
    "strata_delta": strata_lines,
    "examples": examples,
    "judgment_notes": judgment,
  }

  reports = ead_reports()
  write_json(reports / "ead-ambiguity-probe.json", payload)
  (reports / "ead-ambiguity-probe.md").write_text(build_report(payload), encoding="utf-8")
  print(json.dumps({"wrote": str(reports / "ead-ambiguity-probe.md"), "n_outcomes": len(all_outcomes)}, ensure_ascii=False))


if __name__ == "__main__":
  main()
