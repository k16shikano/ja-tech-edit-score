#!/usr/bin/env python3
"""生成 → 二軸判定 → 反復の推敲ループを 1 コマンドで回す。

入力ファイルを見出し単位の節に分割し、節ごとにループを回して全体を組み直す。
採点モデル（pref-sentseq / pref-bt）の学習データが節単位なので、これが正しい粒度である。

各反復で、節の現版から Cursor SDK で推敲案を複数生成し、
pref-sentseq のスコアで順位付け、pref-bt をゲート（元の節に対して
細部が悪化した案を失格）として最良案を選ぶ。

マージンはすべて「元の節」を基準に測る（margin = s(節, 案) - s(節, 節)）。
基準と候補が同一という学習にない入力を含むため、変更しただけで正の
マージンが付く偏りがある。合格ライン（--min-margin）は `make calibrate-margins`
の self 基準の人間編集マージン分布（中央値 3.7、p25 0.4）を根拠に選ぶが、
正のマージンは改善の証明にならない点に注意。
合格するか、改善が止まるか、反復上限に達したら次の節へ移る。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from markdown_sections import split_sections
from pref_scorer import LoadedScorer, load_scorer
from sentseq_utils import split_document_sentences

MAX_SENTS = 128  # pref-sentseq が読む文数の上限（それ以降は採点に反映されない）

PROMPT_TEMPLATES = {
  "plain-v1": (
    "{norms}次の日本語の技術文書の下書きを、意味を保ったまま推敲してください。"
    "推敲後の本文だけを出力してください。前置きや説明は不要です。\n\n"
    "{draft}"
  ),
  "structure-v1": (
    "{norms}次の日本語の技術文書の下書きを、意味を保ったまま推敲してください。"
    "特に、段落の切り方と並び（一つの段落に一つの主張、主張の単位で切る、"
    "並びの論理的な必然性）を見直してください。"
    "推敲後の本文だけを出力してください。前置きや説明は不要です。\n\n"
    "{draft}"
  ),
}

DEFAULT_SKILLS = "japanese-tech-writing,cognitive-rhythm-writing"


def strip_front_matter(text: str) -> str:
  if text.startswith("---"):
    parts = text.split("---", 2)
    if len(parts) == 3:
      return parts[2].lstrip()
  return text


def load_skill_norms(skill_names: str) -> str:
  """カンマ区切りのスキル名から、プロンプトに前置する規範ブロックを作る。"""
  names = [s.strip() for s in skill_names.split(",") if s.strip() and s.strip() != "none"]
  if not names:
    return ""
  skills_root = Path(os.environ.get("CURSOR_SKILLS_DIR", "~/.cursor/skills")).expanduser()
  blocks: list[str] = []
  for name in names:
    path = skills_root / name / "SKILL.md"
    if not path.is_file():
      raise SystemExit(f"skill not found: {path}")
    blocks.append(strip_front_matter(path.read_text(encoding="utf-8")).strip())
  joined = "\n\n---\n\n".join(blocks)
  return (
    "あなたは日本語技術書の編集者です。次の執筆規範に従って推敲します。\n\n"
    "<規範>\n" + joined + "\n</規範>\n\n"
  )


def validate_revision(draft: str, text: str) -> str | None:
  cleaned = text.strip()
  if not cleaned:
    return "empty"
  if cleaned == draft.strip():
    return "identical_to_draft"
  if len(draft) > 0 and (len(cleaned) > len(draft) * 2 or len(cleaned) < len(draft) * 0.5):
    return "length_out_of_range"
  return None


def generate_candidate(
  draft: str, *, prompt_tag: str, norms: str, model: str, cwd: Path, api_key: str
) -> str:
  from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

  prompt = PROMPT_TEMPLATES[prompt_tag].format(norms=norms, draft=draft)
  try:
    result = Agent.prompt(
      prompt,
      AgentOptions(
        api_key=api_key,
        model=model,
        local=LocalAgentOptions(cwd=str(cwd), setting_sources=[]),
      ),
    )
  except CursorAgentError as err:
    raise SystemExit(
      f"Cursor SDK startup failed: {err.message} (retryable={err.is_retryable})"
    ) from err
  if result.status == "error":
    raise SystemExit("Cursor agent run failed: status=error")
  return str(result.result or "").strip()


def revise_section(
  section_label: str,
  source: str,
  *,
  primary: LoadedScorer,
  gate: LoadedScorer,
  args: argparse.Namespace,
  norms: str,
  root: Path,
  api_key: str,
) -> dict:
  """1節ぶんのループ。best テキストと経過を返す。"""

  def margins(text: str) -> tuple[float, float]:
    p = primary.score(source, [text, source], batch_size=4)
    g = gate.score(source, [text, source], batch_size=4)
    return float(p[0] - p[1]), float(g[0] - g[1])

  n_sents = len(split_document_sentences(source))
  if n_sents > MAX_SENTS:
    print(
      f"  警告: 節が {n_sents} 文あり、採点は先頭 {MAX_SENTS} 文しか見ない",
      file=sys.stderr,
      flush=True,
    )

  prompt_tags = list(PROMPT_TEMPLATES)
  current = source
  current_margin = 0.0
  trail: list[dict] = []
  status = "max_iters_reached"

  for it in range(1, args.max_iters + 1):
    iter_rows: list[dict] = []
    for k in range(args.n_candidates):
      tag = prompt_tags[k % len(prompt_tags)]
      started = time.perf_counter()
      text = generate_candidate(
        current, prompt_tag=tag, norms=norms, model=args.model, cwd=root, api_key=api_key
      )
      elapsed = time.perf_counter() - started
      reject = validate_revision(current, text)
      row = {"iter": it, "prompt_tag": tag, "elapsed_s": round(elapsed, 1)}
      if reject:
        row.update({"status": "rejected", "reject_reason": reject})
        print(f"  [iter {it}] cand {k + 1}/{args.n_candidates} ({tag}): rejected ({reject})", flush=True)
      else:
        m_primary, m_gate = margins(text)
        row.update(
          {
            "status": "scored",
            "margin_primary": round(m_primary, 4),
            "margin_gate": round(m_gate, 4),
            "gate_passed": m_gate >= args.gate_min_margin,
            "chars": len(text),
            "text": text,
          }
        )
        print(
          f"  [iter {it}] cand {k + 1}/{args.n_candidates} ({tag}): "
          f"margin={m_primary:+.2f} gate={'pass' if row['gate_passed'] else 'FAIL'}({m_gate:+.2f}) "
          f"({elapsed:.0f}s)",
          flush=True,
        )
      iter_rows.append(row)
    trail.extend(iter_rows)

    eligible = [r for r in iter_rows if r.get("status") == "scored" and r["gate_passed"]]
    if not eligible:
      print("  全候補がゲート不合格または生成失敗。現版を維持", flush=True)
      status = "no_eligible_candidate"
      break
    best = max(eligible, key=lambda r: r["margin_primary"])
    gain = best["margin_primary"] - current_margin
    if gain < args.min_iter_gain:
      print(f"  [iter {it}] 改善幅 {gain:+.2f} < {args.min_iter_gain}。改善停止と判定", flush=True)
      status = "converged"
      break

    current = best["text"]
    current_margin = best["margin_primary"]
    print(f"  [iter {it}] 現版を更新: margin={current_margin:+.2f}（元の節基準）", flush=True)
    if current_margin >= args.min_margin:
      status = "accepted"
      break

  return {
    "section": section_label,
    "status": status,
    "accepted": current_margin >= args.min_margin,
    "final_margin_primary": round(current_margin, 4),
    "chars": len(current),
    "text": current,
    "trail": [{k: v for k, v in row.items() if k != "text"} for row in trail],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--file", required=True, help="下書きファイル（見出し単位の節に分割して処理する）")
  parser.add_argument("--out", help="推敲版の出力先（既定: <file>.revised.md）")
  parser.add_argument("--report", help="経過 JSON の出力先（既定: <file>.revise-report.json）")
  parser.add_argument("--model", default="composer-2.5", help="Cursor agent model id")
  parser.add_argument("--n-candidates", type=int, default=3, help="1反復あたりの生成数")
  parser.add_argument("--max-iters", type=int, default=3, help="節ごとの反復上限")
  parser.add_argument("--min-margin", type=float, default=3.7, help="合格ライン（sentseq、元の節基準、self 基準分布の中央値）")
  parser.add_argument("--gate-min-margin", type=float, default=0.0, help="btゲート（元の節基準）")
  parser.add_argument(
    "--min-iter-gain",
    type=float,
    default=0.1,
    help="1反復の改善幅がこれ未満なら『改善が止まった』として次の節へ",
  )
  parser.add_argument(
    "--min-section-chars",
    type=int,
    default=200,
    help="これより短い節は推敲せずそのまま通す",
  )
  parser.add_argument("--only-sections", default="", help="カンマ区切りの節見出し部分一致。指定節だけ処理")
  parser.add_argument(
    "--skills",
    default=DEFAULT_SKILLS,
    help="生成プロンプトに規範として同梱するスキル名（カンマ区切り、'none' で無効）",
  )
  parser.add_argument("--primary-model", default="outputs/pref-sentseq-3e4")
  parser.add_argument("--gate-model", default="outputs/pref-bt")
  args = parser.parse_args()

  api_key = os.environ.get("CURSOR_API_KEY", "").strip()
  if not api_key:
    raise SystemExit("CURSOR_API_KEY is not set")

  root = Path(__file__).resolve().parents[1]
  draft_path = Path(args.file).expanduser().resolve()
  full_text = draft_path.read_text(encoding="utf-8")
  out_path = Path(args.out).expanduser() if args.out else draft_path.with_suffix(draft_path.suffix + ".revised.md")
  report_path = (
    Path(args.report).expanduser()
    if args.report
    else draft_path.with_suffix(draft_path.suffix + ".revise-report.json")
  )

  sections = split_sections(full_text)
  if not sections:
    raise SystemExit("empty file")
  # 見出しの重複などで節分割が本文を失っていないかの検査
  reassembled = "".join(body for body in sections.values())
  if len(reassembled) < len(full_text) * 0.95:
    raise SystemExit(
      f"節分割で本文が失われる（{len(full_text)} 文字 → {len(reassembled)} 文字）。"
      "見出しの重複が原因の可能性。--only-sections で節を絞るか、ファイルを分けて実行"
    )
  only = [s.strip() for s in args.only_sections.split(",") if s.strip()]

  todo: list[tuple[str, str]] = []
  passthrough = 0
  for label, body in sections.items():
    if len(body) < args.min_section_chars:
      passthrough += 1
      continue
    if only and not any(key in label for key in only):
      passthrough += 1
      continue
    todo.append((label, body))

  norms = load_skill_norms(args.skills)
  est_gen = len(todo) * args.n_candidates  # 1反復ぶんの生成数（反復すればさらに増える）
  print(
    f"節 {len(sections)} 件のうち {len(todo)} 件を処理（スキップ {passthrough}）。"
    f"生成は最低 {est_gen} 回、最大 {est_gen * args.max_iters} 回。"
    f"規範: {args.skills if norms else 'なし'}",
    flush=True,
  )

  primary = load_scorer(Path(args.primary_model))
  gate = load_scorer(Path(args.gate_model))

  results: dict[str, dict] = {}
  for i, (label, body) in enumerate(todo, start=1):
    print(f"[{i}/{len(todo)}] {label}", flush=True)
    results[label] = revise_section(
      label,
      body,
      primary=primary,
      gate=gate,
      args=args,
      norms=norms,
      root=root,
      api_key=api_key,
    )

  revised_parts = [
    results[label]["text"] if label in results else body
    for label, body in sections.items()
  ]
  out_path.write_text("\n".join(part.rstrip() + "\n" for part in revised_parts), encoding="utf-8")

  n_accepted = sum(1 for r in results.values() if r["accepted"])
  report = {
    "file": str(draft_path),
    "out": str(out_path),
    "model": args.model,
    "skills": args.skills if norms else "",
    "n_candidates": args.n_candidates,
    "max_iters": args.max_iters,
    "min_margin": args.min_margin,
    "gate_min_margin": args.gate_min_margin,
    "n_sections": len(sections),
    "n_processed": len(todo),
    "n_accepted": n_accepted,
    "sections": [
      {k: v for k, v in r.items() if k != "text"} for r in results.values()
    ],
  }
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

  print("", flush=True)
  print(f"result: {n_accepted}/{len(todo)} 節が合格ライン（margin ≥ {args.min_margin}）に到達", flush=True)
  for label, r in results.items():
    print(f"  {'o' if r['accepted'] else 'x'} margin={r['final_margin_primary']:+.2f} [{r['status']}] {label}", flush=True)
  print(f"revised: {out_path}", flush=True)
  print(f"report: {report_path}", flush=True)
  if todo and n_accepted < len(todo):
    sys.exit(2)


if __name__ == "__main__":
  main()
