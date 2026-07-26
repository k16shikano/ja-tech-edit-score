#!/usr/bin/env python3
"""生成 → 二軸判定 → 反復の推敲ループを 1 コマンドで回す。

各反復で、現版から Cursor SDK で推敲案を複数生成し、
pref-sentseq のスコアで順位付け、pref-bt をゲート（元の下書きに対して
細部が悪化した案を失格）として最良案を選ぶ。

マージンはすべて「元の下書き」を基準に測る。合格ライン（--min-margin）は
`make calibrate-margins` の人間編集マージン分布（中央値 1.9、p25 0.1）を根拠に選ぶ。
合格するか、改善が止まるか、反復上限に達したら終了し、最良版をファイルに書き出す。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from pref_scorer import load_scorer

PROMPT_TEMPLATES = {
  "plain-v1": (
    "次の日本語の技術文書の下書きを、意味を保ったまま推敲してください。"
    "推敲後の本文だけを出力してください。前置きや説明は不要です。\n\n"
    "{draft}"
  ),
  "structure-v1": (
    "次の日本語の技術文書の下書きを、意味を保ったまま推敲してください。"
    "特に、段落の切り方と並び（一つの段落に一つの主張、主張の単位で切る、"
    "並びの論理的な必然性）を見直してください。"
    "推敲後の本文だけを出力してください。前置きや説明は不要です。\n\n"
    "{draft}"
  ),
}


def validate_revision(draft: str, text: str) -> str | None:
  cleaned = text.strip()
  if not cleaned:
    return "empty"
  if cleaned == draft.strip():
    return "identical_to_draft"
  if len(draft) > 0 and (len(cleaned) > len(draft) * 2 or len(cleaned) < len(draft) * 0.5):
    return "length_out_of_range"
  return None


def generate_candidate(draft: str, *, prompt_tag: str, model: str, cwd: Path, api_key: str) -> str:
  from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions

  prompt = PROMPT_TEMPLATES[prompt_tag].format(draft=draft)
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--file", required=True, help="下書きファイル（1節ぶんを推奨）")
  parser.add_argument("--out", help="最良版の出力先（既定: <file>.revised.md）")
  parser.add_argument("--report", help="経過 JSON の出力先（既定: <file>.revise-report.json）")
  parser.add_argument("--model", default="composer-2.5", help="Cursor agent model id")
  parser.add_argument("--n-candidates", type=int, default=3, help="1反復あたりの生成数")
  parser.add_argument("--max-iters", type=int, default=3, help="反復上限")
  parser.add_argument("--min-margin", type=float, default=1.9, help="合格ライン（sentseq、元下書き基準）")
  parser.add_argument("--gate-min-margin", type=float, default=0.0, help="btゲート（元下書き基準）")
  parser.add_argument(
    "--min-iter-gain",
    type=float,
    default=0.1,
    help="1反復の改善幅がこれ未満なら『改善が止まった』として打ち切る",
  )
  parser.add_argument("--primary-model", default="outputs/pref-sentseq-3e4")
  parser.add_argument("--gate-model", default="outputs/pref-bt")
  args = parser.parse_args()

  api_key = os.environ.get("CURSOR_API_KEY", "").strip()
  if not api_key:
    raise SystemExit("CURSOR_API_KEY is not set")

  root = Path(__file__).resolve().parents[1]
  draft_path = Path(args.file).expanduser().resolve()
  source = draft_path.read_text(encoding="utf-8")
  out_path = Path(args.out).expanduser() if args.out else draft_path.with_suffix(draft_path.suffix + ".revised.md")
  report_path = (
    Path(args.report).expanduser()
    if args.report
    else draft_path.with_suffix(draft_path.suffix + ".revise-report.json")
  )

  primary = load_scorer(Path(args.primary_model))
  gate = load_scorer(Path(args.gate_model))

  def margins(text: str) -> tuple[float, float]:
    """元の下書きを基準にした (sentseq マージン, bt マージン)。"""
    p = primary.score(source, [text, source], batch_size=4)
    g = gate.score(source, [text, source], batch_size=4)
    return float(p[0] - p[1]), float(g[0] - g[1])

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
      text = generate_candidate(current, prompt_tag=tag, model=args.model, cwd=root, api_key=api_key)
      elapsed = time.perf_counter() - started
      reject = validate_revision(current, text)
      row = {"iter": it, "prompt_tag": tag, "elapsed_s": round(elapsed, 1)}
      if reject:
        row.update({"status": "rejected", "reject_reason": reject})
        print(f"[iter {it}] cand {k + 1}/{args.n_candidates} ({tag}): rejected ({reject})", flush=True)
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
          f"[iter {it}] cand {k + 1}/{args.n_candidates} ({tag}): "
          f"margin={m_primary:+.2f} gate={'pass' if row['gate_passed'] else 'FAIL'}({m_gate:+.2f}) "
          f"({elapsed:.0f}s)",
          flush=True,
        )
      iter_rows.append(row)
    trail.extend(iter_rows)

    eligible = [r for r in iter_rows if r.get("status") == "scored" and r["gate_passed"]]
    if not eligible:
      print(f"[iter {it}] 全候補がゲート不合格または生成失敗。現版を維持して終了", flush=True)
      status = "no_eligible_candidate"
      break
    best = max(eligible, key=lambda r: r["margin_primary"])
    gain = best["margin_primary"] - current_margin
    if gain < args.min_iter_gain:
      print(
        f"[iter {it}] 改善幅 {gain:+.2f} < {args.min_iter_gain}。改善が止まったと判定して終了",
        flush=True,
      )
      status = "converged"
      break

    current = best["text"]
    current_margin = best["margin_primary"]
    print(f"[iter {it}] 現版を更新: margin={current_margin:+.2f}（元下書き基準）", flush=True)
    if current_margin >= args.min_margin:
      status = "accepted"
      break

  accepted = current_margin >= args.min_margin
  out_path.write_text(current, encoding="utf-8")
  report = {
    "file": str(draft_path),
    "out": str(out_path),
    "status": status,
    "accepted": accepted,
    "final_margin_primary": round(current_margin, 4),
    "min_margin": args.min_margin,
    "gate_min_margin": args.gate_min_margin,
    "model": args.model,
    "n_candidates": args.n_candidates,
    "max_iters": args.max_iters,
    "trail": [
      {k: v for k, v in row.items() if k != "text"} for row in trail
    ],
  }
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

  print("", flush=True)
  print(f"result: {'accepted' if accepted else status}", flush=True)
  print(f"final margin: {current_margin:+.2f}（合格ライン {args.min_margin}、人間編集の中央値 1.9）", flush=True)
  print(f"best text: {out_path}", flush=True)
  print(f"report: {report_path}", flush=True)
  if not accepted:
    sys.exit(2)


if __name__ == "__main__":
  main()
