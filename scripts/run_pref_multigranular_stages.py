#!/usr/bin/env python3
"""段階 1 から 7 を書いた順に学び、検証 50 件を採点する。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STAGE_OUTPUTS = {
  "1": "outputs/pref-sentseq-keep-pairsplit",
  "2": "outputs/pref-pair-draft",
  "3": "outputs/pref-pair-humantop",
  "4": "outputs/pref-pair-humantop-hunk-then-section",
  "5": "outputs/pref-pair-humantop-hunk-replay",
  "6": "outputs/pref-joint-humantop-hunk-replay",
}

STAGE_TRAIN = {
  "2": "pair_draft",
  "3": "pair_humantop",
  "4": "pair_humantop_hunk_then_section",
  "5": "pair_humantop_hunk_replay",
  "6": "joint_humantop_hunk_replay",
}


def _run(cmd: list[str]) -> None:
  print("+ " + " ".join(cmd), flush=True)
  subprocess.run(cmd, check=True, cwd=str(ROOT))


def _python() -> str:
  venv = ROOT / ".venv" / "bin" / "python3"
  if venv.is_file():
    return str(venv)
  return sys.executable


def _train_common(args: argparse.Namespace) -> list[str]:
  cmd = [
    _python(),
    str(SCRIPTS / "train_pref_multigranular.py"),
    "--section-train-file",
    args.section_train_file,
    "--section-eval-file",
    args.section_eval_file,
    "--hunk-train-file",
    args.hunk_train_file,
    "--model",
    args.model,
    "--text-prefix",
    args.text_prefix,
    "--max-seq-length",
    str(args.max_seq_length),
    "--batch-size",
    str(args.batch_size),
    "--epochs",
    str(args.epochs),
    "--phase1-epochs",
    str(args.phase1_epochs),
    "--phase2-epochs",
    str(args.phase2_epochs),
    "--lr",
    str(args.lr),
    "--device",
    args.device,
    "--seed",
    str(args.seed),
  ]
  if args.max_train:
    cmd.extend(["--max-train", str(args.max_train)])
  if args.max_valid:
    cmd.extend(["--max-valid", str(args.max_valid)])
  return cmd


def _eval(
  *,
  kind: str,
  model: str,
  eval_file: str,
  out: Path,
  gate_model: str | None = None,
) -> dict:
  cmd = [
    _python(),
    str(SCRIPTS / "eval_pref_multigranular.py"),
    "--kind",
    kind,
    "--model",
    model,
    "--eval-file",
    eval_file,
    "--out",
    str(out),
  ]
  if gate_model:
    cmd.extend(["--gate-model", gate_model])
  _run(cmd)
  payload = json.loads(out.read_text(encoding="utf-8"))
  return payload["summary"]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stages", default="1,2,3,4,5,6,7")
  parser.add_argument("--section-train-file", default="data/section_middle/pref_train.jsonl")
  parser.add_argument("--section-eval-file", default="data/section_middle/pref_valid.jsonl")
  parser.add_argument("--hunk-train-file", default="data/pref_keep_split_hunk/train.jsonl")
  parser.add_argument("--section-keep-train", default="data/pref_keep_split_section/train.jsonl")
  parser.add_argument("--section-keep-eval", default="data/pref_keep_split_section/valid.jsonl")
  parser.add_argument("--model", default="cl-nagoya/ruri-v3-30m")
  parser.add_argument("--text-prefix", default="文章: ")
  parser.add_argument("--max-seq-length", type=int, default=256)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--phase1-epochs", type=int, default=10)
  parser.add_argument("--phase2-epochs", type=int, default=40)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--seeds", default="")
  parser.add_argument("--max-train", type=int, default=0)
  parser.add_argument("--max-valid", type=int, default=0)
  parser.add_argument("--gate-model", default="outputs/pref-bt-keep")
  parser.add_argument("--report-dir", default="outputs/pref-multigranular-report")
  args = parser.parse_args()

  stages = [s.strip() for s in args.stages.split(",") if s.strip()]
  seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]
  report_dir = Path(args.report_dir)
  report_dir.mkdir(parents=True, exist_ok=True)
  table: list[dict] = []

  for seed in seeds:
    args.seed = seed
    seed_tag = f"seed{seed}"
    for stage in stages:
      if stage == "1":
        out_dir = STAGE_OUTPUTS["1"]
        _run(
          [
            _python(),
            str(SCRIPTS / "train_pref_sentseq.py"),
            "--model",
            args.model,
            "--train-file",
            args.section_keep_train,
            "--eval-file",
            args.section_keep_eval,
            "--output-dir",
            out_dir,
            "--text-prefix",
            args.text_prefix,
            "--max-seq-length",
            str(args.max_seq_length),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--lr",
            str(args.lr),
          ]
        )
        summary = _eval(
          kind="sentseq",
          model=out_dir,
          eval_file=args.section_eval_file,
          out=report_dir / f"stage1_{seed_tag}.json",
        )
        table.append({"stage": 1, "seed": seed, "output": out_dir, **summary})
        continue
      if stage in STAGE_TRAIN:
        out_dir = STAGE_OUTPUTS[stage]
        if len(seeds) > 1:
          out_dir = f"{out_dir}-{seed_tag}"
        cmd = _train_common(args)
        cmd.extend(["--stage", STAGE_TRAIN[stage], "--output-dir", out_dir])
        _run(cmd)
        summary = _eval(
          kind="multigranular",
          model=out_dir,
          eval_file=args.section_eval_file,
          out=report_dir / f"stage{stage}_{seed_tag}.json",
        )
        table.append({"stage": int(stage), "seed": seed, "output": out_dir, **summary})
        continue
      if stage == "7":
        stage5_dir = STAGE_OUTPUTS["5"]
        if len(seeds) > 1:
          stage5_dir = f"{stage5_dir}-{seed_tag}"
        summary = _eval(
          kind="gated",
          model=stage5_dir,
          eval_file=args.section_eval_file,
          out=report_dir / f"stage7_{seed_tag}.json",
          gate_model=args.gate_model,
        )
        table.append(
          {
            "stage": 7,
            "seed": seed,
            "output": f"{stage5_dir}+{args.gate_model}",
            **summary,
          }
        )
        continue
      raise SystemExit(f"unknown stage {stage!r}")

  summary_path = report_dir / "stages_summary.json"
  summary_path.write_text(
    json.dumps({"rows": table}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(f"wrote {summary_path}", flush=True)
  art = os.environ.get("SAKURA_ARTIFACT_DIR")
  if art:
    dest = Path(art) / "pref-multigranular-report"
    dest.mkdir(parents=True, exist_ok=True)
    dest.joinpath("stages_summary.json").write_text(
      summary_path.read_text(encoding="utf-8"),
      encoding="utf-8",
    )
    for row in table:
      src = Path(row["output"].split("+")[0])
      if src.is_dir() and row["stage"] != 7:
        target = dest / src.name
        if not target.exists():
          subprocess.run(["cp", "-a", str(src), str(target)], check=True)


if __name__ == "__main__":
  main()
