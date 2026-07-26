#!/usr/bin/env python3
"""下書きと推敲に点数を付けるローカル Web サービス。

点数は「下書きを基準にしたマージン」で、revise_loop と同じ定義:
  margin = s(下書き, 推敲) - s(下書き, 下書き)
pref-sentseq（主軸）と pref-bt（ゲート）の二軸で返す。
合格閾値の目盛りには make calibrate-margins の人間編集マージン分布を使う。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pref_scorer import LoadedScorer, load_scorer
from sentseq_utils import split_document_sentences

# 主軸はアンカー学習版（方向感度あり: 逆方向・劣化に負のマージンを返す）。
# 順位付け重視の旧主軸に戻すときは PRIMARY_MODEL=outputs/pref-sentseq-3e4 と
# CALIBRATION_PATH=outputs/acceptance_margin_calibration.json を指定する。
DEFAULT_PRIMARY = ROOT / "outputs" / "pref-sentseq-anchor"
DEFAULT_GATE = ROOT / "outputs" / "pref-bt"
DEFAULT_CALIBRATION = ROOT / "outputs" / "acceptance_margin_calibration_anchor.json"
INDEX_HTML = ROOT / "web" / "index.html"

# アンカー版はマージンの符号が改善/悪化の向きを持つため、合格ラインは 0。
# 人間編集の self 基準分布（valid 651 ペア）は中央値 0.14、正の率 0.60 で、
# 小さな実編集も負に出ることがある点は分布ゲージで補って読む。
DEFAULT_MIN_MARGIN = float(os.environ.get("MIN_MARGIN", "0.0"))
DEFAULT_GATE_MIN_MARGIN = float(os.environ.get("GATE_MIN_MARGIN", "0.0"))

SENTSEQ_MAX_SENTS = 128

app = FastAPI(title="ja-tech-edit-score")

_scorers: dict[str, LoadedScorer] = {}
_calibration: dict = {}


def _load_models() -> None:
  primary_dir = Path(os.environ.get("PRIMARY_MODEL", str(DEFAULT_PRIMARY)))
  gate_dir = Path(os.environ.get("GATE_MODEL", str(DEFAULT_GATE)))
  print(f"loading primary: {primary_dir}", flush=True)
  _scorers["primary"] = load_scorer(primary_dir)
  print(f"loading gate: {gate_dir}", flush=True)
  _scorers["gate"] = load_scorer(gate_dir)
  calibration_path = Path(os.environ.get("CALIBRATION_PATH", str(DEFAULT_CALIBRATION)))
  if calibration_path.is_file():
    _calibration.update(json.loads(calibration_path.read_text(encoding="utf-8")))
  print("ready", flush=True)


@app.on_event("startup")
def startup() -> None:
  _load_models()


class ScoreRequest(BaseModel):
  draft: str
  revision: str


def percentile_position(margin: float, percentiles: dict[str, float]) -> float:
  """人間編集マージン分布のどの百分位に相当するかを、区分線形補間で返す。"""
  points = sorted((float(name[1:]), value) for name, value in percentiles.items())
  if margin <= points[0][1]:
    return points[0][0]
  if margin >= points[-1][1]:
    return points[-1][0]
  for (p_lo, v_lo), (p_hi, v_hi) in zip(points, points[1:]):
    if v_lo <= margin <= v_hi:
      if v_hi == v_lo:
        return p_hi
      return p_lo + (p_hi - p_lo) * (margin - v_lo) / (v_hi - v_lo)
  return points[-1][0]


def axis_result(key: str, draft: str, revision: str) -> dict:
  scorer = _scorers[key]
  scores = scorer.score(draft, [revision, draft], batch_size=4)
  margin = float(scores[0] - scores[1])
  result = {
    "scorer": scorer.kind,
    "model_dir": str(Path(scorer.model_dir).relative_to(ROOT))
    if scorer.model_dir.startswith(str(ROOT))
    else scorer.model_dir,
    "margin": round(margin, 4),
  }
  # 目盛りは運用時と同じ self 基準（基準=下書き自身）の人間編集マージン分布
  calib = _calibration.get(key, {}).get("self_baseline", {})
  if calib.get("percentiles"):
    result["percentiles"] = calib["percentiles"]
    result["percentile_position"] = round(
      percentile_position(margin, calib["percentiles"]), 1
    )
  return result


def build_warnings(draft: str, revision: str) -> list[str]:
  warnings: list[str] = []
  for label, text in (("下書き", draft), ("推敲", revision)):
    n_sents = len(split_document_sentences(text))
    if n_sents > SENTSEQ_MAX_SENTS:
      warnings.append(
        f"{label}が {n_sents} 文あり、pref-sentseq は先頭 {SENTSEQ_MAX_SENTS} 文で"
        "切り詰めて採点します。節単位に分けての採点を推奨します。"
      )
    if len(text) > 1500:
      warnings.append(
        f"{label}が {len(text)} 字あり、pref-bt は最大系列長 512 トークンを超えた"
        "部分を読みません。"
      )
  return warnings


@app.get("/")
def index() -> FileResponse:
  return FileResponse(INDEX_HTML)


@app.get("/api/meta")
def meta() -> dict:
  return {
    "primary": {
      "scorer": _scorers["primary"].kind,
      "model_dir": _scorers["primary"].model_dir,
      "min_margin": DEFAULT_MIN_MARGIN,
      "calibration": _calibration.get("primary"),
    },
    "gate": {
      "scorer": _scorers["gate"].kind,
      "model_dir": _scorers["gate"].model_dir,
      "min_margin": DEFAULT_GATE_MIN_MARGIN,
      "calibration": _calibration.get("gate"),
    },
  }


@app.post("/api/score")
def score(req: ScoreRequest) -> dict:
  draft = req.draft.strip()
  revision = req.revision.strip()
  if not draft or not revision:
    raise HTTPException(status_code=400, detail="draft と revision の両方が必要です")

  primary = axis_result("primary", draft, revision)
  gate = axis_result("gate", draft, revision)
  pass_primary = primary["margin"] >= DEFAULT_MIN_MARGIN
  pass_gate = gate["margin"] >= DEFAULT_GATE_MIN_MARGIN
  return {
    "primary": primary,
    "gate": gate,
    "verdict": {
      "min_margin": DEFAULT_MIN_MARGIN,
      "gate_min_margin": DEFAULT_GATE_MIN_MARGIN,
      "pass_primary": pass_primary,
      "pass_gate": pass_gate,
      "accepted": pass_primary and pass_gate,
    },
    "warnings": build_warnings(draft, revision),
    "identical": draft == revision,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8300)
  args = parser.parse_args()

  import uvicorn

  uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
  main()
