#!/usr/bin/env python3
"""下書きと推敲に点数を付けるローカル Web サービス。

点数は「下書きを基準にしたマージン」で、revise_loop と同じ定義:
  margin = s(下書き, 推敲) - s(下書き, 下書き)
三軸で返す。
- 主軸（pref-sentseq-3e4）: 全体の質。合否と分布ゲージに使う。
- ゲート（pref-bt）: 細部（語彙・文単位）の悪化検出。
- 方向（pref-sentseq-anchor-2stage-v2）: アンカー学習で方向感度を持つ検出器。
  マージンが負なら「入れ替わり・改悪の疑い」を警告する（劣化版の負率 1.00、
  逆方向の負率 0.76。ただし本物の人間編集でも約 4 割は負に出るため、
  合否には使わず警告に留める）。
合格閾値の目盛りには make calibrate-margins の人間編集マージン分布を使う。

公開時の濫用防止として、IP 単位の試行回数制限と入力文字数上限を持つ。
入力本文は採点のあいだだけメモリ上に置き、保存・学習には使わない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pref_scorer import LoadedScorer, load_scorer
from sentseq_utils import split_document_sentences

# 主軸は順位付け重視の pref-sentseq-3e4。方向感度は持たないので、
# 入れ替わり・改悪の検出はアンカー学習版の方向軸が別に担う。
DEFAULT_PRIMARY = ROOT / "outputs" / "pref-sentseq-3e4"
DEFAULT_GATE = ROOT / "outputs" / "pref-bt"
DEFAULT_DIRECTION = ROOT / "outputs" / "pref-sentseq-anchor-2stage-v2"
DEFAULT_CALIBRATION = ROOT / "outputs" / "acceptance_margin_calibration.json"
INDEX_HTML = ROOT / "web" / "index.html"

# 合格ラインは self 基準の人間編集マージン中央値（make calibrate-margins）。
DEFAULT_MIN_MARGIN = float(os.environ.get("MIN_MARGIN", "3.7"))
DEFAULT_GATE_MIN_MARGIN = float(os.environ.get("GATE_MIN_MARGIN", "0.0"))

# 公開向けの濫用防止。ローカル検証では RATE_LIMIT_PER_IP=0 で無効化できる。
RATE_LIMIT_PER_IP = int(os.environ.get("RATE_LIMIT_PER_IP", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", str(24 * 3600)))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "8000"))

SENTSEQ_MAX_SENTS = 128
GITHUB_REPO_URL = "https://github.com/k16shikano/ja-tech-edit-score"

app = FastAPI(title="ja-tech-edit-score")

_scorers: dict[str, LoadedScorer] = {}
_calibration: dict = {}
_rate_lock = threading.Lock()
_rate_hits: dict[str, deque[float]] = defaultdict(deque)


def _load_models() -> None:
  primary_dir = Path(os.environ.get("PRIMARY_MODEL", str(DEFAULT_PRIMARY)))
  gate_dir = Path(os.environ.get("GATE_MODEL", str(DEFAULT_GATE)))
  print(f"loading primary: {primary_dir}", flush=True)
  _scorers["primary"] = load_scorer(primary_dir)
  print(f"loading gate: {gate_dir}", flush=True)
  _scorers["gate"] = load_scorer(gate_dir)
  direction_spec = os.environ.get("DIRECTION_MODEL", str(DEFAULT_DIRECTION))
  if direction_spec and direction_spec != "none":
    print(f"loading direction: {direction_spec}", flush=True)
    _scorers["direction"] = load_scorer(Path(direction_spec))
  calibration_path = Path(os.environ.get("CALIBRATION_PATH", str(DEFAULT_CALIBRATION)))
  if calibration_path.is_file():
    _calibration.update(json.loads(calibration_path.read_text(encoding="utf-8")))
  print("ready", flush=True)


@app.on_event("startup")
def startup() -> None:
  _load_models()


class ScoreRequest(BaseModel):
  draft: str = Field(default="", max_length=MAX_TEXT_CHARS)
  revision: str = Field(default="", max_length=MAX_TEXT_CHARS)


def client_ip(request: Request) -> str:
  forwarded = request.headers.get("x-forwarded-for", "")
  if forwarded:
    return forwarded.split(",")[0].strip() or "unknown"
  if request.client and request.client.host:
    return request.client.host
  return "unknown"


def rate_limit_status(ip: str, *, consume: bool = False) -> dict:
  """IP 単位の残り回数を返す。consume=True なら、空きがあれば今回分を消費する。"""
  if RATE_LIMIT_PER_IP <= 0:
    return {
      "enabled": False,
      "allowed": True,
      "limit": 0,
      "remaining": None,
      "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
      "reset_after_seconds": None,
    }

  now = time.time()
  cutoff = now - RATE_LIMIT_WINDOW_SECONDS
  with _rate_lock:
    hits = _rate_hits[ip]
    while hits and hits[0] < cutoff:
      hits.popleft()
    used = len(hits)
    remaining = max(0, RATE_LIMIT_PER_IP - used)
    reset_after = (
      int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now) if hits else RATE_LIMIT_WINDOW_SECONDS
    )
    if consume:
      if remaining <= 0:
        return {
          "enabled": True,
          "allowed": False,
          "limit": RATE_LIMIT_PER_IP,
          "remaining": 0,
          "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
          "reset_after_seconds": max(1, reset_after),
        }
      hits.append(now)
      remaining -= 1
      reset_after = int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now)
    return {
      "enabled": True,
      "allowed": True,
      "limit": RATE_LIMIT_PER_IP,
      "remaining": remaining,
      "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
      "reset_after_seconds": max(1, reset_after) if used or consume else RATE_LIMIT_WINDOW_SECONDS,
    }


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
def meta(request: Request) -> dict:
  result = {
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
    "notices": {
      "do_not_swap": (
        "主軸の採点は「推敲で悪くなることはない」という想定に近いため、"
        "変更があると正の評価値になりやすい。下書きと推敲を入れ替えて指定しないこと。"
      ),
      "no_storage": "入力文章は評価の計算にだけ使い、サーバに保管したり学習に使ったりしない。",
      "source_repo": GITHUB_REPO_URL,
    },
    "limits": {
      "max_text_chars": MAX_TEXT_CHARS,
    },
    "rate_limit": rate_limit_status(client_ip(request), consume=False),
  }
  if "direction" in _scorers:
    result["direction"] = {
      "scorer": _scorers["direction"].kind,
      "model_dir": _scorers["direction"].model_dir,
    }
  return result


@app.post("/api/score")
def score(req: ScoreRequest, request: Request):
  draft = req.draft.strip()
  revision = req.revision.strip()
  if not draft or not revision:
    raise HTTPException(status_code=400, detail="draft と revision の両方が必要です")
  if len(draft) > MAX_TEXT_CHARS or len(revision) > MAX_TEXT_CHARS:
    raise HTTPException(
      status_code=400,
      detail=f"下書き・推敲はそれぞれ {MAX_TEXT_CHARS} 字以内にしてください",
    )

  limit_info = rate_limit_status(client_ip(request), consume=True)
  if not limit_info["allowed"]:
    reset_after = int(limit_info["reset_after_seconds"] or RATE_LIMIT_WINDOW_SECONDS)
    return JSONResponse(
      status_code=429,
      content={
        "detail": (
          f"試行上限（{limit_info['limit']} 回 / "
          f"{limit_info['window_seconds']} 秒）に達しました。"
          f"約 {reset_after} 秒後に再度お試しください。"
        ),
        "rate_limit": limit_info,
      },
      headers={"Retry-After": str(reset_after)},
    )

  primary = axis_result("primary", draft, revision)
  gate = axis_result("gate", draft, revision)
  pass_primary = primary["margin"] >= DEFAULT_MIN_MARGIN
  pass_gate = gate["margin"] >= DEFAULT_GATE_MIN_MARGIN
  warnings = build_warnings(draft, revision)

  direction = None
  if "direction" in _scorers:
    direction = axis_result("direction", draft, revision)
    if direction["margin"] < 0:
      warnings.append(
        "方向検出器のマージンが負です。下書きと推敲の入れ替わり、または改悪の"
        "疑いがあります（ただし本物の編集でも約 4 割は負に出ます）。"
      )

  return {
    "primary": primary,
    "gate": gate,
    "direction": direction,
    "verdict": {
      "min_margin": DEFAULT_MIN_MARGIN,
      "gate_min_margin": DEFAULT_GATE_MIN_MARGIN,
      "pass_primary": pass_primary,
      "pass_gate": pass_gate,
      "accepted": pass_primary and pass_gate,
    },
    "warnings": warnings,
    "identical": draft == revision,
    "rate_limit": limit_info,
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
