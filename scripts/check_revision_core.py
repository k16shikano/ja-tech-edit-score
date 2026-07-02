#!/usr/bin/env python3
"""Run check_revision logic; shared by CLI and daemon."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from git_branch_utils import (
  current_branch,
  describe_edit_side,
  detect_base_branch,
  head_commit,
  rel_path_in_repo,
  repo_root_for,
  resolve_commit,
  worktree_dirty_for,
)
from mine_branch_pair import Example, infer_project_id, is_usable_pair, mine_file_diff, mine_worktree_diff
from pref_runtime import LoadedPrefModel, load_pref_model, score_edit_vs_base_batch


@dataclass
class HunkScore:
  index: int
  line_old: int | None
  line_new: int | None
  score_edit: float
  score_base: float
  margin: float
  verdict: str
  source_preview: str
  edited_preview: str
  source_reference: str
  pref_detail: dict[str, float]


def default_home() -> Path:
  import os

  return Path(os.environ.get("JA_TECH_EDIT_SCORE_HOME", Path.home() / "dev/ja-tech-edit-score")).expanduser()


def default_model_dir() -> Path:
  import os

  override = os.environ.get("JA_TECH_EDIT_SCORE_MODEL", "").strip()
  if override:
    return Path(override).expanduser()
  return default_home() / "outputs/pref-static"


def verdict_from_margin(margin: float, *, uncertain: float, reject: float) -> str:
  if margin >= uncertain:
    return "preferred_edit"
  if margin <= -reject:
    return "reject_edit"
  if margin <= -uncertain:
    return "preferred_base"
  return "uncertain"


def preview(text: str, limit: int = 120) -> str:
  one_line = " ".join(text.split())
  if len(one_line) <= limit:
    return one_line
  return one_line[: limit - 1] + "…"


def load_model_metrics(model_dir: Path) -> dict:
  metrics_path = model_dir / "metrics.json"
  if not metrics_path.is_file():
    return {}
  try:
    return json.loads(metrics_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return {}


def compute_score_stats(hunks: list[HunkScore]) -> dict[str, float | int]:
  if not hunks:
    return {}
  margins = [h.margin for h in hunks]
  edits = [h.score_edit for h in hunks]
  bases = [h.score_base for h in hunks]
  n = len(hunks)
  return {
    "hunk_count": n,
    "mean_margin": sum(margins) / n,
    "min_margin": min(margins),
    "max_margin": max(margins),
    "mean_score_edit": sum(edits) / n,
    "mean_score_base": sum(bases) / n,
    "min_score_edit": min(edits),
    "max_score_edit": max(edits),
    "edit_win_rate": sum(1 for h in hunks if h.margin > 0) / n,
    "file_score_edit": sum(edits) / n,
  }


def summarize(hunks: list[HunkScore]) -> dict[str, int]:
  counts = {
    "total": len(hunks),
    "preferred_edit": 0,
    "uncertain": 0,
    "preferred_base": 0,
    "reject_edit": 0,
  }
  for hunk in hunks:
    counts[hunk.verdict] = counts.get(hunk.verdict, 0) + 1
  return counts


def collect_examples(params: dict[str, Any]) -> tuple[list[Example], dict[str, Any]]:
  file_path = Path(params["file"]).expanduser()
  repo = Path(params.get("repo") or "").expanduser().resolve() if params.get("repo") else repo_root_for(file_path)
  if not (repo / ".git").exists():
    raise ValueError(f"not a git repository: {repo}")

  edit_branch = (params.get("edit") or "").strip() or current_branch(repo)
  base_branch = (params.get("base") or "").strip() or detect_base_branch(repo, edit_branch or "HEAD")
  rel_path = rel_path_in_repo(repo, params["file"])
  project_id = infer_project_id(repo, (params.get("project_id") or "").strip() or None)
  committed = bool(params.get("committed"))

  base_commit = resolve_commit(repo, base_branch)
  if committed:
    if not edit_branch:
      raise ValueError("detached HEAD; pass edit for committed mode")
    edit_commit = resolve_commit(repo, edit_branch)
    examples = mine_file_diff(
      repo,
      base=base_branch,
      edit=edit_branch,
      path=rel_path,
      project_id=project_id,
      base_commit=base_commit,
      edit_commit=edit_commit,
    )
    diff_mode = "committed (base..edit)"
    compare_label = f"{base_branch} → {edit_branch}"
    edit_commit_meta = edit_commit[:8]
  else:
    examples = mine_worktree_diff(
      repo,
      base=base_branch,
      path=rel_path,
      project_id=project_id,
      edit_branch=edit_branch,
    )
    edit_side, edit_token = describe_edit_side(repo, rel_path, branch=edit_branch or "HEAD")
    diff_mode = "working-tree (base vs 作業ツリー, 未コミット含む)"
    compare_label = f"{base_branch} → {edit_side}"
    edit_commit_meta = edit_token

  if not examples:
    hint = ""
    if not committed and not worktree_dirty_for(repo, rel_path):
      hint = (
        f" (no uncommitted diff vs {base_branch!r}; edit file or use committed mode)"
      )
    raise ValueError(f"no scorable hunks for {rel_path}{hint}")

  meta = {
    "file": rel_path,
    "repo": str(repo),
    "base_branch": base_branch,
    "edit_branch": edit_branch or None,
    "compare_label": compare_label,
    "diff_mode": diff_mode,
    "base_commit": base_commit[:8],
    "edit_commit": edit_commit_meta,
    "head_commit": head_commit(repo)[:8],
    "worktree_dirty": worktree_dirty_for(repo, rel_path),
    "project_id": project_id,
  }
  return examples, meta


def score_examples_batch(
  examples: list[Example],
  loaded: LoadedPrefModel,
  *,
  uncertain_threshold: float,
  reject_threshold: float,
) -> list[HunkScore]:
  usable = [ex for ex in examples if is_usable_pair(ex.source_text, ex.edited_text)]
  if not usable:
    return []

  pairs = [(ex.source_text, ex.edited_text) for ex in usable]
  scored = score_edit_vs_base_batch(pairs, loaded)

  rows: list[HunkScore] = []
  for example, (score_edit, score_base, pref_detail) in zip(usable, scored, strict=True):
    margin = score_edit - score_base
    rows.append(
      HunkScore(
        index=len(rows) + 1,
        line_old=example.line_old,
        line_new=example.line_new,
        score_edit=score_edit,
        score_base=score_base,
        margin=margin,
        verdict=verdict_from_margin(
          margin,
          uncertain=uncertain_threshold,
          reject=reject_threshold,
        ),
        source_preview=preview(example.source_text),
        edited_preview=preview(example.edited_text),
        source_reference=example.source_reference,
        pref_detail=pref_detail or {},
      )
    )
  return rows


def build_check_payload(
  params: dict[str, Any],
  loaded: LoadedPrefModel,
) -> dict[str, Any]:
  model_dir = Path(params.get("model") or default_model_dir())
  uncertain_threshold = float(params.get("uncertain_threshold", 0.12))
  reject_threshold = float(params.get("reject_threshold", 0.25))
  only_flagged = bool(params.get("only_flagged"))

  examples, meta = collect_examples(params)
  hunks = score_examples_batch(
    examples,
    loaded,
    uncertain_threshold=uncertain_threshold,
    reject_threshold=reject_threshold,
  )
  if not hunks:
    raise ValueError("no scorable hunks after filtering")

  summary = summarize(hunks)
  statistics = compute_score_stats(hunks)
  model_metrics = load_model_metrics(model_dir)
  thresholds = {"uncertain": uncertain_threshold, "reject": reject_threshold}
  hunk_payload = [asdict(h) for h in hunks]
  if only_flagged:
    hunk_payload = [h for h in hunk_payload if h["verdict"] != "preferred_edit"]

  return {
    "meta": {
      **meta,
      "model_dir": str(model_dir),
    },
    "summary": summary,
    "statistics": statistics,
    "thresholds": thresholds,
    "model_metrics": {
      k: model_metrics[k]
      for k in ("eval_accuracy", "eval_log_loss", "eval_samples", "train_samples", "embedding_model")
      if k in model_metrics
    },
    "hunks": hunk_payload,
  }
