from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from generated_pref_utils import (  # noqa: E402
  build_sentseq_model,
  feature_dim_for,
  preference_loss,
  score_from_doc_vectors_with_length_toggle,
)


def test_tie_loss_prefers_zero_margin() -> None:
  delta_pos = torch.tensor([2.0], requires_grad=True)
  delta_zero = torch.tensor([0.0], requires_grad=True)
  delta_neg = torch.tensor([-2.0], requires_grad=True)
  loss_pos = float(preference_loss(delta_pos, "tie").item())
  loss_zero = float(preference_loss(delta_zero, "tie").item())
  loss_neg = float(preference_loss(delta_neg, "tie").item())
  assert loss_zero < loss_pos
  assert loss_zero < loss_neg


def test_length_features_off_reduces_head_input_dim() -> None:
  d_model = 256
  assert feature_dim_for(d_model, use_length_features=True) == d_model * 4 + 5
  assert feature_dim_for(d_model, use_length_features=False) == d_model * 4 + 1

  model_on = build_sentseq_model(embed_dim=32, d_model=16, use_length_features=True)
  model_off = build_sentseq_model(embed_dim=32, d_model=16, use_length_features=False)
  assert model_on.head.linear.in_features == 16 * 4 + 5
  assert model_off.head.linear.in_features == 16 * 4 + 1

  source_vec = torch.randn(2, 16)
  candidate_vec = torch.randn(2, 16)
  len_s = torch.tensor([10.0, 20.0])
  len_c = torch.tensor([11.0, 19.0])
  score_off = score_from_doc_vectors_with_length_toggle(
    model_off,
    source_vec,
    candidate_vec,
    len_source=len_s,
    len_candidate=len_c,
    use_length_features=False,
  )
  score_on = score_from_doc_vectors_with_length_toggle(
    model_on,
    source_vec,
    candidate_vec,
    len_source=len_s,
    len_candidate=len_c,
    use_length_features=True,
  )
  assert score_off.shape == (2,)
  assert score_on.shape == (2,)


def test_preference_ab_targets() -> None:
  delta = torch.tensor([1.5], requires_grad=True)
  loss_a = preference_loss(delta, "a")
  loss_b = preference_loss(delta, "b")
  assert float(loss_a.item()) < float(loss_b.item())


def test_evaluate_generated_preferences_three_way_metrics() -> None:
  import numpy as np

  from generated_pref_utils import evaluate_generated_preferences  # noqa: PLC0415

  deltas = np.array([2.0, -2.0, 0.0, 1.0, -0.1])
  prefs = ["a", "b", "tie", "a", "tie"]
  metrics = evaluate_generated_preferences(deltas, prefs, tie_threshold=0.5)
  assert metrics["three_way_accuracy"] == 1.0
  assert metrics["three_way_macro_recall"] == 1.0
  assert metrics["recall_a"] == 1.0
  assert metrics["recall_b"] == 1.0
  assert metrics["recall_tie"] == 1.0
  assert metrics["confusion"]["a->a"] == 2
  assert metrics["confusion"]["b->b"] == 1
  assert metrics["confusion"]["tie->tie"] == 2


def test_evaluate_generated_preferences_partial_recall() -> None:
  import numpy as np

  from generated_pref_utils import evaluate_generated_preferences  # noqa: PLC0415

  deltas = np.array([2.0, -2.0, 0.0])
  prefs = ["a", "a", "tie"]
  metrics = evaluate_generated_preferences(deltas, prefs, tie_threshold=0.5)
  assert metrics["three_way_accuracy"] == 2 / 3
  assert metrics["recall_a"] == 0.5
  assert metrics["recall_tie"] == 1.0
  assert abs(metrics["three_way_macro_recall"] - 0.75) < 1e-9
