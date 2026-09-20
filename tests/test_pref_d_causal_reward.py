import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

from pref_d_causal_reward import PREF_CAND_TOK, PREF_DRAFT_TOK, encode_pref_pair


class _Tok:
  def __init__(self) -> None:
    self.vocab = {
      PREF_DRAFT_TOK: 10,
      PREF_CAND_TOK: 11,
    }
    self.next_id = 12

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    del add_special_tokens
    if text in self.vocab:
      return [self.vocab[text]]
    out = []
    for ch in text:
      if ch not in self.vocab:
        self.vocab[ch] = self.next_id
        self.next_id += 1
      out.append(self.vocab[ch])
    return out


def test_encode_pref_pair_keeps_candidate_suffix() -> None:
  tok = _Tok()
  ids, last_idx = encode_pref_pair(tok, "abcd", "yz", max_length=8)
  assert last_idx == len(ids) - 1
  assert ids[-2:] == tok.encode("yz")
  assert tok.encode(PREF_CAND_TOK)[0] in ids


def test_encode_pref_pair_truncates_draft_first() -> None:
  tok = _Tok()
  ids, last_idx = encode_pref_pair(tok, "1234567890", "yz", max_length=6)
  assert last_idx == len(ids) - 1
  assert ids[-2:] == tok.encode("yz")
  assert tok.encode(PREF_CAND_TOK)[0] in ids
