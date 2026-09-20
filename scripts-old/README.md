# 探索用に退避したスクリプト

`docs-old/` と同じく、過去の実験の再現用である。`make help` からは外した。

動かすときはこのディレクトリを直接指定する。現行の部品（埋め込みの共通処理や `train_pref_bt.py` など）を import するものがあるので、リポジトリ根から次のようにする。

```bash
PYTHONPATH=scripts:scripts-old python scripts-old/<名前>.py
```

| 塊 | 何をしていたか |
|----|----------------|
| `build_pref_dataset.py` など | 旧い選好データ（`pref_dataset`、main 先端比較を含む）の構築と分割 |
| `train_pref_static.py` / `train_pref_ce.py` | そのデータでの線形分類器・交差符号器 |
| `build_composition_neg_pref.py`、`score_hard_eval.py`、`build_hard_eval_v2*.py` | 段落結合・分割・逆転などの難試験 |
| `generate_machine_revisions.py`、`build_machine_neg_pref.py` | 機械推敲を負例にした Bradley-Terry 再学習 |
| `extract_revision_activations.py` など | 層活性のステアリング |
| `extract_paragraph_transitions.py`、`build_structure_eval.py` | 段落遷移と構成の層別評価 |
| `build_anchor_pairs.py`、`eval_pref_*_xproject.py` | アンカー学習と書籍横断 LOPO |
| `build_dpo_dataset.py` | DPO 用データの下準備（工程 8d まで着手しない） |

採掘の書籍単位 heldout（`batch_mine_heldout_sections.sh`）もここへ移した。分割はペア単位（`scripts/rebuild_pairsplit_data.py`）が現行である。

旧い Docker イメージ定義（`Dockerfile.pref-ce` / `Dockerfile.pref-sentseq` / `Dockerfile.steering`）はリポジトリ根に残してある。中の `COPY` はこちらを指す。
