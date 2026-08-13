# いま使うスクリプト

入口は `make help`。ここは、`scripts/` に残しているものの対応表である。

## 推敲前後ペア（採掘・レビュー・分割）

レビュー済みの下書きと人間の推敲を集め、ペア単位で学習と検証に分ける。

- 採掘: `mine_branch_pair.py`、`mine_section_pairs.py`、`batch_mine_sections.sh`、`git_pre_merge.py`
- レビュー: `edit_sft_review_server.py`、`export_reviewed_keeps.py`、`build_revision_corpus.py`
- 分割: `rebuild_pairsplit_data.py`、`build_pref_from_keep.py`

## 推敲モデル（SFT）

- 学習: `train_edit_sft.py`、`dok_edit_sft.sh`
- 生成: `generate_edit_sft.py`、`dok_edit_sft_eval.sh`、`generation_integrity.py`

## ブラインド判定（工程 5–7）

- `select_blind_items.py`、`build_blind_pairs.py`、`blind_judge_server.py`、`analyze_blind_judgments.py`

## 質の評価器の現行教師（工程 8-mid）

同じ下書きの三つ組み（人間の推敲 ＞ Composer 推敲 ＞ 下書き）。人が付けるのは劣化の有無だけ。

- `generate_section_composer_revisions.py`、`middle_degrade_server.py`、`build_section_triples.py`

## 既存の文列型・Bradley-Terry 型（工程 3 で学び直したもの）

下書き対人間の推敲だけで学習した評価器。Web / 順位付けの既定。質の三つ組み学習の対象ではない。

- `train_pref_sentseq.py`、`train_pref_bt.py`、`pref_scorer.py`、`score_server.py`

## 実験済み（質の教師には使わない）

工程 6 の 360 件の相対選択で文列型を学んだ実験。

- `train_generated_pref_sentseq.py`、`build_generated_pref_data.py`

探索用の旧スクリプトは `scripts-old/`。
