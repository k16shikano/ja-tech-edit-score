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

## ブラインド判定

- `select_blind_items.py`、`build_blind_pairs.py`、`blind_judge_server.py`、`analyze_blind_judgments.py`
- B の検証 50 件（人間の推敲対 Composer）: `build_pref_valid_gold_vs_composer_pairs.py`（`make pref-valid-blind-judge`）。集計は `eval_pref_valid50_gold_vs_composer.py`

## 教師データ B と評価器

同じ下書きの三つ組み。人が付けるのは劣化の有無だけ。

- `generate_section_composer_revisions.py`、`middle_degrade_server.py`、`build_section_triples.py`
- 文列型: `train_pref_sentseq.py`（出力は `outputs/pref-sentseq-section-triples`）
- InfoNCE: `train_pref_nce.py`（`make build-section-middle-nce-image`。出力は `outputs/pref-nce-section`）。採点は `pref_nce_runtime.py`
- 人間検出: `train_pref_detect.py`（`make build-section-middle-detect-image`。出力は `outputs/pref-detect-section`）。採点は文列型と同じ `pref_sentseq_runtime.py`
- 検出に Composer 対下書きの項を足す: 同じ `train_pref_detect.py` に `--composer-over-draft`（`make build-section-middle-detect-cd-image`。出力は `outputs/pref-detect-cd-section`）

## 一対採点の継続学習（実施済み）

- 学習: `train_pref_multigranular.py`、`run_pref_multigranular_stages.py`、`dok_pref_multigranular.sh`
- 採点: `eval_pref_multigranular.py`、`pref_multigranular_runtime.py`
- 独立人手判定の下書き固定: `freeze_8d_items.py`

## 文列型・Bradley-Terry 型

判定 Web / 順位付けの既定の主評価器は `outputs/pref-sentseq-section-triples`。ゲートは `outputs/pref-bt-keep`。

- `train_pref_sentseq.py`、`train_pref_bt.py`、`pref_scorer.py`、`score_server.py`

探索用の旧スクリプトは `scripts-old/`。
