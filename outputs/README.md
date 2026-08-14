# 学習済み選好評価モデル（配布物）

`pref-static/` に同梱する選好評価モデルを置く。
公開 CLI（`ja-tech-edit-score-check`, `ja-tech-edit-score-compare`）はこのディレクトリを既定で参照する。

| ファイル | 内容 |
|----------|------|
| `pref-static/model.joblib` | 分類器（StandardScaler + LogisticRegression） |
| `pref-static/metrics.json` | 学習時の評価指標 |

学び直しは `scripts-old/train_pref_static.py`（`make train` は外した）。
`make clean-model` は再学習の前処理としてこのディレクトリを削除する。

## 実験出力（ローカル、原則コミットしない）

`make rank` / `make revise` / 判定 Web の既定は `pref-sentseq-keep`（節）と `pref-bt-keep`（断片のゲート）。工程 3 の学び直しは `pref-sentseq-keep-pairsplit` / `pref-bt-keep-pairsplit`。

| パス | 内容 |
|------|------|
| `pref-sentseq-keep/` | レビュー済み節ペアの文列型 |
| `pref-bt-keep/` | レビュー済み断片ペアの Bradley-Terry 型 |
| `pref-sentseq-keep-pairsplit/` | 工程 3 の分割で学び直した文列型 |
| `pref-bt-keep-pairsplit/` | 工程 3 の分割で学び直した Bradley-Terry 型 |
| `pref-sentseq-section-triples/` | 工程 8-mid の三つ組みで学び直した文列型（DOK 成果物の置き先） |
| `Qwen__Qwen3-8B-pairsplit-v2/` | 工程 3 の SFT adapter |
| `edit-sft-eval-v3/` | 工程 4 の生成 |
| `generated-pref-sentseq*` | 工程 8c。質の教師には使わない |
| `pref-bt/` / `pref-ce/` / `pref-sentseq/` など keep なし | 旧データ由来。現行の採点・採否に使わない |
