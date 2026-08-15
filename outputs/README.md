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

`make rank` / `make revise` / 判定 Web の既定は `pref-sentseq-section-triples`（教師データ B の文列型）と `pref-bt-keep`（断片のゲート）。ペア単位分割の学び直しは `pref-sentseq-keep-pairsplit` / `pref-bt-keep-pairsplit`。

| パス | 内容 |
|------|------|
| `pref-sentseq-keep/` | レビュー済み節ペアの文列型 |
| `pref-bt-keep/` | レビュー済み断片ペアの Bradley-Terry 型 |
| `pref-sentseq-keep-pairsplit/` | ペア単位分割で学び直した文列型 |
| `pref-bt-keep-pairsplit/` | ペア単位分割で学び直した Bradley-Terry 型 |
| `pref-sentseq-section-triples/` | 教師データ B の文列型 |
| `pref-nce-section/` | InfoNCE |
| `pref-detect-section/` | 人間検出 |
| `pref-detect-cd-section/` | 人間検出に Composer 対下書きの項を足したもの |
| `Qwen__Qwen3-8B-pairsplit-v2/` | SFT adapter |
| `edit-sft-eval-v3/` | 生成器の評価生成 |
