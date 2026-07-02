# 学習済み選好評価モデル（配布物）

`pref-static/` に同梱する選好評価モデルを置く。
推論（`ja-tech-edit-score-check`, `ja-tech-edit-score-compare`）はこのディレクトリを既定で参照する。

| ファイル | 内容 |
|----------|------|
| `pref-static/model.joblib` | 分類器（StandardScaler + LogisticRegression） |
| `pref-static/metrics.json` | 学習時の評価指標 |

`make train` で上書き更新する。
`make clean-model` は再学習の前処理としてこのディレクトリを削除する。
