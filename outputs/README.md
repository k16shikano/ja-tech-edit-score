# 学習済み選好評価モデル（配布物）

`pref-static/` に同梱する選好評価モデルを置く。
推論（`ja-tech-edit-score-check`, `ja-tech-edit-score-compare`）はこのディレクトリを既定で参照する。

| ファイル | 内容 |
|----------|------|
| `pref-static/model.joblib` | 分類器（StandardScaler + LogisticRegression） |
| `pref-static/metrics.json` | 学習時の評価指標 |

`make train` で上書き更新する。
`make clean-model` は再学習の前処理としてこのディレクトリを削除する。

## 実験出力（ローカル、原則コミットしない）

| パス | 内容 |
|------|------|
| `pref-bt/` | Bradley-Terry 報酬（凍結 ruri + 線形ヘッド） |
| `pref-ce/` | cross-encoder 報酬（modernbert-ja-130m）。**選好評価の本線**（難試験で BT を上回る） |
| `hard_eval_report_{bt,ce}.{json,md}` | 難試験の採点レポート |
| `steering/<model_slug>/` | 系統3フェーズ A：活性とプローブ報告（原稿本文は含まない） |

`rank`（Best-of-N）と `converge --mode bt` は、`--model` / `--bt-model` に渡したディレクトリの
`meta.json` を見て BT / CE を自動判別する。`bin/ja-tech-edit-score-rank` と `make rank` は
`pref-ce/` があればそちらを既定にする（`JA_TECH_EDIT_SCORE_RANK_MODEL` / `RANK_MODEL` で上書き可）。
CE のスコアは BT とスケールが違う（マージン閾値は取り直す）。
