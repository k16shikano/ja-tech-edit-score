# 学習データ（ローカル、コミットしない）

プロジェクト固有の学習成果物を置くディレクトリ。

## 生成されるファイル（gitignore 対象）

- **`examples.raw.jsonl`**：ブランチ diff から採掘した修正前と修正後のペア
- **`examples.db`**：SQLite への取り込み結果
- `dpo_dataset.jsonl`, `dpo_curated.jsonl`, `pref_dataset.jsonl`
- **`pref_split/`**：train / valid / test 分割
- `curate_report.json`

## リポジトリに含まれるテンプレート

- `examples.schema.json`
- `examples.template.jsonl`
- `batch_import_repos.example.txt`（一括採掘用。コピーして `batch_import_repos.txt` を作成）

## ローカル設定（gitignore）

- `batch_import_repos.txt`：一括採掘するリポジトリパスの一覧

生成はリポジトリルートで `make data` を実行する。
選好評価モデルの再学習は `make train`（出力は `outputs/pref-static/` を上書きする）。
推論に使うモデルはリポジトリ同梱の `outputs/pref-static/` を参照する。
