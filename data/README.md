# 学習データ（ローカル、コミットしない）

プロジェクト固有の学習成果物を置くディレクトリ。

## 生成されるファイル（gitignore 対象）

- **`examples.raw.jsonl`**：ブランチ diff から採掘した修正前と修正後のペア（**hunk 単位**。空行が落ち、段落境界が消える）
- **`examples.section.raw.jsonl`**：節（見出し単位）の source/edited ペア（`make mine-sections`）。段落境界を保持
- **`section_mining_manifest.json`**：過去採掘の repo / branch / path 一覧（再採掘用）
- **`examples.db`**：SQLite への取り込み結果
- `dpo_dataset.jsonl`, `dpo_curated.jsonl`, `pref_dataset.jsonl`
- **`revision_pairs.jsonl`**：steering / 編集モデル用の draft・revised 対照（`make steering-pairs`）
- **`pref_split/`**：train / valid / test 分割
- `curate_report.json`

## リポジトリに含まれるテンプレート

- `examples.schema.json`
- `examples.template.jsonl`
- `batch_import_repos.example.txt`（一括採掘用。コピーして `batch_import_repos.txt` を作成）

## ローカル設定（gitignore）

- `batch_import_repos.txt`：一括採掘するリポジトリパスの一覧

生成はリポジトリルートで `make data`（hunk）または `make mine-sections`（節）を実行する。
選好データの現行は `make pairsplit-data` / `make pref-keep-data`（`pref_keep_split_hunk/`、`pref_keep_split_section/`）。
教師データ B の三つ組みは `section_middle/`。
A1 学習側の三群生成で使う japanese-tech-writing のコピーは `a1_probe/`。
作り方の計画は [docs/DATA.md](../docs/DATA.md)。
旧い `pref_dataset` と `make train`（pref-static）は `scripts-old/`。
推論の公開 CLI は同梱の `outputs/pref-static/` を参照する。

## 選抜難試験（Hard Eval）

- スキーマ: `hard_eval.schema.json`
- テンプレート: `hard_eval.template.jsonl`（例文は架空。原稿の段落を貼らない）
- ラベル済み実データ: `hard_eval/`（gitignore。手順は [docs-old/HARD-EVAL.md](../docs-old/HARD-EVAL.md)）
