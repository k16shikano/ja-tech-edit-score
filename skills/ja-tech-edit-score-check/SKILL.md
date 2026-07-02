---
name: ja-tech-edit-score-check
description: >-
  開いている Markdown 原稿について、推敲ブランチの編集内容が ja-tech-edit-score 選好評価モデルに
  沿っているか hunk 単位で採点する。ユーザーが「推敲チェック」「選好チェック」
  「pref チェック」と言ったとき、または edit/* ブランチ上の章ファイルの選好スコアを
  確認したいときに使う。
disable-model-invocation: true
---

# ja-tech-edit-score 選好チェック

推敲ブランチのファイルについて、下書きブランチとの diff を hunk 単位で採点し、
**選好評価モデル**が edit 方向を支持するかを報告する。

## 前提

- ツール本体: 環境変数 `JA_TECH_EDIT_SCORE_HOME`（未設定時は clone 先。例: `~/dev/ja-tech-edit-score`）
- CLI: `ja-tech-edit-score-check`（`make install-bin` で `~/.local/bin` に symlink）
- ラッパーが PATH にない場合: `"$JA_TECH_EDIT_SCORE_HOME/bin/ja-tech-edit-score-check"`
- 学習済みモデル: `"$JA_TECH_EDIT_SCORE_HOME/outputs/pref-static/"`（リポジトリ同梱）
- 対象リポジトリは **edit/** ブランチ上であること（例: `edit/chapter-name`）
- 下書きブランチは `edit/foo` から `foo` を自動推定する（必要なら `--base`）
- **未コミットの編集を含む**（既定: `git diff <base> -- file`）。コミット済みのみは `--committed`
- 2 回目以降はモデル常駐デーモンにより高速（初回のみデーモン起動で約 9 秒）

## 手順

1. ユーザーが開いているファイル、または `@` で指定された Markdown ファイルのパスを特定する
2. パスは **絶対パス** に解決する
3. シェルで次を実行する（**必ず実際に実行し、自前解析で代用しない**）:

```bash
ja-tech-edit-score-check "/absolute/path/to/file.md" --format markdown
```

PATH に CLI がない場合:

```bash
"${JA_TECH_EDIT_SCORE_HOME:-$HOME/dev/ja-tech-edit-score}/bin/ja-tech-edit-score-check" \
  "/absolute/path/to/file.md" \
  --format markdown
```

4. 標準出力をそのままユーザーに提示する（数値サマリー、全 hunk スコア、対称評価内訳を含む）
5. `要確認` または `preferred_base` の hunk があれば、著者による確認を促す

## オプション

| オプション | 用途 |
|-----------|------|
| `--format markdown` | Cursor チャット向け（**既定でこれを使う**） |
| `--format text` | ターミナル向け |
| `--format json` | 機械処理向け |
| `--only-flagged` | 問題 hunk のみ |
| `--base BRANCH` | 下書きブランチを明示 |
| `--edit BRANCH` | 現在ブランチ名の表示用（省略時: 現在ブランチ） |
| `--committed` | 作業ツリーではなく `base..edit` のコミット済み diff のみ |
| `--no-daemon` | デーモンを使わず都度モデルを読み込む |

## 判定の読み方

| verdict | 意味 |
|---------|------|
| `preferred_edit` | 選好評価モデルは edit 方向を支持 |
| `uncertain` | 差が小さい。著者が判断 |
| `preferred_base` | 下書き側の方が選好に近い。要確認 |
| `reject_edit` | edit 方向が強く非選好 |

選好評価モデルは学習データの文体と推敲パターンに依存する。
低スコアの hunk が必ずしも誤りとは限らない。

## 2候補の比較

推敲案 A と B を比較するときは `ja-tech-edit-score-compare` を使う:

```bash
ja-tech-edit-score-compare \
  --source-file /tmp/source.txt \
  --candidate-a-file /tmp/a.txt \
  --candidate-b-file /tmp/b.txt
```

## トラブルシュート

- `model not found` → `outputs/pref-static/model.joblib` があるか確認。更新する場合は `make train`
- `cannot infer base branch` → `--base <draft-branch>` を付ける
- `no scorable hunks` → ファイルが下書きと同一（未保存または未変更）。編集後に再実行
- `--committed` 使用時は推敲ブランチにコミットが必要
- Skill が見つからない → リポジトリで `make install-skills` を実行
