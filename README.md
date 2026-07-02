# ja-tech-edit-score

日本語技術文書の **選好評価モデル**

- Git のブランチ間で、原稿と編集済みの内容との差分を学習する
- 編集中の内容について、学習したモデルに照らしたスコア（いわば編集の品質）を計測する。[docs/WORKFLOW.md](docs/WORKFLOW.md) を参照

## 想定している使い方

- 編集結果と原稿がどちらがより「良い」かを判定する（あくまでも学習したモデルに照らして）→ `bin/ja-tech-edit-score-check` コマンド
- 2つの編集結果のどちらがより「良い」かを判定する（あくまでも学習したモデルに照らして）→ `bin/ja-tech-edit-score-compare` コマンド
- フロンティアモデルなどに推敲候補をいくつか生成させて、そのうちでスコアが最上位のものを選ぶ（ここには実装は置いていない）

## リポジトリの構成

| 区分 | 内容 |
|------|------|
| 公開 | 学習と評価用スクリプト、Makefile、データスキーマ、学習済み選好評価モデル（`outputs/pref-static/`）、Cursor Skill（`skills/`） |
| ローカル | `data/` の学習データ（各自で採掘） |

- 学習済み選好評価モデルは、作者（@k16shikano）の編集内容を学習したもの
- SKILLは、Cursorでの編集中に編集結果をモデルに照らして評価させるもの

## セットアップ

リポジトリには学習済みモデル（`outputs/pref-static/`）が同梱されている。
`make train` はモデルを更新するときだけ実行すればよい。

```bash
cd ~/dev/ja-tech-edit-score
make venv
make install-bin      # ~/.local/bin に ja-tech-edit-score-check / ja-tech-edit-score-compare
make install-skills   # ~/.cursor/skills に ja-tech-edit-score-check
```

## 学習データの追加

下書きブランチ（原稿）と推敲ブランチ（編集済み）の diff から、修正前と修正後のペアを `data/examples.raw.jsonl` に追記する。

```bash
make data \
  DIR=/path/to/book-repo \
  ORG=draft-chapter \
  EDT=edit/draft-chapter
```

単一ファイルに限定する場合:

```bash
make data \
  DIR=/path/to/book-repo \
  ORG=draft-chapter \
  EDT=edit/draft-chapter \
  PATH=src/chapter.md
```

`PROJECT_ID` を省略すると、リポジトリのディレクトリ名（例: `book-repo`）を使う。
同一内容のペアは追記時にスキップされる。
複数リポジトリ、複数章は `make data` を繰り返す。

## 選好評価モデルの再学習

学習データを追加したあと、同梱モデルを更新する。

```bash
make train
```

出力先: `outputs/pref-static/`（`model.joblib`, `metrics.json`）

環境変数:

- **`EMBED_MODEL`**：既定 `hotchpotch/static-embedding-japanese`
- **`TRUNCATE_DIM`**：既定 `256`

## 2候補の比較

推敲案が二つある段落について、どちらが選好に近いかを判定する。

```bash
ja-tech-edit-score-compare \
  --source-text "修正前の段落" \
  --candidate-a "候補A" \
  --candidate-b "候補B"
```

ファイルから読む場合は `--source-file`, `--candidate-a-file`, `--candidate-b-file` を使う。

## 推敲中の選好チェック

`edit/*` ブランチ上で、作業ツリーの内容（未コミットを含む）を下書きブランチと hunk 単位で採点する。

```bash
cd /path/to/your/repo
ja-tech-edit-score-check src/chapter.md --format markdown
```

ベースブランチは `edit/foo` から `foo` を自動推定する。
コミット済み diff のみを見る場合は `--committed` を付ける。

## 推論の高速化

初回実行時にモデル常駐デーモンを自動起動する。
2回目以降はおおよそ 0.1 秒で応答する。

```bash
make daemon       # 手動で先に起動する場合
make daemon-stop  # 停止
```

デーモンを使わない場合は `--no-daemon`、または `JA_TECH_EDIT_SCORE_NO_DAEMON=1`（起動のたびに約 9 秒）。

## データパイプライン

```
make data
  git diff ORG..EDT → examples.raw.jsonl（追記）

make train
  import → DPO → curate → pref dataset → split → train_pref_static
```

## 選好評価モデルの内部構成

| 層 | 内容 |
|----|------|
| 埋め込み | `hotchpotch/static-embedding-japanese`（固定） |
| 分類器 | StandardScaler + LogisticRegression（学習） |
| 入力 | 修正前テキスト、候補 A、候補 B から構成する特徴量 |

学習済みモデル（`model.joblib`）には分類器の重みのみが含まれる。
原稿や編集済みの本文は Git または `data/examples.raw.jsonl` に残る。

## ドキュメント

| ファイル | 内容 |
|----------|------|
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | 章推敲の手順 |
| [docs/SFT-DPO.md](docs/SFT-DPO.md) | 生成モデル向け SFT / DPO 構想メモ（未実装） |
| [data/README.md](data/README.md) | 学習データディレクトリの説明 |
| [outputs/README.md](outputs/README.md) | 同梱モデルの説明 |
| [skills/README.md](skills/README.md) | Cursor Skill のインストール |
| [skills/ja-tech-edit-score-check/SKILL.md](skills/ja-tech-edit-score-check/SKILL.md) | 推敲選好チェック Skill |

## ライセンス

MIT（`LICENSE`）
