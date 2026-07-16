# 高火力 DOK で系統3フェーズAを進める手順

この文書だけ読めば足りるように書く。
用語は初出で説明し、さくらの画面操作と手元コマンドを交互に示す。

## いまやりたいこと（1行）

推敲ペアを GPU 上の言語モデルに通して「層ごとの内部表現」をとり、あとで手元の CPU で線形プローブする。

この段階ではモデルを学習（重み更新）しない。読むだけである。

## なぜ普通にファイルを送れないか

高火力 DOK の SSH は「中でコマンドを打つため」のもので、ファイル転送（`scp` / `rsync`）は公式に使えない。

したがってデータの入れ方はマニュアルどおり次のどちらかになる。

- あらかじめ **実行に必要なファイルを入れた箱（コンテナイメージ）** を作っておき、その箱を DOK で起動する
- または外部ストレージから取る（今回は使わない）

この手順は前者だけを使う。

## 登場するもの（この4つだけ覚える）

| 呼び名 | なにか |
|--------|--------|
| **コンテナイメージ** | プログラム・データ・実行手順をまとめた「箱」。Docker で作る。 |
| **コンテナレジストリ** | その箱を置いておくさくら側の倉庫。DOK はここから箱を取る。 |
| **高火力 DOK タスク** | 倉庫の箱を GPU 付きマシンで一度実行する依頼。終わると消える。 |
| **アーティファクト** | タスクが残した生成物。終了後に画面からダウンロードできる。 |

オブジェクトストレージは使わない。AWS も出さない。

---

## 全体の流れ（地図）

1. 手元で推敲ペアファイルがあることを確認する  
2. さくらに「箱の倉庫」（コンテナレジストリ）を一度作る  
3. 手元で箱を作り、倉庫に預ける  
4. DOK に「この箱を GPU で実行して」と依頼する  
5. 終わったら成果物（活性ファイル）をダウンロードする  
6. 手元でプローブする  

いま動いている対話用ノートブックは、この用途では使わない。転送できないからである。余計な課金を止めるなら終了してよい。

---

## 手順0: 手元の準備確認

リポジトリ:

```text
/home/k16/dev/ja-tech-edit-score
```

次があること。

```bash
cd /home/k16/dev/ja-tech-edit-score
test -s data/revision_pairs.jsonl && echo OK_pairs
test -f Dockerfile.steering && echo OK_dockerfile
```

`OK_pairs` が出ないときだけ:

```bash
make steering-pairs
```

Docker はこのマシンに入っている前提で進める（未導入なら先に Docker Engine を入れる）。

---

## 手順1: さくらの「箱の倉庫」を作る（初回だけ）

ブラウザで **さくらのクラウド** にログインする。

1. 左メニューから **コンテナレジストリ** を開く  
   （メニュー名が「グローバル」や「LAB」配下のこともある。なければ画面上部検索で「コンテナレジストリ」）
2. **追加** でレジストリを作る  
   - 名前: わかりやすい任意名（例: `ja-tech-edit`）  
   - 公開設定: **非公開**（原稿由来のデータが入るため）
3. 作成したレジストリを開き、**ユーザ** を追加する  
   - ユーザ名とパスワードを決めて控える（あとで `docker login` と DOK 認証で使う）  
   - 権限は push と pull ができるもの（例: All）

控える情報はこの3つ。

- レジストリのホスト名: `（あなたが付けた名前）.sakuracr.jp`  
- ユーザ名  
- パスワード  

詳細は公式: [コンテナレジストリ](https://manual.sakura.ad.jp/cloud/appliance/container-registry/index.html)

---

## 手順2: 手元で箱を作り、倉庫へ預ける

ターミナル（手元）で、手順1の値に置き換えて実行する。

```bash
cd /home/k16/dev/ja-tech-edit-score

# 倉庫へのログイン（ユーザ名・パスワードは手順1）
docker login （名前）.sakuracr.jp

# 箱を作って倉庫へ送る
export REGISTRY=（名前）.sakuracr.jp
./scripts/build_push_steering_image.sh
```

成功すると、だいたい次のように表示される。

```text
pushed: （名前）.sakuracr.jp/steering-phase-a:latest
```

箱の中身:

- 推敲ペア `revision_pairs.jsonl`
- 抽出スクリプト
- 起動時に「抽出して成果をアーティファクト置き場へコピーする」処理

言語モデル本体（数GB〜）は箱に入れていない。実行時に Hugging Face からダウンロードする。

ビルドには時間がかかることがある（依存パッケージのインストール）。

---

## 手順3: 高火力 DOK でタスクを作る

ブラウザで **高火力 DOK** を開く（さくらのクラウドから入れる）。

### 3-a. レジストリ認証を DOK に教える（初回）

DOK の設定に「コンテナレジストリの認証情報」を登録する画面がある。
そこで:

- ホスト名: `（名前）.sakuracr.jp`
- ユーザ名 / パスワード: 手順1のもの

を登録する。これがないと DOK が非公開の箱を取れない。

### 3-b. タスク新規作成

次を指定する。

| 項目 | 入れるもの |
|------|------------|
| 名前 | 任意（例: `steering-phase-a-smoke`） |
| イメージ | `（名前）.sakuracr.jp/steering-phase-a:latest` |
| レジストリ認証 | いま登録したもの |
| GPUプラン | まず **V100**（足りなければあとで H100） |
| コマンド | **空のまま**（箱の起動処理が自動で走る） |
| エントリーポイント | **空のまま** |
| 環境変数 | 最初の確認用に `LIMIT` = `64` を追加。読み取り方は `PROMPT_MODES`（既定 `reading norms`。初回の mean-pool を再取得するなら `none` を加える） |
| SSH | OFF でよい（ファイル転送には使えない。今回不要） |

`LIMIT` は書籍横断のラウンドロビンで選ばれるため、スモークでも複数書籍が入り、
手元プローブ（leave-one-project-out）がそのまま動く。

作成／実行する。

動いているあいだ課金される。起動〜終了までが対象。

### 3-c. 最初はスモーク、通ったら本番

`LIMIT=64` で最後まで行き、アーティファクトが取れたら成功。

同じ手順でタスクを再作成し、環境変数 `LIMIT` を `0` にするか削除して全件（約6055ペア）を回す。

公式のタスク説明: [タスクの実行方法](https://manual.sakura.ad.jp/cloud/koukaryoku-container/running-tasks.html)

---

## 手順4: 成果物を取る

1. タスクが成功終了するまで待つ  
2. タスクの **詳細** を開く  
3. **アーティファクト** から `tar.gz` などをダウンロードする（保管期限の目安は 72 時間）  

中身の目安（`PROMPT_MODES` の各モードごとに 1 ディレクトリ）:

- `Qwen__Qwen3-8B--reading/activations.npz` と `meta.json`
- `Qwen__Qwen3-8B--norms/activations.npz` と `meta.json`
- （`none` を含めた場合は `Qwen__Qwen3-8B/`）

本文は入っていない。

手元リポジトリに置く例:

```bash
cd /home/k16/dev/ja-tech-edit-score
mkdir -p outputs/steering
# ダウンロードした中身を outputs/steering/ へ展開・コピー
# （Qwen__Qwen3-8B--reading などのディレクトリごと置く）
```

---

## 手順5: 手元でプローブ（GPU不要）

```bash
cd /home/k16/dev/ja-tech-edit-score
make steering-probe MODEL=Qwen/Qwen3-8B VARIANT=reading
make steering-probe MODEL=Qwen/Qwen3-8B VARIANT=norms
```

各ディレクトリの `probe_report.md` を読む。
`pair`（差分符号当て）の最良層 micro が、だいたい 0.95 前後に近いかを見る（中止判定の目安）。
第1回（mean-pool）は single 0.54 / pair 0.64 だった。これを上回るかが焦点である。

---

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| `docker login` 失敗 | レジストリ名・ユーザ・パスワード、公開設定が非公開なのに認証間違いないか |
| DOK がイメージを取れない | DOK 側のレジストリ認証登録漏れ |
| GPU メモリ不足で落ちる | プランを H100 にする。または `BATCH_SIZE=1`（既定）のまま `MAX_LENGTH=1024` を環境変数で試す |
| Hugging Face のダウンロード失敗 | ネット制限やモデルゲート。別の公開モデル ID を環境変数 `MODEL` で指定 |
| アーティファクトが空 | タスクログに Python エラーが出ていないか。成功終了しているか |

---

## やらないこと（この手順では不要）

- SSH で `scp` / `rsync` / `tar \| ssh`
- オブジェクトストレージ
- JupyterLab へのアップロード
- 対話ノート上での長時間手作業

---

## 関連ファイル（リポジトリ内）

| パス | 役割 |
|------|------|
| `Dockerfile.steering` | 箱の中身の定義 |
| `scripts/dok_steering_extract.sh` | 箱起動時に走る処理（`PROMPT_MODES` の各モードを順に抽出） |
| `scripts/build_push_steering_image.sh` | 箱を作って倉庫へ送る |
| `data/revision_pairs.jsonl` | 推敲ペア（箱に入る） |
| `data/tech-writing-norms.md` | 判定プロンプトに前置する文章規範（箱に入る） |
| `docs/ACTIVATION-STEERING.md` | 理論とフェーズ全体 |
