# 高火力 DOK で系統1フェーズ1（編集 SFT）を進める手順

[EDIT-MODEL.md](EDIT-MODEL.md) のフェーズ1。
推敲ペアでオープンウェイト LLM に QLoRA SFT をかけ、LoRA アダプタだけを持ち帰る。
生成品質でフロンティアを置き換えるためではなく、**同じベース＋規範スキルとの比較実験**のためである。

系統3と同じく、ファイル転送に `scp` / `rsync` は使えない。
箱（イメージ）に学習データを同梱し、レジストリ経由で DOK に渡す。

---

## 手順0: 手元の準備

```bash
cd /home/k16/dev/ja-tech-edit-score
test -s data/edit_sft/train.jsonl || make edit-sft-data
test -f Dockerfile.edit-sft && echo OK_dockerfile
```

レジストリ（例: `ja-tech-edit.sakuracr.jp`）と DOK 用のレジストリ認証は、系統3で作ったものを流用してよい。

---

## 手順1: 箱を作って倉庫へ預ける

```bash
export REGISTRY=ja-tech-edit.sakuracr.jp
chmod +x scripts/build_push_edit_sft_image.sh
./scripts/build_push_edit_sft_image.sh
```

成功すると `pushed: …/edit-sft:latest` と出る。
箱には `data/edit_sft/train.jsonl` と学習脚本が入る。
ベース LLM の重みは入らない（実行時に Hugging Face から取得）。

**非公開レジストリのみ**に push すること（原稿由来のため）。

---

## 手順2: DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（名前）.sakuracr.jp/edit-sft:latest` |
| レジストリ認証 | 登録済みのもの |
| GPU | まず **V100**。VRAM 不足や 4bit で落ちるなら **H100** |
| コマンド / エントリーポイント | **空** |
| SSH | OFF でよい |

環境変数（最初はスモーク）:

| 変数 | スモーク例 | 本番例 |
|------|------------|--------|
| `LIMIT` | `64` | `0` または未設定 |
| `EPOCHS` | `1` | `2` |
| `MODEL` | 未設定で `Qwen/Qwen3-8B` | 同じでよい |
| `LORA_R` | 未設定（16） | 必要なら `32` |

タスク終了後、課金は止まる（実行インスタンスは破棄される）。
レジストリ上のイメージ保管料は別途かかる。

---

## 手順3: 成果物を取る

アーティファクトから展開する例:

```bash
mkdir -p outputs/edit-sft
# Qwen__Qwen3-8B/adapter/ と train_meta.json をここへ
```

中身の目安:

- `adapter/`（LoRA 重みと tokenizer 設定）
- `train_meta.json`（件数・ハイパーパラのみ。原稿本文なし）

中間の `checkpoints/` は箱側でアーティファクトから除く。

---

## 手順4: held-out 評価生成（別イメージ）

学習済み LoRA を箱に同梱し、held-out 下書きを二条件で推敲する。

| mode | 内容 |
|------|------|
| `adapter` | ベース＋LoRA。SFT と同じ短い指示 |
| `base_norms` | ベースのみ。規範全文を前置した指示 |

### 4-a. build & push

```bash
export REGISTRY=ja-tech-edit.sakuracr.jp
chmod +x scripts/build_push_edit_sft_eval_image.sh
./scripts/build_push_edit_sft_eval_image.sh
```

イメージ例: `…/edit-sft-eval:latest`  
同梱: `adapter/`、`heldout.jsonl`、`tech-writing-norms.md`（いずれも非公開原稿由来）。

### 4-b. DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（名前）.sakuracr.jp/edit-sft-eval:latest` |
| GPU | V100（不足なら H100） |
| コマンド | **空** |

| 環境変数 | スモーク | 本番 |
|----------|----------|------|
| `LIMIT` | `64` | `0` |
| `EVAL_MODES` | 未設定で `adapter base_norms` | 同じでよい |
| `MAX_NEW_TOKENS` | 未設定（512） | 必要なら上げる |
| `MAX_INPUT_TOKENS` | 未設定（3072） | 規範＋下書きが長いとき注意 |
| `LOAD_IN_4BIT` | 未設定（`1`） | V100 では `1` のまま |

V100 32GB で fp16 全文読み＋規範前置だと、attention の一時領域で OOM しやすい。
既定は 4bit 読み込みと入力長上限付き。それでも落ちるなら H100、または `EVAL_MODES=adapter` のみで先に通す。

生成は **Qwen3 の思考モードを既定で無効**（`enable_thinking=False`）にする。
以前のスモークでは `base_norms` に英語 CoT（`<think>`）が大量混入していた。
採点側でも残存ブロックを落とす。意図的に思考させたいときだけ生成脚本に `--enable-thinking` を付ける。

`base_norms` のプロンプトは **推敲後本文のみ**を厳守させる（メタ前置き・解説・不当な膨張を禁止）。
対照条件として BT 比較する前に、短い `LIMIT` でメタ無し・長さ比が概ね妥当かを先に確認する。

スモークの `LIMIT` は書籍横断のラウンドロビンである。

### 4-c. 成果物を手元へ

```bash
mkdir -p outputs/edit-sft-eval
# adapter.jsonl と base_norms.jsonl をここへ
```

### 4-d. 手元で BT 採点（GPU 不要）

```bash
make edit-sft-score
# または
python scripts/score_edit_sft_eval.py \
  --eval-dir outputs/edit-sft-eval \
  --bt-model outputs/pref-bt
```

`score_report.md` の **adapter vs base_norms の勝率**を見る。
中止・続行の目安は [EDIT-MODEL.md](EDIT-MODEL.md) の「本系統の位置づけ」どおり。

---

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| CUDA OOM（学習） | H100。または `MAX_SEQ_LENGTH=1024`、`GRAD_ACCUM` を増やす |
| CUDA OOM（評価生成） | イメージを再ビルド（4bit 既定）。それでもだめなら H100、または `EVAL_MODES=adapter` のみ。`MAX_INPUT_TOKENS=2048` |
| bitsandbytes エラー | V100 の CUDA 版差。イメージの PyTorch CUDA を確認 |
| Hugging Face 取得失敗 | ネット制限・ゲート。`MODEL` を公開モデルに変える |
| 生成にメタ前置き・解説・不当な膨張 | 評価イメージを再ビルド（`base_norms` プロンプト強化・長さ上限）。短い `LIMIT` で長さ比とメタ無しを確認してから比較 |

---

## 関連ファイル

| パス | 役割 |
|------|------|
| `Dockerfile.edit-sft` | 学習用の箱 |
| `Dockerfile.edit-sft-eval` | 評価生成用の箱 |
| `scripts/train_edit_sft.py` | QLoRA SFT |
| `scripts/generate_edit_sft.py` | held-out 推敲生成 |
| `scripts/score_edit_sft_eval.py` | BT 採点（手元） |
| `scripts/dok_edit_sft.sh` | 学習の起動処理 |
| `scripts/dok_edit_sft_eval.sh` | 評価生成の起動処理 |
| `scripts/build_push_edit_sft_image.sh` | 学習イメージの push |
| `scripts/build_push_edit_sft_eval_image.sh` | 評価イメージの push |
| `data/edit_sft/train.jsonl` | 学習データ |
| `data/edit_sft/heldout.jsonl` | 評価用 held-out |
| `requirements-edit-sft.txt` | GPU 依存 |
| `docs/EDIT-MODEL.md` | 理論と位置づけ |
