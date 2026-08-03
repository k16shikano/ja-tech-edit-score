# 高火力 DOK で系統1フェーズ1（編集 SFT）を進める手順

[EDIT-MODEL.md](EDIT-MODEL.md) のフェーズ1。
推敲ペアでオープンウェイト LLM に QLoRA SFT をかけ、LoRA アダプタだけを持ち帰る。

いまの問い：**推敲変換を LoRA で後付けできるか**（素のベースと LoRA 付きの生成比較が本線）。
adapter vs base_norms の採点器勝率は、再現用の副次スキームとして残す（下記手順4・[EDIT-MODEL.md](EDIT-MODEL.md) 履歴）。

系統3と同じく、ファイル転送に `scp` / `rsync` は使えない。
箱（イメージ）に学習データを同梱し、レジストリ経由で DOK に渡す。

---

## 手順0: 手元の準備

```bash
cd /home/k16/dev/ja-tech-edit-score
make edit-sft-export-keeps
test -s data/edit_sft_all/train.jsonl && echo OK_all
test -f Dockerfile.edit-sft && echo OK_dockerfile
# 件数: data/edit_sft_all/stats.json
```

学習に使うのは **人手レビュー keep のみ**（節 + 空行なし hunk の合流 = `data/edit_sft_all/`）。
二系統の分離管理は `edit_sft_section` / `edit_sft_hunk_nopara`。詳細は [EDIT-SFT-CORPUS.md](EDIT-SFT-CORPUS.md)。

レジストリ（さくらのコンテナレジストリ。接続先はコントロールパネルに表示される `（名前）.sakuracr.jp`）と DOK 用のレジストリ認証は、系統3で作ったものを流用してよい。

---

## 手順1: 箱を作って倉庫へ預ける

```bash
export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
chmod +x scripts/build_push_edit_sft_image.sh
./scripts/build_push_edit_sft_image.sh
```

成功すると `pushed: …/edit-sft:latest` と出る。
箱には `data/edit_sft_all/train.jsonl`（全 keep）と学習脚本が入る。
ベース LLM の重みは入らない（実行時に Hugging Face から取得）。

**非公開レジストリのみ**に push すること（原稿由来のため）。

---

## 手順2: DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（コンテナレジストリ名）.sakuracr.jp/edit-sft:latest` |
| レジストリ認証 | 登録済みのもの |
| GPU | まず **V100**。VRAM 不足や 4bit で落ちるなら **H100** |
| コマンド / エントリーポイント | **空** |
| SSH | バッチなら OFF でよい。対話試用は [DOK-EDIT-SFT-CHAT.md](DOK-EDIT-SFT-CHAT.md)（SSH ON） |

環境変数（イメージ既定。上書き可）:

| 変数 | 既定 | 意味 |
|------|------|------|
| `LIMIT` | `0` | 全件学習 |
| `EPOCHS` | `15` | 有効バッチ 8 で更新おおよそ 3200 回（train 1717）。各ペア約 15 周 |
| `MAX_SEQ_LENGTH` | `8192` | 現行の採用ペアは全員 8192 トークン未満（2048 では約 17% が切れる） |
| `MODEL` | `Qwen/Qwen3-8B` | |
| `LORA_R` | `16` | 前回と同じ |
| `LEARNING_RATE` | `2e-4` | 前回 Qwen3-8B QLoRA と同じ。下げない（下記） |

#### 学習率を 2e-4 のままにする理由

この実験の問いは「推敲変換を LoRA で後付けできるか」である。  
リポジトリ内で Qwen3-8B に LoRA が載った既知の設定は、前回 hunk SFT の **lr=2e-4・r=16・batch 1・accum 8** だけである。

| | 前回（壊れたデータ） | 今回（全 keep） |
|--|--|--|
| 件数 | 5317 | 1717（train）+ heldout 259 |
| エポック | 2 | 15 |
| 更新回数（目安） | 約 1300 | 約 3220（1717÷8×15） |
| 各ペアの周回 | 2 | 15 |
| 学習率 | 2e-4 | **2e-4（据え置き）** |

更新回数は前回より多い。ここでも lr は前回と揃える。  
件数・系列長・エポックはデータ側の必然として変え、**LoRA が動く側の lr は前回と揃える**。

15 周は覚え込みやすい。それは held-out との生成比較で読む。lr を先に下げて偽陰性を増やす方が、この問いには悪い。

GPU は V100 で試し、8192 で OOM なら H100。

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

学習済み LoRA を箱に同梱し、held-out 下書きを推敲する。

**本線の読み方**：同じ短い指示で、素のベース（または adapter）と人間稿（`gold`）を見比べ、「後付けできたか」を判断する。
採点器による adapter vs base_norms は副次（再現用）。

| mode | 内容 |
|------|------|
| `adapter` | ベース＋LoRA。SFT と同じ短い指示 |
| `base_norms` | ベースのみ。規範全文を前置した指示（副次比較用） |

### 4-a. build & push

```bash
export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
chmod +x scripts/build_push_edit_sft_eval_image.sh
./scripts/build_push_edit_sft_eval_image.sh
```

イメージ例: `…/edit-sft-eval:latest`  
同梱: `adapter/`、`heldout.jsonl`、`tech-writing-norms.md`（いずれも非公開原稿由来）。

### 4-b. DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（コンテナレジストリ名）.sakuracr.jp/edit-sft-eval:latest` |
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

### 4-d. 手元で採点（GPU 不要）

```bash
make edit-sft-score
# または（pref-bt / pref-sentseq など）
PYTHONPATH=scripts python scripts/score_edit_sft_eval.py \
  --eval-dir outputs/edit-sft-eval \
  --model outputs/pref-bt
PYTHONPATH=scripts python scripts/score_edit_sft_eval.py \
  --eval-dir outputs/edit-sft-eval \
  --model outputs/pref-sentseq-3e4 \
  --report outputs/edit-sft-eval/score_report_sentseq.json
```

`score_report.md` の **adapter vs base_norms の勝率**は副次指標。
中止・続行の本線は [EDIT-MODEL.md](EDIT-MODEL.md) の「後付けできたか」である。

### 履歴：旧 hunk 教師での採点（信頼しない）

壊れた hunk 教師の LoRA に対する再採点（2026-07-30）: 生成は `outputs/edit-sft/ef742b08-..._artifact/` を再利用し、
pref-bt 勝率 0.269、pref-sentseq-3e4 勝率 0.276。
手順4のスキーム再現用の記録であり、現行結論には使わない。詳細は [EDIT-MODEL.md](EDIT-MODEL.md) 履歴。

---

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| CUDA OOM（学習） | H100。`MAX_SEQ_LENGTH` を下げない（2048 以下は節が切れる）。どうしても V100 だけなら `GRAD_ACCUM` を増やすかバッチを維持したまま GPU を上げる |
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
| `scripts/score_edit_sft_eval.py` | 採点（手元。`--model` で bt / sentseq 等） |
| `scripts/dok_edit_sft.sh` | 学習の起動処理 |
| `scripts/dok_edit_sft_eval.sh` | 評価生成の起動処理 |
| `scripts/build_push_edit_sft_image.sh` | 学習イメージの push |
| `scripts/build_push_edit_sft_eval_image.sh` | 評価イメージの push |
| `data/edit_sft_all/train.jsonl` | 学習データ（全 keep = 節+空行なし） |
| `data/edit_sft_all/heldout.jsonl` | 評価用 held-out |
| `data/edit_sft_section/` / `data/edit_sft_hunk_nopara/` | 系統別の保管 |
| `requirements-edit-sft.txt` | GPU 依存 |
| `docs/EDIT-MODEL.md` | 理論と位置づけ |
