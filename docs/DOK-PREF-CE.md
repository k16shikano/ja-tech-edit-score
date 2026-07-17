# DOK で cross-encoder 報酬モデル（段階2b）

[ROADMAP.md](ROADMAP.md) 段階2b の実行手順。
`(source, candidate)` を連結して 1 スカラーを出す cross-encoder を、推敲ペアで fine-tune する。
凍結埋め込み＋線形ヘッド（pref-bt、LOPO micro 0.975）に **LOPO で勝てるか** が採否の判定。

## 前提

- `data/pref_dataset.jsonl` と `data/pref_split/` が手元にある
  - hunk のみ: `make train` のデータ生成部
  - **節ペア込み**: `make section-pref-data`（hunk を `pref_dataset.hunk.jsonl` に退避してマージ）
- さくら高火力 DOK と非公開レジストリ（原稿由来データをイメージに同梱するため公開レジストリ不可）

## 手順1: build & push

```bash
# 節ペア込みデータを作る（初回または再採掘後）
make section-pref-data

# ローカル検証
make build-pref-ce-image          # → pref-ce:local

# DOK 用 push
export REGISTRY=（名前）.sakuracr.jp
./scripts/build_push_pref_ce_image.sh
```

イメージ例: `…/pref-ce:latest`
同梱: `pref_dataset.jsonl`、`pref_split/train.jsonl`、`pref_split/valid.jsonl`（非公開原稿由来）。

## 手順2: DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（名前）.sakuracr.jp/pref-ce:latest` |
| GPU | V100 で可（130m。310m なら H100 か `GRADIENT_CHECKPOINTING=1`） |
| コマンド | **空** |

| 環境変数 | スモーク | 本番 | 長文化実験 |
|----------|----------|------|------------|
| `MODE` | `xproject` | `xproject` → 勝ったら `train` | `train` |
| `ONLY_PROJECTS` | `ir-system`（1 fold だけ） | 未設定（全 fold） | 未設定 |
| `BASE_MODEL` | 未設定（`sbintuitions/modernbert-ja-130m`） | 同じ。比較で `310m` も | 同じ |
| `MAX_LENGTH` | 未設定（512） | 未設定（512） | **`2048`**（まずここ） |
| `BATCH_SIZE` | 未設定（16） | 未設定 | **`4`**（V100 なら。OOM なら `2`） |
| `GRADIENT_CHECKPOINTING` | 未設定 | 未設定 | **`1`**（長文時推奨） |
| `EPOCHS` / `LR` | 未設定（2 / 3e-5） | 必要なら調整 | 同じ |

### 長文化実験（2026-07-17）

モデル自体は `max_position_embeddings=8192`。いままでの 512 は学習・採点の設定上限。
学習ペアの大半は 512 以内（p95≈405）だが、held-out 節ペアは p50≈1206・p90≈1945 で、
512 では 14%、**2048 なら 92%** が切らずに入る。構成試験（v2）の切り詰めを外すのが主目的。

```bash
# イメージは既存の ja-tech-edit.sakuracr.jp/pref-ce:latest で可（再 build 不要）
# DOK 環境変数:
MODE=train
MAX_LENGTH=2048
BATCH_SIZE=4
GRADIENT_CHECKPOINTING=1
```

成果物: `outputs/pref-ce-ml2048/`（in-domain valid pair accuracy 1.0）。

Hard Eval 比較（2026-07-18）:

| 試験 | モデル | Top-1 | ペア | human>deg-reverse | human>fable |
|------|--------|------:|-----:|------------------:|------------:|
| v2 短文（512予算） | beyond-para | 0.00 | 0.450 | 4/24 | — |
| v2 短文 | **ml2048** | 0.00 | **0.546** | **10/24** | — |
| v2 長文（2048予算, 14/24が>512） | beyond-para（切る） | 0.00 | 0.408 | 2/24 | — |
| v2 長文 | **ml2048** | 0.00 | **0.529** | **8/24** | — |
| v2 長文 | BT | 0.00 | 0.446 | 3/24 | — |
| v2b | beyond-para | 0.29 | 0.736 | — | 7/24 |
| v2b | **ml2048** | **0.42** | **0.792** | — | **10/24** |
| v2b | BT | 0.67 | 0.833 | — | 16/24 |

長文化で構成・微差とも改善するが、Top-1=0（構成改悪を human より上に置きがち）と
fable 過大評価は残る。切り詰め除去だけでは足りない。

- `MODE=xproject`: fold ごとにベースから学習し直す LOPO。成果物は `eval_ce_xproject.json`
- `MODE=train`: `pref_split` の train/valid で 1 本学習。成果物は `pref-ce/`（HF モデル一式 + metrics）

fold は 13〜14 個あるので、LOPO 全体は「1 fold の学習時間 × fold 数」かかる。
先に `ONLY_PROJECTS` で 1 fold 回し、学習時間と精度の見当を付けてから全 fold にする。

## 手順3: 判定

`eval_ce_xproject.json` の `micro_pair_accuracy` を pref-bt（0.975）と比べる。

- 明確に上回る（目安 +1pt 以上）→ `MODE=train` で 1 本作り、`outputs/pref-ce/` に置く
- 同等以下 → 凍結埋め込み＋線形ヘッドで十分という知見。段階2b は閉じる

**選抜への採用**は LOPO だけでは決めない。LLM ベース＋人手の難試験（[HARD-EVAL.md](HARD-EVAL.md)）で pref-bt と比較する。

ローカルで動かす場合（GPU があれば）:

```bash
make eval-ce-xproject ONLY_PROJECTS=ir-system   # スモーク
make eval-ce-xproject                            # 全 fold
make train-ce                                    # 1 本学習 → outputs/pref-ce
```

## 関連ファイル

| パス | 役割 |
|------|------|
| `scripts/train_pref_ce.py` | 学習（BT 損失、単発 train/valid） |
| `scripts/eval_pref_ce_xproject.py` | LOPO 評価（fold ごとに学習し直し） |
| `scripts/pref_ce_runtime.py` | 読み込み・採点（採用時に rank から使う） |
| `scripts/dok_pref_ce.sh` | DOK 起動処理 |
| `Dockerfile.pref-ce` | 箱 |
| `scripts/build_section_pref_pipeline.sh` | 節ペア → pref 化 → hunk とマージ → split |
| `scripts/build_pref_ce_image.sh` | ローカル build または REGISTRY 指定で push |
| `scripts/build_push_pref_ce_image.sh` | `build_pref_ce_image.sh` へのエントリ（後方互換） |

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| `docker login` / push 失敗 | レジストリ認証。非公開レジストリのみ push |
| tokenizer で `sentencepiece` / `protobuf` エラー | イメージを再ビルド（`requirements-pref-ce.txt` に同梱済み） |
