# DOK で cross-encoder 報酬モデル（段階2b）

[ROADMAP.md](ROADMAP.md) 段階2b の実行手順。
`(source, candidate)` を連結して 1 スカラーを出す cross-encoder を、推敲ペアで fine-tune する。
凍結埋め込み＋線形ヘッド（pref-bt、LOPO micro 0.975）に **LOPO で勝てるか** が採否の判定。

## 前提

- `data/pref_dataset.jsonl` と `data/pref_split/` が手元にある（`make train` のデータ生成部）
- さくら高火力 DOK と非公開レジストリ（原稿由来データをイメージに同梱するため公開レジストリ不可）

## 手順1: build & push

```bash
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

| 環境変数 | スモーク | 本番 |
|----------|----------|------|
| `MODE` | `xproject` | `xproject` → 勝ったら `train` |
| `ONLY_PROJECTS` | `ir-system`（1 fold だけ） | 未設定（全 fold） |
| `BASE_MODEL` | 未設定（`sbintuitions/modernbert-ja-130m`） | 同じ。比較で `310m` も |
| `EPOCHS` / `LR` / `BATCH_SIZE` | 未設定（2 / 3e-5 / 16） | 必要なら調整 |

- `MODE=xproject`: fold ごとにベースから学習し直す LOPO。成果物は `eval_ce_xproject.json`
- `MODE=train`: `pref_split` の train/valid で 1 本学習。成果物は `pref-ce/`（HF モデル一式 + metrics）

fold は 13〜14 個あるので、LOPO 全体は「1 fold の学習時間 × fold 数」かかる。
先に `ONLY_PROJECTS` で 1 fold 回し、学習時間と精度の見当を付けてから全 fold にする。

## 手順3: 判定

`eval_ce_xproject.json` の `micro_pair_accuracy` を pref-bt（0.975）と比べる。

- 明確に上回る（目安 +1pt 以上）→ `MODE=train` で 1 本作り、`outputs/pref-ce/` に置いて `rank` / `converge` の CE 対応を進める
- 同等以下 → 凍結埋め込み＋線形ヘッドで十分という知見。段階2b は閉じる

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
| `scripts/build_push_pref_ce_image.sh` | build & push |

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| `docker login` / push 失敗 | レジストリ認証。非公開レジストリのみ push |
| tokenizer で `sentencepiece` / `protobuf` エラー | イメージを再ビルド（`requirements-pref-ce.txt` に同梱済み） |
