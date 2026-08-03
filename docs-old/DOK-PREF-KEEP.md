# レビュー済み keep で評価器を学習する（DOK / GPU）

旧 `pref_dataset` / `pref_split` と旧成果物（`outputs/pref-bt`、`outputs/pref-sentseq-*`）は使わない。
**単位を混ぜない。**

| 成果物 | 学習データ |
|--------|------------|
| `outputs/pref-bt-keep` | 空行なし hunk（`data/pref_keep_split_hunk/`） |
| `outputs/pref-sentseq-keep` | 節・空行あり（`data/pref_keep_split_section/`） |

## 履歴: 単位混在の一回目

節と hunk を混ぜて学習した成果物は、末尾 `-mixed` に退避した。

| パス | 内容 |
|------|------|
| `outputs/pref-bt-keep-mixed` | keep 全単位（節+hunk）で学習した BT |
| `outputs/pref-sentseq-keep-mixed` | 同上の文列 |

当時の採点レポート（LoRA 再採点・難試験）も、モデルパスが混在版を指している。

- `outputs/edit-sft-eval/score_report_*_keep.*`
- `outputs/hard_eval_v*_report_*_keep.*`

現行の本線は分離学習（上表）である。混在版は比較用の履歴であり、採否の既定には使わない。

関連: [HARD-EVAL.md](HARD-EVAL.md)、[ROADMAP.md](ROADMAP.md)、`.cursor/rules/no-old-scorers.mdc`。

## 手順1: データ（手元）

```bash
make pref-keep-data
```

hunk 分割と節分割の両方を書く。

## 手順2: build & push

```bash
export REGISTRY=（コンテナレジストリ名）.sakuracr.jp
chmod +x scripts/build_push_pref_keep_image.sh
./scripts/build_push_pref_keep_image.sh
```

イメージ例: `…/pref-keep:latest`

## 手順3: DOK タスク

| 項目 | 入れるもの |
|------|------------|
| イメージ | `（コンテナレジストリ名）.sakuracr.jp/pref-keep:latest` |
| GPU | V100 以上 |
| コマンド | **空** |

| 環境変数 | スモーク | 本番 |
|----------|----------|------|
| `MODE` | `both` | `both` |
| `DEVICE` | 未設定（`cuda`） | 同じ |
| `BT_EPOCHS` | `5` | 未設定（`80`） |
| `SENTSEQ_EPOCHS` | `2` | 未設定（`40`） |
| `SENTSEQ_LR` | 未設定（`3e-4`） | 同じ |

`MODE=bt` / `MODE=sentseq` で片方だけも可。

## 手順4: 成果物を手元へ

```bash
mkdir -p outputs/pref-bt-keep outputs/pref-sentseq-keep
# アーティファクトの pref-bt-keep/ と pref-sentseq-keep/ をここへ
```

`DATA_UNIT.txt` に `unit=hunk` または `unit=section` が入る。混在版（`-mixed`）とは別名なので上書きしない。

## うまくいかないとき

| 症状 | 見ること |
|------|----------|
| COPY 失敗 | `make pref-keep-data` 後にイメージを作り直す |
| OOM（sentseq） | `SENTSEQ_BATCH_SIZE=32` |
| 件数が想定外 | `data/pref_keep/build_report_{hunk,section}.json` |

## 関連パス

| パス | 役割 |
|------|------|
| `scripts/build_pref_from_keep.py` | keep → 選好分割 |
| `scripts/dok_pref_keep.sh` | DOK 入口 |
| `Dockerfile.pref-keep` | 学習箱 |
