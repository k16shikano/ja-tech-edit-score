# SFT / DPO 構想メモ

将来、生成モデル向けの **SFT**（教師あり fine-tune）や **DPO**（Direct Preference Optimization）を検討するときのメモ。
現時点では **未実装** である。

[WORKFLOW.md](WORKFLOW.md) の選好評価モデル（pref-static）と役割が異なる。
本メモは実装計画ではなく、方向性の整理用である。

## 現状（本リポジトリ）

`make train` のパイプラインには `dpo_dataset.jsonl` という名前の中間ファイルがある。

```
make data → import → build_dpo_dataset → curate → build_pref_dataset → train_pref_static
```

この **DPO 形式の JSONL** は、**pref-static（選好評価モデル）を学習するための中間データ** である。
生成 LLM の fine-tune には使っていない。

各レコードはおおよそ次の形を持つ（`scripts/build_dpo_dataset.py` 参照）。

- `prompt`：推敲依頼の指示
- `input`：修正前テキスト（必要なら参照行付き）
- `chosen`：採用された推敲後テキスト
- `rejected`：非採用側（既定は修正前と同一）

pref-static 側では、この chosen / rejected を **ペア選好の教師** に変換し、埋め込みと分類器で学習する。

## 構想：生成モデル向け SFT

**目的**：推敲案そのものを生成するモデルを、採掘済みペアから学習する。

| 項目 | 内容 |
|------|------|
| 入力 | 修正前段落、章コンテキスト、推敲規範（プロンプト） |
| 教師 | Git ブランチ diff から採掘した `edited_text` |
| データ源 | 既存の `data/examples.raw.jsonl`（ローカル、非公開） |
| 位置づけ | pref-static の **上流**。候補を出す側 |

検討事項（未決定）：

- ベースモデル（日本語技術文向けの既存 instruct モデル等）
- 1 段落単位か、節単位か
- Pandoc 記法や sec ラベルを壊さない制約を学習にどう入れるか
- 学習データの `project_id` やファイルパスを公開リポジトリに含めない運用

## 構想：生成モデル向け DPO

**目的**：SFT 後（またはベースモデル上）で、「採用推敲」と「非採用」を直接比較して選好を強化する。

| 項目 | 内容 |
|------|------|
| chosen | 推敲ブランチ側（人間が採用した `edited_text`） |
| rejected | 下書き側（`source_text`）、または弱い自動候補 |
| データ源 | 上記と同じ採掘パイプライン。`build_dpo_dataset.py` の出力形式を流用できる可能性 |
| 位置づけ | SFT の **補強**。文体の「好ましい方向」を直接最適化 |

pref-static との関係：

| レイヤ | 役割 |
|--------|------|
| pref-static（現行） | 埋め込みと分類器で **採点・2 候補の選択**。軽量、同梱モデルで推論可能 |
| DPO 生成モデル（構想） | **推敲文を生成**。計算コストと運用負荷は大きい |

両者は排他ではない。
生成で複数候補を出し、pref-static で選ぶ、という二段構成もあり得る。

## データパイプラインの共有

採掘（`make data`）は SFT / DPO / pref-static で **共通** にできる想定である。

```
Git diff（下書き vs 推敲）
        ↓
examples.raw.jsonl（ローカル）
        ↓
    ┌───┴───┐
    ↓       ↓
pref-static   生成 SFT / DPO（将来）
（現行）      （未実装）
```

公開リポジトリに含めるのは **スクリプト、スキーマ、同梱 pref-static モデル** のみ。
原稿ペア本体は引き続きローカルの `data/` に置く。

## 実装しない理由（現時点）

- 推敲の主経路は Cursor 等のフロンティアモデルと `japanese-tech-writing` 規範で足りている
- pref-static だけで選好チェックと 2 候補比較が可能
- 生成 fine-tune は GPU、評価、バージョン管理のコストが pref-static より一段大きい

必要になったタイミングで、別リポジトリまたは本リポジトリの `scripts/` 拡張として切り出す。

## 関連ファイル（現行）

| パス | 内容 |
|------|------|
| `scripts/build_dpo_dataset.py` | chosen / rejected JSONL の生成 |
| `scripts/curate_dpo_dataset.py` | 引用のみ差分などの除外 |
| `scripts/build_pref_dataset.py` | pref 学習用への変換（swap 拡張あり） |
| `data/examples.schema.json` | 採掘ペアのスキーマ |
