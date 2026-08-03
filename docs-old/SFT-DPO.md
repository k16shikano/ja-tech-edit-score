# SFT / DPO 構想メモ（旧）

将来、生成モデル向けの **SFT** / **DPO** を検討するときの初期メモ。
**実験計画と理論の正式版は [EDIT-MODEL.md](EDIT-MODEL.md) に移した。**
本ファイルはデータ形式とパイプライン共有の説明を残す。

活性化ステアリング（系統3）は [ACTIVATION-STEERING.md](ACTIVATION-STEERING.md)。
全体の優先順位は [ROADMAP.md](ROADMAP.md)。

## 現状（本リポジトリ）

`make train` のパイプラインには `dpo_dataset.jsonl` という名前の中間ファイルがある。

```
make data → import → build_dpo_dataset → curate → build_pref_dataset → train_pref_static
```

この **DPO 形式の JSONL** は、**pref-static（選好評価モデル）を学習するための中間データ** である。
生成 LLM の fine-tune にはまだ使っていない。

各レコードはおおよそ次の形を持つ（`scripts/build_dpo_dataset.py` 参照）。

- `prompt`：推敲依頼の指示
- `input`：修正前テキスト（必要なら参照行付き）
- `chosen`：採用された推敲後テキスト
- `rejected`：非採用側（既定は修正前と同一）

pref-static 側では、この chosen / rejected を **ペア選好の教師** に変換し、埋め込みと分類器で学習する。

編集モデル用・steering 用の清潔な対照ペアは:

```bash
make steering-pairs
# → data/revision_pairs.jsonl （draft / revised）
```

## 構想の要約

詳細・フェーズ分け・報酬ハッキング対策は [EDIT-MODEL.md](EDIT-MODEL.md) を読む。

- ゼロ生成ではなく **編集モデル**（\(y \mid x\)）として SFT する
- 静的 DPO（rejected=下書き）だけでは弱く、BT 報酬による **on-policy** 反復へ進む
- kNN 実例注入は不採用（過去に失敗）

## データパイプラインの共有

採掘（`make data`）は pref-static / 編集モデル / steering で **共通** である。

```
Git diff（下書き vs 推敲）
        ↓
examples.raw.jsonl（ローカル）
        ↓
dpo_curated.jsonl
        ├── pref-static / pref-bt（現行）
        ├── revision_pairs.jsonl → steering フェーズ A（脚本あり）
        └── 編集モデル SFT / DPO（未実装）
```

公開リポジトリに含めるのは **スクリプト、スキーマ、同梱 pref-static モデル** のみ。
原稿ペア本体は引き続きローカルの `data/` に置く。

## 関連ファイル（現行）

| パス | 内容 |
|------|------|
| `scripts/build_dpo_dataset.py` | chosen / rejected JSONL の生成 |
| `scripts/curate_dpo_dataset.py` | 引用のみ差分などの除外 |
| `scripts/build_pref_dataset.py` | pref 学習用への変換（swap 拡張あり） |
| `scripts/export_revision_pairs.py` | draft / revised 対照ペア |
| `data/examples.schema.json` | 採掘ペアのスキーマ |
| [EDIT-MODEL.md](EDIT-MODEL.md) | 系統1の理論と実験計画 |
| [ACTIVATION-STEERING.md](ACTIVATION-STEERING.md) | 系統3 |
