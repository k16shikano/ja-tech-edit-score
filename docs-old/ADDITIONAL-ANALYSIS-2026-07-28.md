# 追加分析報告（Hard Eval 統計・長さ交絡・プローブ全層・pref-bt LOPO 6274）

作成日: 2026-07-28

## 0. 作業開始時の棚卸し

### 0.1 データセット版と件数

| 識別 | 件数 | 意味 | 主なパス |
|------|-----:|------|----------|
| hunk curated | 6055 | hunk 由来のキュレーション選好 / revision 対照 | `data/dpo_curated.jsonl`, `data/revision_pairs.jsonl` |
| unique pref（節込み） | **6274** | swap 除外のユニーク選好（hunk+section） | `unique_preference_pairs(data/pref_dataset.jsonl)` |
| pref_dataset 行 | 12548 | 6274 の swap 増強 | `data/pref_dataset.jsonl` |
| pref_split | 9984 / 1302 / 1262 | train/valid/test（swap 込み行） | `data/pref_split/` |
| unique valid | 651 | valid のユニーク | DOK-PREF-SENTSEQ の較正などで使用 |
| 旧 bt LOPO | **5887** | 節ペア追加前の LOPO 総ペア | `outputs/eval_bt_xproject.json` |

6055 と 6274 を混同しない。旧 bt LOPO（5887）と sentseq LOPO（6274）はデータ版が異なる。

### 0.2 モデル

| 名前 | 設定識別 | 成果物 | 備考 |
|------|----------|--------|------|
| pref-bt | default | `outputs/pref-bt` | 凍結 ruri + 線形 |
| pref-ce | default / beyond-para / … | `outputs/pref-ce*` | cross-encoder |
| pref-sentseq | lr1e-4-ep20 | `outputs/pref-sentseq` | metrics に lr 未記録、docs/Makefile 上 1e-4・20ep |
| pref-sentseq | lr1e-4-ep40 | `outputs/pref-sentseq-40` | |
| pref-sentseq | **lr3e-4-ep40（採用）** | `outputs/pref-sentseq-3e4` | |
| pref-sentseq | d512-l4-lr1e-4-ep40 | `outputs/pref-sentseq-512` | 容量増は不採用 |

### 0.3 Hard evaluation

| 集合 | 項目数 | 候補 | 参照順位の性質 | 入力 |
|------|-------:|------|----------------|------|
| v1 | 20 | 6（human, copy, composer, fable, gpt-5.6, grok） | 構築時に human を最上位に固定。独立盲検順位ではない | `bases_v1_labeled.jsonl` |
| v2b | 24 | 3（human, fable, copy） | 構築時参照 `human>fable>copy` | `bases_v2b_human_fable_copy.jsonl` |
| v2c | 24 | 3（human, machine=composer-2.5, copy） | 構築時参照 `human>machine>copy` | `bases_v2c_human_machine_copy.jsonl` |

候補生成: v1 は composer/fable/gpt-5.6/grok、v2b は Fable、v2c は composer-2.5。

既存スコアレポートは `outputs/hard_eval_*_report_*.json`（`summary` + `items[].scores`）。項目別正誤はレポートから再構成可能。

### 0.4 プローブ（Qwen3-8B）

| 成果物 | 内容 |
|--------|------|
| `outputs/steering/Qwen__Qwen3-8B/activations.npz` | 6055×37×4096、mean pooling |
| `probe_report.json` | 単独文章プローブ全 37 層 |
| `probe_paired_diff.json` | 原文・修正差分符号当て全 37 層 |
| `...--reading/` / `...--norms/` | **n=64 スモークのみ**（全件ではない） |

### 0.5 追加分析の実行可否（開始時点）

| 分析 | 可否 | 根拠 |
|------|------|------|
| 項目別正誤行列 | 可 | 既存 hard_eval レポート |
| McNemar / 対応付き bootstrap | 可（新規スクリプト） | 項目単位 top1 |
| 268/300 再確認 | 可 | v1 sentseq40 / sentseq3e4 レポートの scores |
| 候補長メタ・長さ交絡 | 可 | labeled jsonl の len + レポート scores |
| プローブ全層 CSV/図 | 可（再集計） | 既存 JSON。再抽出不要 |
| pref-bt LOPO 6274 | 要再実行 | 既存は 5887。CPU で `eval_pref_bt_xproject.py` |

埋め込みのディスクキャッシュは無い。BT LOPO は実行ごとに ruri で再エンコードする。

---

## 1. 実行した分析（新規実行 vs 再集計）

| 作業 | 区分 |
|------|------|
| Hard Eval 項目正誤・候補メタ CSV | **既存出力の再集計** |
| McNemar / 対応付き bootstrap | **新規計算**（入力は既存レポート） |
| v1 268/300 再確認 | **既存スコアからの再計算** |
| 長さ交絡 | **新規計算**（既存スコア+本文長） |
| プローブ全層曲線 | **既存出力の再集計**（抽出は未再実行） |
| pref-bt LOPO 6274 | **新規実験実行**（同一 `pref_dataset` ユニーク 6274） |

---

## 2. 公開成果物

```text
results/
  hard_eval_item_correctness.csv
  hard_eval_candidate_metadata.csv
  mcnemar_results.csv
  paired_bootstrap_results.csv
  length_control_results.csv
  probe_layer_results.csv
  pref_bt_lopo_6274.csv          # LOPO 完了後に export
  figures/
    probe_layer_accuracy.pdf|.svg
    length_score_relationship.pdf|.svg
  raw/                           # 中間・詳細（本文なし）
```

候補本文・書籍本文・埋め込み・活性値は含めない。`candidate_id` は SHA-256 短縮ハッシュ。`candidate_role` は評価上の役割名（human/copy/…）で、本文復元には使えない。

### 文字数定義

- Python `len(text)`（Unicode コードポイント数）
- 改行・Markdown 記号を含む
- 前処理なし（`score_hard_eval.py` の length Spearman と同じ）

---

## 3. 再計算した主要数値

### 3.1 v1 pairwise 268/300（lr1e-4-ep40 vs lr3e-4-ep40）

ソース: `outputs/hard_eval_v1_report_sentseq40.json` と `...sentseq3e4.json` の候補スコアから再計算（`scripts/recheck_v1_sentseq_pairwise.py`）。

| | lr1e-4-ep40 | lr3e-4-ep40 |
|--|------------:|------------:|
| top1 | 17/20 | 16/20 |
| pairwise | **268/300** | **268/300** |
| ペア判定の不一致 | 8 組（同一 268 だが判定内容は異なる） |

- 両方 top1 正解: 16 項目
- a のみ top1: `he-v1-13`
- b のみ top1: なし
- **完全に同じ pairwise 判定ではない**（`identical_pairwise_judgments=false`）

ドキュメントの「双方 0.893（268/300）」は件数として正しい。判定の同一性までは成り立たない。

### 3.2 Wilson 95% CI（top1、主要モデル）

| 集合 | モデル | 正解 | 率 | Wilson 95% |
|------|--------|-----:|---:|------------|
| v1 | sentseq lr1e-4-ep40 | 17/20 | 0.850 | [0.640, 0.948] |
| v1 | sentseq lr3e-4-ep40 | 16/20 | 0.800 | [0.584, 0.919] |
| v1 | pref-bt | 9/20 | 0.450 | [0.258, 0.658] |
| v2b | sentseq lr3e-4-ep40 | 17/24 | 0.708 | [0.508, 0.851] |
| v2b | pref-bt | 16/24 | 0.667 | [0.467, 0.820] |
| v2c | sentseq lr3e-4-ep40 | 17/24 | 0.708 | [0.508, 0.851] |
| v2c | pref-bt | 14/24 | 0.583 | [0.388, 0.755] |

項目数 20〜24。広い区間である。

### 3.3 McNemar（exact、両側）と不一致数

代表例（詳細は `results/mcnemar_results.csv`）:

| 比較 | n | a_only | b_only | p_exact | acc 差 |
|------|--:|-------:|-------:|--------:|-------:|
| v1 sentseq-3e4 vs bt | 20 | 8 | 1 | 0.039 | +0.35 |
| v1 sentseq-3e4 vs ce | 20 | 7 | 1 | 0.070 | +0.30 |
| v1 sentseq-40 vs sentseq-3e4 | 20 | 1 | 0 | 1.0 | +0.05 |
| v2b sentseq-3e4 vs bt | 24 | 2 | 1 | 1.0 | +0.042 |
| v2c sentseq-3e4 vs bt | 24 | 4 | 1 | 0.375 | +0.125 |

多重比較補正は**主結果として適用していない**（探索的）。p 値だけで優劣を断定しない。不一致数が小さい。

### 3.4 対応付き bootstrap（項目単位、B=100000, seed=0）

`results/paired_bootstrap_results.csv`。pairwise 率差は項目を cluster として再標本化。候補対を独立二項とはしていない。

### 3.5 長さ交絡（探索的）

学習データ（unique train 4992）: 編集が長い 1669 / 短い 3089 / 同長 234 → 多数派向きは **shorter**。

| 集合 | モデル | orig top1 | length-only longer | shorter | 項目内中心化残差 top1 | 項目内 ρ 中央値 | 全候補まとめ ρ |
|------|--------|----------:|-------------------:|--------:|----------------------:|----------------:|---------------:|
| v1 | sentseq-3e4 | 16/20 | **17/20** | 1/20 | 5/20 | 0.94 | 0.90 |
| v1 | sentseq-40 | 17/20 | 17/20 | 1/20 | 8/20 | 0.94 | 0.88 |
| v2b | sentseq-3e4 | 17/24 | 18/24 | 3/24 | 7/24 | 0.50 | 0.73 |
| v2c | sentseq-3e4 | 17/24 | 16/24 | 3/24 | 6/24 | 0.50 | 0.77 |

残差化は**同一評価集合で係数推定**した探索的分析。0.05 程度の差を一般化しない。

**支持されること:** v1 では「長い候補を選ぶ」だけの基準が sentseq と同程度の top1 に達し、sentseq 得点と長さの項目内相関が高い。長さ交絡を無視して段落構成理解だけに帰属できない。

**支持されないこと:** 「長さだけが成績の原因である」ことの証明にはならない（残差後も一部ヒットが残るが、n が小さい）。

長さだけの基準は**定量的**に示せる。一方、参照順位が非盲検・構築時設定であることは**設計上の問題**であり、同じ「定量的に示した」とまとめてはいけない。

### 3.6 プローブ全層

- 単独文章: 最良層 33、micro ≈ 0.542（事後選択）
- ペア差分: 最良は `probe_paired_diff.json` の best（事後選択）
- pooling: **mean のみ**（全 6055）。「mean が最良」とは言えない
- reading/norms は n=64 のみで主表に未統合
- 単一モデル（Qwen3-8B）のみ
- 低精度を「内部に推敲方針が存在しない」証明には使わない

---

## 4. pref-bt LOPO 6274

実行コマンド:

```bash
python scripts/eval_pref_bt_xproject.py \
  --input data/pref_dataset.jsonl \
  --report results/raw/eval_bt_xproject_6274.json \
  --epochs 80 --lr 1e-2
```

seed は脚本内で fold ごとに `seed=0`。埋め込みは CPU・ディスクキャッシュなし（ユニーク文のエンコードに約 8 分）。

| 指標 | pref-bt（本次 6274） | pref-sentseq（既存 LOPO 6274） |
|------|---------------------:|-------------------------------:|
| micro pair accuracy | **0.973** | 0.905 |
| macro pair accuracy | **0.960** | 0.910 |
| folds / pairs | 14 / 6274 | 14 / 6274 |

同一データ版・同一 project fold。旧 `outputs/eval_bt_xproject.json`（5887）とは比較しない。

project 別は `results/pref_bt_lopo_6274.csv` と `results/raw/lopo_6274_bt_vs_sentseq.json`。project 単位 bootstrap による macro の 95% CI も同 JSON に記載（再標本化単位は project）。

**解釈の注意:** LOPO の「下書き vs 人間編集」ペア正解率で bt が高いことは、Hard Eval（人間 vs 機械案）での sentseq の相対的強さとは別課題である。両者を一つの能力指標に潰さない。

---

## 5. 再現用スクリプト

| スクリプト | 役割 |
|------------|------|
| `scripts/export_hard_eval_metrics.py` | 項目正誤・候補メタ |
| `scripts/run_paired_statistics.py` | McNemar・bootstrap・Wilson |
| `scripts/recheck_v1_sentseq_pairwise.py` | 268/300 再確認 |
| `scripts/analyze_length_confound.py` | 長さ交絡 |
| `scripts/export_probe_layer_curve.py` | プローブ全層 CSV/図 |
| `scripts/eval_pref_bt_xproject.py` | BT LOPO（既存） |
| `scripts/export_pref_bt_lopo_csv.py` | LOPO CSV 整形 |

いずれも `--overwrite` なしでは既存出力を上書きしない。入力欠損時は停止する。

---

## 6. 再現できなかった / 欠けているもの

| 項目 | 状態 |
|------|------|
| pref-ce の v2c レポート | **欠落**（v2c の CE 採点ファイルなし） |
| pref-sentseq ベースの metrics に lr | 未記録（設定は docs/Makefile からの同定） |
| Qwen revision ハッシュ | meta に無し（空欄） |
| reading/norms 全 6055 プローブ | 未実施（要 GPU） |
| 端から端の推敲ループ実証 | 未実施（本報告の対象外・未実証） |
| 埋め込みディスクキャッシュ | 存在しない |

---

## 7. 結果から支持される主張 / されない主張

### 支持される

- Hard Eval は項目数が少なく、top1 差の不確かさは大きい（Wilson 区間が広い）
- v1 で sentseq（3e-4）と bt の top1 差は不一致 8 vs 1、exact p≈0.039だが、n=20・探索的比較である
- v1 の sentseq は候補長と強く連動し、length-only（longer）が同程度の top1 に達する → **長さ交絡を無視できない**
- 268/300 は両設定で再確認されたが、ペア判定は同一ではない
- プローブは全層を探索し最良を事後選択している；平均プーリング以外の全件比較は無い

### 支持されない

- 生成 LLM が人間編集者の推敲方針を獲得した、という主張
- pref-sentseq の成績＝段落構成理解の証拠
- プローブ低精度＝内部表現に推敲方針が無いことの証明
- モデル間の「同等」（有意差なし≠同等）
- 未実施のループエンジニアリング実証

### 目的との対応

- **ツール:** 外部評価器をループに組み込む可能性は、Hard Eval・二軸運用・公開採点サービスまでの工学的検討として残るが、端から端の実証はしていない
- **研究:** 測定上の落とし穴として、(1) 構築時参照順位の非盲検性、(2) 候補長交絡、(3) 小標本での過解釈、(4) プローブ最良層の事後選択バイアス、をデータに即して示した

---

## 8. 今後必要な GPU 実験（実施していない）

- reading / norms プロンプトでの全 6055 活性抽出と全層プローブ
- （任意）別モデルでのプローブ再現

---

## 9. 参照順位についての明記

- **v1:** 人間候補が機械的に参照最上位として置かれている。独立人手の盲検順位ではない
- **v2b/v2c:** 順位は評価集合構築時の参照順位であり、独立した人手品質評価ではない
