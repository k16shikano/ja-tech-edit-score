# ブラインド判定 集計結果

## 1. 概要

本書は、工程 6 の人手ブラインド判定 360 件を `scripts/analyze_blind_judgments.py` で集計した結果である。
判定の生データは `data/blind_eval/judgments.jsonl`、ペア定義は `data/blind_eval/pairs.jsonl` にある。
数値の詳細は `data/blind_eval/analysis.json` に同一内容を JSON で保存している。

- 判定対象節数：60（各節 6 比較、計 360 判定）
- 生成物の出所：`outputs/edit-sft-eval-v3/`（工程 C の本生成）
- 評価器：`outputs/pref-sentseq-keep-pairsplit`（文列型）、`outputs/pref-bt-keep-pairsplit`（Bradley-Terry 型）
- 工程 8-mid の文列型：`outputs/pref-sentseq-section-triples`（同じ 360 件を採点し直したもの。§5.5）

再生成：`make analyze-blind-judgments`

## 2. 用語

PLAN.md の用語表に加え、本集計で使う識別子を次に固定する。

**下書き**：判定対象節の推敲前原文。`items.jsonl` の `draft`、画面上の「文脈」、評価器採点時の `source` に使う。

**人間の推敲**：同一節のレビュー済み正解文。`items.jsonl` の `gold`。

**素のモデル**：LoRA を載せない Qwen3-8B。PLAN の「素のモデル」と同じ。

**推敲モデル**：SFT で得た LoRA を載せた Qwen3-8B。PLAN の「推敲モデル」と同じ。

**貪欲生成**：各ステップで最確語のみを選ぶ生成。同一入力なら出力は 1 本に定まる（PLAN 用語表）。

**サンプリング生成**：温度 0.7・top_p 0.9・シード固定で語をサンプルする生成。同一入力から複数案を出す。

**素のモデル・貪欲生成**（データ上の識別子 `base_greedy`）：素のモデルが指示文付きで貪欲生成した 1 文。
ファイル `outputs/edit-sft-eval-v3/base_greedy.jsonl` の `generated`。

**推敲モデル・貪欲生成**（`adapter_greedy`）：推敲モデルが同一指示文で貪欲生成した 1 文。
ファイル `outputs/edit-sft-eval-v3/adapter_greedy.jsonl` の `generated`。

**推敲モデル・選抜1本**（`adapter_selected`）：推敲モデルがサンプリング 8 本を出し、
見出し・【図】【コード】の本数を保った案だけを残したうえで評価器（文列型）が最高点を採った 1 文。
健全なサンプルが 0 本のときは `adapter_greedy` にフォールバックする（`scripts/build_blind_pairs.py`）。

**相対選択**：判定者が「候補 A のほうがまし」「候補 B のほうがまし」「同等」のいずれかを選んだ結果。
`judgments.jsonl` の `choice`（`a` / `b` / `tie`）。

**致命的欠落・破壊**（`a_broken` / `b_broken`）：見出し・【図】【コード】の欠落、内容の破壊、文体（ですます／である）の変更など、
採用できない欠陥があると判定者が付けたチェック。

**事実上の差分なし**（`a_noedit` / `b_noedit`）：下書きと完全一致でなくても、言い換え程度で編集と呼べないと判定者が付けたチェック。

**失格勝ち**：相対選択で勝った側の相手に致命的欠落・破壊がある勝ち。
**無編集勝ち**：相対選択で勝った側に事実上の差分なしがある勝ち。
**品質勝ち**：上記のどちらでもない勝ち。勝った側が編集として成立しているとみなす。

**破壊率**（本集計）：比較2の 60 節について、`adapter_greedy` に致命的欠落・破壊チェックが付いた割合。

**無編集率**（人手）：比較2の 60 節について、`adapter_greedy` に事実上の差分なしチェックが付いた割合。

**改善率**（本集計）：比較2の 60 節について、`adapter_greedy` が相対選択で勝ち、かつ品質勝ちに分類された割合。

**Wilson 95% CI**：二項比率の 95% 信頼区間。

## 3. 比較の種類

判定対象 60 節それぞれについて、次の 6 ペアを出題した。
左右（候補 A / B）の割付はシード固定乱数で決め、`pairs.jsonl` の `swapped` に記録する。

- **比較1**：下書き vs 素のモデル・貪欲生成
- **比較2**：下書き vs 推敲モデル・貪欲生成
- **比較3**：素のモデル・貪欲生成 vs 推敲モデル・貪欲生成
- **比較4**：下書き vs 推敲モデル・選抜1本
- **比較5**：人間の推敲 vs 推敲モデル・選抜1本
- **比較6**：人間の推敲 vs 下書き

## 4. 推敲モデルに関する集計

対象は主に **推敲モデル・貪欲生成**（`adapter_greedy`）と **推敲モデル・選抜1本**（`adapter_selected`）。

### 4.1 三つの率（`adapter_greedy`、比較2・60 節）

| 指標 | 定義 | 値 | Wilson 95% CI |
|---|---|---:|---:|
| 破壊率 | 致命的欠落・破壊チェック | 2/60 = 0.033 | [0.009, 0.114] |
| 無編集率 | 事実上の差分なしチェック | 6/60 = 0.100 | [0.047, 0.201] |
| 改善率 | 相対選択で勝ちかつ品質勝ち | 35/60 = 0.583 | [0.457, 0.699] |

### 4.2 機械計測（`adapter_greedy`、判定対象 60 節）

下書きと生成文の文字列類似度。成否基準には使わない（PLAN 判定基準）。

- 完全一致：2/60 = 0.033
- 類似度 ≥ 0.95：17/60 = 0.283
- 類似度中央値：0.848

### 4.3 比較3（成否基準）

比較3は **素のモデル・貪欲生成** と **推敲モデル・貪欲生成** の相対選択 60 件。
PLAN の成立基準は推敲モデル側の勝ちが 36/60 以上。

- 推敲モデル・貪欲生成の勝ち：36/60 = 0.600（Wilson [0.474, 0.714]）
- 成立基準 36/60：成立
- 片側二項検定（H1: 勝率 > 0.5）：p = 0.0775

### 4.4 各比較の相対選択勝率

「勝ち」は `choice` が当該候補側を指した件数。分母は各比較 60 件（同点含む）。

| 比較 | 集計対象候補 | 勝ち | 同点 | 勝率 |
|---|---|---:|---:|---:|
| 比較1 | 素のモデル・貪欲生成 | 2/60 | 53 | 0.033 |
| 比較2 | 推敲モデル・貪欲生成 | 36/60 | 8 | 0.600 |
| 比較4 | 推敲モデル・選抜1本 | 36/60 | 0 | 0.600 |
| 比較5 | 推敲モデル・選抜1本 | 11/60 | 0 | 0.183 |
| 比較6 | 人間の推敲 | 52/60 | 1 | 0.867 |

比較3 における文字数比（推敲モデル・貪欲生成 / 素のモデル・貪欲生成）中央値：0.989

### 4.5 勝ちの内訳（相対選択で勝った側の分類）

分母は当該比較で同点を除いた勝敗件数。

- **比較1**：失格勝ち 1/7, 無編集勝ち 3/7, 品質勝ち 3/7
- **比較2**：無編集勝ち 16/52, 品質勝ち 36/52
- **比較3**：失格勝ち 4/54, 無編集勝ち 15/54, 品質勝ち 35/54
- **比較4**：失格勝ち 8/60, 無編集勝ち 16/60, 品質勝ち 36/60
- **比較5**：失格勝ち 7/60, 無編集勝ち 1/60, 品質勝ち 52/60
- **比較6**：失格勝ち 6/59, 無編集勝ち 2/59, 品質勝ち 51/59

### 4.6 候補ラベルごとの絶対評価（出現回数ベース）

同一の生成物が複数比較に登場するため、分母はそのラベルとして画面に示された回数。
例：`adapter_greedy` は比較2・比較3 に各 1 回ずつ出るので分母 120。

| 識別子 | 日本語名 | 出現回数 | 致命的欠落・破壊 | 事実上の差分なし |
|---|---|---:|---:|---:|
| `adapter_greedy` | 推敲モデル・貪欲生成 | 120 | 5 (0.042) | 10 (0.083) |
| `adapter_selected` | 推敲モデル・選抜1本 | 120 | 17 (0.142) | 2 (0.017) |
| `base_greedy` | 素のモデル・貪欲生成 | 120 | 11 (0.092) | 95 (0.792) |
| `draft` | 下書き | 240 | 0 (0.000) | 225 (0.938) |
| `gold` | 人間の推敲 | 120 | 12 (0.100) | 3 (0.025) |

## 5. 評価器に関する集計

### 5.1 測定方法

各判定ペアについて、下書き（`context_draft`）を source、候補 A/B を candidates として評価器が点数を付けた。
点数が高い側を **評価器の選択**（`a` または `b`）とする。
**人手の選択**は `judgments.jsonl` の `choice`（`a` / `b` / `tie`）。

**一致**：比較可能ペアのうち、人手の選択と評価器の選択が同じ（どちらも `a`、またはどちらも `b`）。

**比較可能ペア**：人手が `a` または `b` を選び、かつ評価器が同点でないペア。
候補に致命的欠落・破壊チェック（`a_broken` / `b_broken`）が付いていても、人手が選好を付けていれば比較可能に含める。

**人手同等**（`tie`）：2 候補のどちらも採用したくないと判定者が付けた件数。評価器一致の分母には入れない。

**欠陥報告あり**（参考）：その比較種別 60 件のうち、候補 A または B の少なくとも一方に
致命的欠落・破壊チェックが付いている件数。一致率の計算とは独立。

**スコアと文字数の Spearman 相関**：全候補（720 点）について、評価器点数と UTF-8 文字数の順位相関。

### 5.2 人手の相対選択（比較種別ごと）

評価器とは独立。各比較種別 60 件について、判定者がどう付けたか。

| 比較 | 全件 | 人手同等 | 人手 A/B 選択 | 欠陥報告あり |
|---|---:|---:|---:|---:|
| 比較1 | 60 | 53 (0.883) | 7 (0.117) | 7 (0.117) |
| 比較2 | 60 | 8 (0.133) | 52 (0.867) | 2 (0.033) |
| 比較3 | 60 | 6 (0.100) | 54 (0.900) | 7 (0.117) |
| 比較4 | 60 | 0 (0.000) | 60 (1.000) | 12 (0.200) |
| 比較5 | 60 | 0 (0.000) | 60 (1.000) | 9 (0.150) |
| 比較6 | 60 | 1 (0.017) | 59 (0.983) | 8 (0.133) |

### 5.3 文列型（`pref-sentseq-keep-pairsplit`）

- スコアと文字数の Spearman：0.932

| 比較 | 全件 | 人手同等 | 比較可能 | 評価器一致 | 欠陥報告あり |
|---|---:|---:|---:|---:|---:|
| 比較1 | 60 | 53 | 7 | 2/7 (0.286) | 7 |
| 比較2 | 60 | 8 | 52 | 31/52 (0.596) | 2 |
| 比較3 | 60 | 6 | 54 | 30/54 (0.556) | 7 |
| 比較4 | 60 | 0 | 60 | 36/60 (0.600) | 12 |
| 比較5 | 60 | 0 | 60 | 29/60 (0.483) | 9 |
| 比較6 | 60 | 1 | 59 | 52/59 (0.881) | 8 |

**不一致**：比較可能 292 件のうち **112 件**（`analysis.json` → `scorers.sentseq.disagreements`）。

### 5.4 Bradley-Terry 型（`pref-bt-keep-pairsplit`）

- スコアと文字数の Spearman：-0.131

| 比較 | 全件 | 人手同等 | 比較可能 | 評価器一致 | 欠陥報告あり |
|---|---:|---:|---:|---:|---:|
| 比較1 | 60 | 53 | 7 | 2/7 (0.286) | 7 |
| 比較2 | 60 | 8 | 52 | 36/52 (0.692) | 2 |
| 比較3 | 60 | 6 | 54 | 31/54 (0.574) | 7 |
| 比較4 | 60 | 0 | 60 | 36/60 (0.600) | 12 |
| 比較5 | 60 | 0 | 60 | 29/60 (0.483) | 9 |
| 比較6 | 60 | 1 | 59 | 52/59 (0.881) | 8 |

**不一致**：比較可能 292 件のうち **106 件**（`analysis.json` → `scorers.bt.disagreements`）。

### 5.5 文列型（`pref-sentseq-section-triples`）

工程 8-mid の三つ組みで学び直した文列型で、同じ 360 件を採点し直した結果である。
判定 60 件のうち節 50 件は、この文列型の valid（最良 epoch の選定）に入っている。比較 6（人間の推敲対下書き）は、その valid の対と同じ文である。比較 2・3・5 の候補は教師に無い生成文である。
比較 4・5 の「選抜1本」は、当時の文列型が選んだ文のままである。選抜をやり直した結果ではない。

- スコアと文字数の Spearman：0.940

| 比較 | 全件 | 人手同等 | 比較可能 | 評価器一致 | 欠陥報告あり |
|---|---:|---:|---:|---:|---:|
| 比較1 | 60 | 53 | 7 | 2/7 (0.286) | 7 |
| 比較2 | 60 | 8 | 52 | 36/52 (0.692) | 2 |
| 比較3 | 60 | 6 | 54 | 31/54 (0.574) | 7 |
| 比較4 | 60 | 0 | 60 | 36/60 (0.600) | 12 |
| 比較5 | 60 | 0 | 60 | 32/60 (0.533) | 9 |
| 比較6 | 60 | 1 | 59 | 52/59 (0.881) | 8 |

**不一致**：比較可能 292 件のうち **103 件**（`analysis.json` → `scorers.sentseq_section_triples.disagreements`）。

## 6. 入力ファイル

- **ペア定義**（`pairs`）：/home/k16/dev/ja-tech-edit-score/data/blind_eval/pairs.jsonl
- **人手判定**（`judgments`）：/home/k16/dev/ja-tech-edit-score/data/blind_eval/judgments.jsonl
- **判定対象 60 節**（`items`）：/home/k16/dev/ja-tech-edit-score/data/blind_eval/items.jsonl
- **推敲モデル・貪欲生成 jsonl**（`adapter_greedy`）：outputs/edit-sft-eval-v3/adapter_greedy.jsonl
- **文列型評価器**（`sentseq_model`）：/home/k16/dev/ja-tech-edit-score/outputs/pref-sentseq-keep-pairsplit
- **BT 型評価器**（`bt_model`）：/home/k16/dev/ja-tech-edit-score/outputs/pref-bt-keep-pairsplit
- **工程 8-mid の文列型**（`sentseq_section_triples_model`）：outputs/pref-sentseq-section-triples
- **判定件数**（`n_pairs`）：360
- **判定対象節数**（`n_items`）：60

## 7. 難試験（v2b / v2c）

工程 6 の 360 件は、判定 50 節が文列型の valid と重なる。人間の推敲対下書きの一致だけでは、学び直しの根拠にならない。そこで、別の節 24 件について、人間の推敲・別の推敲案・下書きの三点を同じ二つの文列型で採点した。合格ラインは置かない。見るのは、人間の推敲が下書きより上か、人間の推敲が別の推敲案より上か、別の推敲案が下書きより上か、点数と文字数の相関である。

- **v2b**：人間の推敲 / Fable の推敲 / 下書き。中間は教師に使っていない生成器である
- **v2c**：人間の推敲 / Composer の推敲 / 下書き。中間は三つ組みの教師と同じ系統である

採点の source は下書きそのもの（`base_text` = 下書き）。段落結合・分割・逆転の定義順位は使っていない。

24 件の書籍は keep 分割の学習側にも出る。人間の推敲本文が学習側の gold と完全一致したのは `he-v2-01`・`he-v2-02`・`he-v2-06` の 3 件。下表は 24 件全件である。この 3 件を除いても、下の向きは変わらない。

JSON は `outputs/hard_eval_v2{b,c}_report_sentseq_{keep_pairsplit,section_triples}.json`。点差の内訳は同名の `*_margins_*.json`。

再生成：

```text
PYTHONPATH=scripts:scripts-old .venv/bin/python3 scripts-old/score_hard_eval.py \
  --input data/hard_eval/bases_v2b_human_fable_copy.jsonl \
  --scorer sentseq --model outputs/pref-sentseq-keep-pairsplit \
  --report outputs/hard_eval_v2b_report_sentseq_keep_pairsplit.json
# v2c は bases_v2c_human_machine_copy.jsonl。モデルをもう一方も同様。
PYTHONPATH=scripts:scripts-old .venv/bin/python3 scripts-old/analyze_hard_eval_margins.py \
  --report outputs/hard_eval_v2b_report_sentseq_keep_pairsplit.json \
  --mid fable --out outputs/hard_eval_v2b_margins_sentseq_keep_pairsplit.json
# v2c は --mid machine。
```

### 7.1 人間の推敲が最上か（Top-1）

分母は 24 件。最上は人間の推敲である、という参考順位との一致。

| 試験 | keep-pairsplit | section-triples |
|---|---:|---:|
| v2b（中間は Fable） | 16/24 (0.667) | 17/24 (0.708) |
| v2c（中間は Composer） | 18/24 (0.750) | 15/24 (0.625) |

### 7.2 候補対ごとの勝ち（評価器の点数）

| 試験 | 対 | keep-pairsplit | section-triples |
|---|---|---:|---:|
| v2b | 人間の推敲 ＞ 下書き | 20/24 | 20/24 |
| v2b | 人間の推敲 ＞ Fable | 16/24 | 17/24 |
| v2b | Fable ＞ 下書き | 23/24 | 24/24 |
| v2c | 人間の推敲 ＞ 下書き | 20/24 | 20/24 |
| v2c | 人間の推敲 ＞ Composer | 18/24 | 15/24 |
| v2c | Composer ＞ 下書き | 21/24 | 23/24 |

人間の推敲対下書きで外した 4 件（`he-v2-08`・`he-v2-12`・`he-v2-13`・`he-v2-21`）は、両評価器で同じである。この 4 件は、その段落窓では人間の推敲本文と下書きが同一である。評価器の点数も人間と下書きで同点である。人間の推敲が下書きと異なる 20 件では、両評価器とも人間の推敲 ＞ 下書きは 20/20 である。

### 7.3 点数と文字数

項目ごとのスコアと文字数の Spearman の平均。

| 試験 | keep-pairsplit | section-triples |
|---|---:|---:|
| v2b | 0.244 | 0.203 |
| v2c | 0.451 | 0.277 |

### 7.4 この測定が示していること

人間の推敲対下書きは、学び直しの前後で 20/24 のままである。外した 4 件は人間の推敲と下書きが同一文で、点数も同点である。本文が異なる 20 件では両評価器とも 20/20 である。360 件の比較 6 と違い、ここは valid の 50 節そのものではない。学び直しは、この対を動かしていない。

Fable を中間にした試験では、三つ組みの文列型は人間の推敲を Fable より上に置く件が 1 件増え、Fable を下書きより上に置く件は 24 件すべてになった。教師に無い生成器でも、中間を下書きより上へは寄せている。

Composer を中間にした試験では、三つ組みの文列型は Composer を下書きより上に置く件が増え、人間の推敲を Composer より上に置く件は減った。三つ組みの教師は「人間の推敲 ＞ Composer ＞ 下書き」だが、この 24 件では下書きとの差は付きやすくなり、人間の推敲との差は付きにくくなっている。Composer に合わせた尺度になった、という疑いが残る。Fable 側ではその崩れは出ていない。

いまのところ、三つ組みの文列型を「人間の推敲が、意味を保った別案より必ず上」と信頼する根拠にはならない。360 件の一致と、この測定を並べた当時の記録である。

v2 / v2b / v2c の 24 件は、制御改悪とトークン上限のための段落窓である。候補を差し替えたあとも単位は窓のままなので、現行の評価目標（節の下書きに対する人間の推敲と、意味を保った別案）のための試験ではない。以後、評価器の採否・成功指標・学び直しの根拠にしない。

## 8. 難試験（setwise 文列型）

同じ 24 件について、三つ組み setwise 文列型（`outputs/pref-setwise-section-triples`、architecture=`local_stream_then_joint_tokens_v2`）でも採点した。§7 と同じ v2b / v2c 入力を使う。strict 比較の eps は 1e-6。人間の推敲本文と下書きが同一の候補は runtime で同値化する。

2026-08-14 の旧 DOK 成果物（architecture=`joint_transformer_on_sentence_tokens`）は、全候補が一様同点に崩壊し学習できていなかった。以下は修正版のみの結果である。

### 8.1 学習（valid 50 件）

328 train / 50 valid、40 epoch、best epoch 26。

| 指標 | 値 |
|---|---:|
| listwise loss | 1.4521102905 |
| human strict top-1 | 34/50 |
| exact full order | 19/50 |
| human ＞ Composer | 36/50 |
| human ＞ draft | 41/50 |
| Composer ＞ draft | 31/50 |

一様同点崩壊は解消し、listwise 損失と valid 指標は動いている。

### 8.2 人間の推敲が最上か（strict top-1）

分母 24 件。§7.1 の top1 とは同点の扱いが異なる場合がある。

| 試験 | setwise |
|---|---:|
| v2b（中間は Fable） | 11/24（top1 tie 1） |
| v2c（中間は Composer） | 12/24（top1 tie 1） |

pairwise 一致（参考順位との 3 方向一致の合計）：v2b 39/72、v2c 40/72。

### 8.3 候補対ごとの勝ち（strict `>`。win / tie / loss）

| 試験 | 対 | setwise |
|---|---|---:|
| v2b | 人間の推敲 ＞ 下書き | 14 win / 4 tie / 6 loss |
| v2b | 人間の推敲 ＞ Fable | 13 / 0 / 11 |
| v2b | Fable ＞ 下書き | 12 / 0 / 12 |
| v2c | 人間の推敲 ＞ 下書き | 14 / 4 / 6 |
| v2c | 人間の推敲 ＞ Composer | 14 / 0 / 10 |
| v2c | Composer ＞ 下書き | 12 / 0 / 12 |

tie 4 件（`he-v2-08`・`he-v2-12`・`he-v2-13`・`he-v2-21`）は人間の推敲本文と下書きが同一である。本文が異なる 20 件では、人間の推敲 ＞ 下書きは 14/20 である。

### 8.4 点数と文字数

項目ごとのスコアと文字数の Spearman の平均。

| 試験 | setwise |
|---|---:|
| v2b | 0.286 |
| v2c | 0.348 |

### 8.5 §7 の二モデルとの比較

§7 の表は改変しない。主な比較は 3 方向の strict pair 勝数、とくに本文が異なる 20 件の human ＞ draft である。

| 試験 | 対 | keep-pairsplit | section-triples | setwise |
|---|---|---:|---:|---:|
| v2b | 人間の推敲 ＞ 下書き（24 件） | 20/24 | 20/24 | 14/24 |
| v2b | 人間の推敲 ＞ 下書き（本文が異なる 20 件） | 20/20 | 20/20 | 14/20 |
| v2b | 人間の推敲 ＞ Fable | 16/24 | 17/24 | 13/24 |
| v2b | Fable ＞ 下書き | 23/24 | 24/24 | 12/24 |
| v2c | 人間の推敲 ＞ 下書き（24 件） | 20/24 | 20/24 | 14/24 |
| v2c | 人間の推敲 ＞ 下書き（本文が異なる 20 件） | 20/20 | 20/20 | 14/20 |
| v2c | 人間の推敲 ＞ Composer | 18/24 | 15/24 | 14/24 |
| v2c | Composer ＞ 下書き | 21/24 | 23/24 | 12/24 |

### 8.6 判断

修正版 setwise は、旧 joint-only 版の一様同点崩壊を解消し、学習自体は成立した。

一方、目的である「人間の推敲を最良に置く」は、v2b で strict top-1 が 11/24、v2c で 12/24 にとどまる。本文が異なる 20 件でも human ＞ draft は 14/20 で、§7 の keep-pairsplit・section-triples の 20/20 より悪い。human ＞ 別案も、v2b で 13/24、v2c で 14/24 で、keep-pairsplit（16/24・18/24）や BT 三つ組み section-triples（17/24・15/24）を上回らない。middle ＞ draft は v2b・v2c とも 12/24 で、旧二モデルの 21〜24/24 より大幅に低い。

この setwise 成果物を「人間の推敲を最良に置く評価器」として採用する根拠はない。三文を同時入力するだけでは、§7 で見えていた問題は解決しなかった。

出力：`outputs/hard_eval_v2b_report_setwise.json`、`outputs/hard_eval_v2c_report_setwise.json`。

再生成：

```text
make hard-eval-setwise INPUT=data/hard_eval/bases_v2b_human_fable_copy.jsonl \
  MODEL=outputs/pref-setwise-section-triples \
  REPORT=outputs/hard_eval_v2b_report_setwise.json
make hard-eval-setwise INPUT=data/hard_eval/bases_v2c_human_machine_copy.jsonl \
  MODEL=outputs/pref-setwise-section-triples \
  REPORT=outputs/hard_eval_v2c_report_setwise.json
```

## 9. 難試験（setwise human-top 文列型）

§8 の full-order 版と同じ 24 件について、human-top 損失版（`outputs/pref-setwise-section-human-top`、architecture=`local_stream_then_joint_tokens_v2`、loss_mode=`human_top`）でも採点した。§7 と同じ v2b / v2c 入力を使う。strict 比較の eps は 1e-6。人間の推敲本文と下書きが同一の候補は runtime で同値化する。

human-top 損失は ListMLE / Plackett-Luce の第 1 項のみ（`logsumexp([s_h,s_c,s_d]) - s_h`）。human ＞ Composer と human ＞ draft を課す。Composer ＞ draft は損失にも checkpoint 選択にも使わない。

### 9.1 学習（valid 50 件）

328 train / 50 valid、40 epoch、lr 3e-4、batch 32、best epoch 16。full-order 版（§8.1）と同条件。

| 指標 | human-top | full-order（§8.1） |
|---|---:|---:|
| human strict top-1 | 30/50 (0.60) | 34/50 |
| human ＞ Composer | 31/50 (0.62) | 36/50 |
| human ＞ draft | 35/50 (0.70) | 41/50 |
| Composer ＞ draft | 26/50 (0.52) | 31/50 |
| exact full order | 14/50 (0.28) | 19/50 |
| human-top loss | 0.9477953911 | — |
| listwise loss | 1.6015752554 | 1.4521102905 |

Composer ＞ draft と exact full order は診断値である。教師・checkpoint 選択には使わない。

### 9.2 人間の推敲が最上か（strict top-1）

分母 24 件。

| 試験 | human-top | full-order（§8.2） |
|---|---:|---:|
| v2b（中間は Fable） | 12/24（top1 tie 3） | 11/24（top1 tie 1） |
| v2c（中間は Composer） | 12/24（top1 tie 3） | 12/24（top1 tie 1） |

### 9.3 候補対ごとの勝ち（strict `>`。win / tie / loss）

| 試験 | 対 | human-top | full-order（§8.3） |
|---|---|---:|---:|
| v2b | 人間の推敲 ＞ 下書き | 13 / 4 / 7 | 14 / 4 / 6 |
| v2b | 人間の推敲 ＞ Fable | 17 / 0 / 7 | 13 / 0 / 11 |
| v2b | Fable ＞ 下書き | 10 / 0 / 14 | 12 / 0 / 12 |
| v2c | 人間の推敲 ＞ 下書き | 13 / 4 / 7 | 14 / 4 / 6 |
| v2c | 人間の推敲 ＞ Composer | 17 / 0 / 7 | 14 / 0 / 10 |
| v2c | Composer ＞ 下書き | 8 / 0 / 16 | 12 / 0 / 12 |

tie 4 件（`he-v2-08`・`he-v2-12`・`he-v2-13`・`he-v2-21`）は人間の推敲本文と下書きが同一である。本文が異なる 20 件では、人間の推敲 ＞ 下書きは human-top とも 13/20 である。

### 9.4 点数と文字数

項目ごとのスコアと文字数の Spearman の平均。

| 試験 | human-top | full-order（§8.4） |
|---|---:|---:|
| v2b | 0.307 | 0.286 |
| v2c | 0.283 | 0.348 |

### 9.5 §7・§8 との比較

§7・§8 の表は改変しない。主な比較は hard の human ＞ 別案と、本文が異なる 20 件の human ＞ draft である。

| 試験 | 対 | keep-pairsplit（§7） | full-order setwise（§8） | human-top setwise |
|---|---|---:|---:|---:|
| v2b | 人間の推敲 ＞ 下書き（本文が異なる 20 件） | 20/20 | 14/20 | 13/20 |
| v2b | 人間の推敲 ＞ Fable | 16/24 | 13/24 | 17/24 |
| v2c | 人間の推敲 ＞ 下書き（本文が異なる 20 件） | 20/20 | 14/20 | 13/20 |
| v2c | 人間の推敲 ＞ Composer | 18/24 | 14/24 | 17/24 |

### 9.6 判断

valid（§9.1）は full-order 版より全指標で悪化した（human strict top-1 34/50 → 30/50、human ＞ Composer 36/50 → 31/50、human ＞ draft 41/50 → 35/50）。hard の primary top-1 も v2b で 11/24 → 12/24、v2c では 12/24 のままである。本文相違 20 件の human ＞ draft は 14/20 → 13/20 と悪化した。

hard の human ＞ 別案だけ v2b で 13/24 → 17/24、v2c で 14/24 → 17/24 と数値が上がったが、valid の human 系指標はすべて full-order より低く、本文相違の human ＞ draft も悪化している。1 seed・24 件の loss trade-off の観測であり、human-top が setwise を改善した証拠とは言えない。

よって「Composer ＞ draft を外せば setwise が解決する」とは言えない。この成果物は採用しない。次工程は §9.7 を参照する。

### 9.7 追加診断（2026-08-14）

#### 9.7.1 保存 checkpoint の eval mode 再評価（train / valid）

| 指標 | full-order train | full-order valid | human-top train | human-top valid |
|---|---:|---:|---:|---:|
| human strict top-1 | 0.756 | 0.680 | 0.732 | 0.600 |
| human ＞ Composer | 0.777 | 0.720 | 0.771 | 0.620 |
| human ＞ draft | 0.912 | 0.820 | 0.805 | 0.700 |
| Composer ＞ draft | 0.695 | 0.620 | 0.494 | 0.520 |
| exact full order | 0.503 | 0.380 | 0.335 | 0.280 |
| listwise / human-top loss | 1.169 | 1.452 | 0.748 | 0.948 |

human-top は train 教師すら十分再現せず、valid との gap も残る。

#### 9.7.2 full-order → human-top の hard H>D 差分（24 件）

改善 1 件、悪化 2 件、不変 21 件。第三候補を Fable ↔ Composer で替えても H−D 符号反転 0 件、human-top margin 相関 0.9999。第三候補の干渉は H>D 失敗の主因ではない。

#### 9.7.3 human-top が外す 7 件と pointwise

human-top が本文相違 H>D を外す 7 件は、pointwise keep-pairsplit では全件正（margin +0.39〜+3.27）。setwise で source 相対信号が弱い可能性はある。ただし明示 diff / abs / cos / length だけが原因かは未分離。pointwise とは encoder・head・loss・教師密度・独立 forward も異なる。

#### 9.7.4 v2c・本文相違 20 件の H>C

| モデル | H>C | strict top-1 |
|---|---:|---:|
| pointwise keep-pairsplit | 17/20 | 17/20 |
| pointwise section-triples | 15/20 | 15/20 |
| setwise full-order | 13/20 | 12/20 |
| setwise human-top | 14/20 | 12/20 |

#### 9.7.5 人間の推敲本文が下書きと同一の 4 件

`he-v2-08`・`he-v2-12`・`he-v2-13`・`he-v2-21` は、いずれも `unit=window`、3 段落窓である。由来の節は人間が編集している（段落数が変わっている）。トークン予算のため節から連続 3 段落を切った結果、その窓では人間の推敲が下書きと同じになった。

人間の推敲 ＞ Composer は定義である。人間が直さなかった箇所を Composer が書き換えたとき、評価器が書き換えを上に置かないかを見る項目であり、除外しない。人間対下書きは同一文なので同点が正しい。strict top-1 で人間を唯一の最上にするのは、下書き候補と本文が同じなので不可能である。keep-pairsplit の v2c top-1 miss 6 件のうち 08 / 12 / 13 の 3 件がこれに当たる。

人間の推敲本文が下書きと異なる 20 件で、keep-pairsplit が人間 ＞ Composer を外したのは 03・06・23（margin −1.63、−4.21、−3.03）である。これは定義順位に対する現行評価器の失敗であり、人間側がよいかを人が付け直す対象ではない。

#### 9.7.6 判断の訂正

人間の推敲 ＞ Composer を未観測として、v2c で人が付け直す、という次工程は取らない。明示差分を setwise へ戻す実装も、現時点では決定しない。

v2 / v2b / v2c の 24 件は評価器の採否に使わない。次の評価器設計は未決である。

出力：`outputs/hard_eval_v2b_report_setwise_human_top.json`、`outputs/hard_eval_v2c_report_setwise_human_top.json`。

再生成：

```text
make hard-eval-setwise INPUT=data/hard_eval/bases_v2b_human_fable_copy.jsonl \
  MODEL=outputs/pref-setwise-section-human-top \
  REPORT=outputs/hard_eval_v2b_report_setwise_human_top.json
make hard-eval-setwise INPUT=data/hard_eval/bases_v2c_human_machine_copy.jsonl \
  MODEL=outputs/pref-setwise-section-human-top \
  REPORT=outputs/hard_eval_v2c_report_setwise_human_top.json
```


