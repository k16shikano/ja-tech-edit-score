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

本集計で使う識別子を次に固定する。

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

2026-08-14 の旧成果物（architecture=`joint_transformer_on_sentence_tokens`）は、全候補が一様同点に崩壊し学習できていなかった。以下は修正版のみの結果である。

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

## 10. 段階 1 から 7 の評価器（検証 50 節）

2026-08-15。`outputs/pref-multigranular/pref-multigranular-report/stages_summary.json` と各段階の item JSON から集計した。

`data/section_middle/pref_valid.jsonl` と同じ 50 三つ組（下書き・人間の推敲・Composer）を使う。成功は人間の推敲が最上（同点可）。今回の採点では最上同点は 0 件だった。

段階 5 が造る評価器である。段階 1 から 4 はその手前である。段階 6 は同じ学習を三点同時入力で行う。段階 7 は `outputs/pref-bt-keep` で不合格を付けたあと、段階 5 で点を付ける。

段階 2 から 7 は、学習に `--seed 0`、`--seed 1`、`--seed 2` を渡して 3 回行った。`--seed` は、追加した Transformer の重みを乱数で置くときと、学習例の順番を並べ替えるときに使う。段階 1 の文列型学習はこの引数を受け取らず、成果物は 1 個である。

段階 1 は `outputs/pref-sentseq-keep-pairsplit` である。学習では、各節について下書きと人間の推敲だけを渡し、人間の推敲の点が高くなるように重みを更新する。検証では、同じ計算を人間の推敲、Composer、下書きのそれぞれに適用する。表の「人間＞Composer」は、そうして出た人間の推敲の点が Composer の点より高かった件数である。人間の推敲と Composer のどちらがよいかは、学習では教えていない。段階 1 を `--seed` を変えて 3 回回したが、出力先が同じなので後の実行が前を上書きした。表の段階 1 の 3 行は、最後に残った 1 個の重みを 3 回数えたものである。

### 10.1 検証 50 節（段階 × `--seed`）

| 段階 | `--seed` | 人間最上 | 人間＞Composer | 人間＞下書き |
|---:|---:|---:|---:|---:|
| 1 | 0 | 46/50 | 46/50 | 50/50 |
| 1 | 1 | 46/50 | 46/50 | 50/50 |
| 1 | 2 | 46/50 | 46/50 | 50/50 |
| 2 | 0 | 34/50 | 35/50 | 41/50 |
| 2 | 1 | 37/50 | 38/50 | 43/50 |
| 2 | 2 | 36/50 | 38/50 | 44/50 |
| 3 | 0 | 31/50 | 35/50 | 38/50 |
| 3 | 1 | 38/50 | 40/50 | 44/50 |
| 3 | 2 | 35/50 | 37/50 | 40/50 |
| 4 | 0 | 29/50 | 32/50 | 35/50 |
| 4 | 1 | 36/50 | 38/50 | 39/50 |
| 4 | 2 | 35/50 | 35/50 | 40/50 |
| 5 | 0 | 28/50 | 34/50 | 36/50 |
| 5 | 1 | 33/50 | 35/50 | 39/50 |
| 5 | 2 | 37/50 | 40/50 | 42/50 |
| 6 | 0 | 32/50 | 38/50 | 36/50 |
| 6 | 1 | 35/50 | 40/50 | 37/50 |
| 6 | 2 | 31/50 | 35/50 | 40/50 |
| 7 | 0 | 27/50 | 33/50 | 35/50 |
| 7 | 1 | 32/50 | 34/50 | 38/50 |
| 7 | 2 | 36/50 | 39/50 | 41/50 |

### 10.2 人間を最上に置けなかった件

段階 1 は 4 件で人間の推敲を最上に置けなかった。4 件とも Composer の点のほうが人間より高く、下書きの点は人間より低い。人間が下書きより低い件は 0 である。

段階 5 の `--seed 2`（この段階では人間を最上に置けた件数が最も多い実行。37 件）は、13 件で人間の推敲を最上に置けなかった。評価器が人間の推敲より高い点を付けた文で分けると、次のとおりである。

- Composer の点は人間の推敲より高く、下書きの点は人間の推敲より低い：5 件
- 下書きの点は人間の推敲より高く、Composer の点は人間の推敲より低い：3 件
- Composer の点も下書きの点も、人間の推敲より高い：5 件

この 13 件のうち 10 件は、段階 1 では人間を最上に置けていた。段階 1 で置けなかった 4 件のうち、段階 5 では置けたのは 1 件、段階 1 でも段階 5 でも置けなかったのは 3 件である。

段階 7 は、段階 5 の点の前に `outputs/pref-bt-keep` で不合格を付ける。`--seed` が 0、1、2 のいずれでも、段階 5 では置けていた 1 件が置けなくなり、段階 5 では置けていなかった件が新たに置けるようにはならなかった。

### 10.3 学習（best epoch）

| 段階 | 出力 | `--seed 0` | `--seed 1` | `--seed 2` |
|---:|---|---:|---:|---:|
| 2 | pair-draft | 14 | 22 | 11 |
| 3 | pair-humantop | 12 | 24 | 12 |
| 4 | hunk-then-section | 18 | 42 | 37 |
| 5 | hunk-replay | 27 | 32 | 46 |
| 6 | joint replay | 14 | 44 | 24 |

### 10.4 段階 5 の train / valid gap（human among top）

| `--seed` | train | valid |
|---:|---:|---:|
| 0 | 0.725 | 0.56 |
| 1 | 0.893 | 0.66 |
| 2 | 0.928 | 0.74 |

### 10.5 判断

段階 5 は検証 50 節で人間の推敲を最上に置けなかった（`--seed 2` で 37 件）。

段階 1 の表で人間最上が 46 件と出るのは、人間の推敲の点が Composer の点より高かった件が 46、下書きの点より低かった件が 0 だからである。この 46 は、人間の推敲と Composer のどちらがよいかを学習した結果ではない。段階 2 以降をこの 46 件と比べて「戻らない」とは書かない。

段階 1 が学習した対では、この 50 節は 50 件とも人間の推敲の点のほうが下書きより高い。下書きと候補を一対で Transformer に渡す段階 2 では、それが 41 から 44 件になる。段階 5 では 36 から 42 件である。人間対 Composer の損失を足したあとも、学習した対（人間の推敲と下書き）で段階 1 より件数が減っている。

出力：`outputs/pref-multigranular/`。

### 10.6 工程 6 の 60 件（人間の推敲対選抜生成）

2026-08-15。工程 6 で人が読み比べた 60 件である。各件は下書きを見せたうえで、人間の推敲と、推敲モデルが 8 本出したうち当時の評価器が選んだ 1 本を左右隠して並べ、人が「採用するならどちらか」を付けた対である。同等は 0 件。段階 1 から 7 の評価器が同じ二つの文に点を付け、点が高い側と人の選択が一致した件数を数えた。評価器の同点は 0 件だったので、分母は 60 である。

段階 1 は `outputs/pref-multigranular/pref-sentseq-keep-pairsplit` の 1 個の重みである。一致は 30/60 だった。§5.3 の 29/60 は、工程 7 当時の同じ名前の成果物による数字である。

再生成：`.venv/bin/python3 scripts/eval_pref_multigranular_blind60.py`

| 段階 | `--seed 0` | `--seed 1` | `--seed 2` |
|---:|---:|---:|---:|
| 1 | 30/60 | （同じ重み） | （同じ重み） |
| 2 | 37/60 | 43/60 | 32/60 |
| 3 | 31/60 | 35/60 | 29/60 |
| 4 | 29/60 | 28/60 | 26/60 |
| 5 | 27/60 | 31/60 | 36/60 |
| 6 | 26/60 | 35/60 | 34/60 |
| 7 | 27/60 | 31/60 | 36/60 |

段階 7 の一致件数は、同じ `--seed` の段階 5 と同じだった。

出力：`outputs/pref-multigranular/blind60_gold_vs_adapter_selected.json`。

## 11. 人間検出（工程 6 の 60 件）

2026-08-15。`outputs/pref-detect-section` が、§10.6 と同じ 60 件に点を付けた。人の同等は 0 件、評価器の同点は 0 件なので、分母は 60 である。

人が選んだ側と、点が高い側が一致したのは 29 件である。人間の推敲の点が、アダプタが 8 本出したうち当時の評価器が選んだ 1 本より高かったのは 24 件である。人は 49 件で人間の推敲を選び、11 件でその生成を選んだ。人が人間の推敲を選んだ 49 件のうち一致は 21 件、人が生成を選んだ 11 件のうち一致は 8 件である。

60 件のうち節 50 件、段落内 10 件である。節では一致 24 件、人間の推敲の点が高いのは 20 件。段落内では一致 5 件、人間の推敲の点が高いのは 4 件。

対する生成は、データ A で SFT したアダプタが 8 本出したうち、当時の評価器が選んだ 1 本である。検証 50 件の Composer より、学習した人間の推敲に近い文面になりやすい。

出力：`outputs/pref-detect-section/blind60_gold_vs_adapter_selected.json`。

人間が「人間の推敲」と判定できた 49 件のうち、評価器が「人間の推敲」と判定できなかった数は下記の通り。

- 人間の推敲を 1、下書きと Composer を 0 として学んだ `outputs/pref-detect-section` では 28 件
- その損失に、Composer の点が下書きより高くなる項を足した `outputs/pref-detect-cd-section` では 28 件
- InfoNCE で学んだ `outputs/pref-nce-section` では 25 件
- 教師データ B の文列型 Bradley-Terry である `outputs/pref-sentseq-section-triples` では 25 件

## 12. B の検証 50 件（人間の推敲対 Composer）

2026-08-15。学習用データ B の検証 50 件について、人間の推敲と Composer の推敲を左右に並べ、役割を隠して人間が「どちらがましか」を判定した。
人間が人間の推敲を選んだのは 43 件、Composer の推敲を選んだのは 4 件、同等は 3 件である。

人間が「人間の推敲」と判定できた 43 件のうち、評価器が「人間の推敲」と判定できなかった数は下記の通り。

- 人間の推敲を 1、下書きと Composer を 0 として学んだ `outputs/pref-detect-section` では 4 件
- その損失に、Composer の点が下書きより高くなる項を足した `outputs/pref-detect-cd-section` では 5 件
- InfoNCE で学んだ `outputs/pref-nce-section` では 11 件
- 教師データ B の文列型 Bradley-Terry である `outputs/pref-sentseq-section-triples` では 2 件

出力：`outputs/blind50_pref_valid_gold_vs_composer.json`。



