# Structure Eval（パラグラフライティング層別評価）

## 目的

「パラグラフライティングができているか」を測る正当な測定器を、実編集ペア（下書き vs 人間編集）だけから作る。機械的な改悪（段落結合・過剰分割・順序逆転）は使わない。`.cursor/rules/preference-eval-goal.mdc` の方針に従う。

## パイプライン

```bash
make structure-eval-data   # アラインメント + 層別 Hard Eval 生成
make structure-eval-score  # 既存選好モデルで層別採点（unclassified 含む）
```

成果物:

- `data/section_edit_profile.jsonl` — 節ペアごとの編集プロファイル（coverage, expression_change.total 含む）
- `data/section_edit_profile_preview.md` — 分布と代表例
- `data/hard_eval/{structure_dominant,expression_dominant,mixed,unclassified}.jsonl`
- `data/structure_train_candidates.jsonl` — raw 由来の構成支配候補（学習用）
- `outputs/structure_eval/summary.md` — モデル×層の勝率・点差表

## アラインメントと指標

各節ペアについて:

1. `\n\n` で段落分割
2. 各段落を文分割（`。` `！` `？` と閉じ括弧の後。コードブロック内は分割しない）
3. 各文に (段落番号, 段落内番号) を付与
4. `difflib.SequenceMatcher.ratio()` で類似度 0.6 以上を候補とし、貪欲法で一対一対応

### alignment.coverage

対応文の文字数 / その側の全文字数。下書き・編集後それぞれ算出し、`min(source, edited)` を記録する。

アラインメント失敗（対応が極端に少ない）を構成支配層から除外するため、structure-dominant では **両側 coverage ≥ 0.7** を必須とする。

### expression_change

- `matched`: 対応文の `1 − 平均類似度`（従来どおり）
- `total`: 未対応文の文字数も重み 1.0 として含めた全体の表現変化量  
  `(matched_expr × matched_chars + unmatched_chars) / total_chars`
- `unmatched_char_ratio`: 未対応文（追加・削除）の文字数 / 両側文字数合計

層分けの high/low 判定は **`expression_change.total`** を使う。structure-dominant の low 条件だけ **`expression_change.matched < 0.15`** のまま（対応文の語句変化が小さいことを要求）。

### structure_change

- `order_inversion_rate`: 対応文の相対順序の転倒数を正規化
- `grouping_change_rate`: 対応文ペアのうち、下書きと編集後で同段落かどうかが変わった比率
- `paragraph_count_delta_rate`: 段落数変化率
- `overall`: 上記3つの最大値

### 層分け用 structure_signal

```
structure_signal = max(order_inversion_rate, grouping_change_rate, paragraph_count_delta_rate)
```

held-out では順序・グルーピングだけが大きい例が少なく、段落再構成が主な構成変化になるため。

## 層の定義（held-out 112 節ペア）

| 閾値 | 値 |
|---|---|
| min_matched_sentences | 2 |
| min_alignment_coverage | 0.7（structure-dominant のみ必須） |
| high_expression_change | 0.15（`expression_change.total`） |
| low_expression_change | 0.15（`expression_change.matched`、structure-dominant） |
| high_structure_change | 0.20 |

| 層 | 条件 | held-out 件数（2048 token 以内） |
|---|---|---:|
| structure-dominant | structure_signal ≥ 0.20 かつ matched < 0.15 かつ **coverage ≥ 0.7** | **9**（fits_512: 3） |
| expression-dominant | total ≥ 0.15 かつ structure_signal < 0.20 | **8**（fits_512: 1） |
| mixed | total ≥ 0.15 かつ structure_signal ≥ 0.20 | **13**（fits_512: 2） |
| unclassified | 上記3層のいずれにも入らない | **72**（held-out 層外 78、fits_512: 9） |

トークン長 2048 超の drop: structure-dominant 1、expression-dominant 1、mixed 2、unclassified 6。

### coverage 修正による structure-dominant の変化

| 指標 | 修正前 | 修正後 |
|---|---:|---:|
| structure-dominant（held-out、2048 以内） | 18 | **9** |
| coverage 不足で除外 | — | **8** |

除外例（いずれも `alignment_coverage_below_threshold`）:

- `websocket-from-security-...-CSWSH-...`: coverage=0.049/1.000、matched=0、total=0.907、del=32（アラインメントほぼ失敗）
- `websocket-from-security-...-FIN-opcode-...`: coverage=1.000/0.448、matched=0、total=0.381、add=20
- `picoruby-...-PicoRuby-...`: coverage=0.613/0.695、matched=0.011、total=0.357

websocket-from-security プロジェクトの大規模追加・削除を伴う節が、旧定義では structure-dominant に混入していた。

raw 学習候補（`structure_train_candidates.jsonl`）: **27** 件。

## Hard Eval 形式

各項目:

- `base_text` = 下書き
- 候補 = `human`（編集後）と `base`（下書きのまま）
- `human.best_id = "human"`, `rank = ["human", "base"]`
- `status: "labeled"`

## 採点結果

### human > base 勝率

| 層 | pref-bt | pref-ce-beyond-para | pref-ce-ml2048 | pref-ce-with-negative-construct-example |
|---|---:|---:|---:|---:|
| structure_dominant（n=9） | 100% | 100% | 100% | 100% |
| expression_dominant（n=8） | 100% | 100% | 100% | 100% |
| mixed（n=13） | 100% | 100% | 100% | 100% |
| unclassified（n=72） | 90.3% | 100% | 98.6% | 97.2% |

3層（structure / expression / mixed）は全モデルで勝率 100%。unclassified のみ pref-bt で 90.3%、他 CE モデルでも 97–100%。

### 点差（score(human) − score(base)）

| 層 | モデル | 中央値 | 最小 | 中央値/モデル中央値 |
|---|---|---:|---:|---:|
| structure_dominant | pref-bt | 6.68 | 1.74 | **1.65** |
| structure_dominant | pref-ce-beyond-para | 68.61 | 9.44 | **0.995** |
| structure_dominant | pref-ce-ml2048 | 74.43 | 27.25 | **0.970** |
| structure_dominant | pref-ce-with-negative-construct-example | 86.84 | 6.08 | **1.34** |
| expression_dominant | pref-bt | 9.20 | 3.99 | 2.27 |
| expression_dominant | pref-ce-beyond-para | 117.13 | 42.22 | 1.70 |
| mixed | pref-bt | 11.58 | 3.99 | 2.86 |
| mixed | pref-ce-with-negative-construct-example | 163.75 | 58.03 | 2.52 |
| unclassified | pref-bt | 3.24 | **−0.23** | **0.80** |
| unclassified | pref-ce-beyond-para | 35.95 | 0.00 | **0.52** |
| unclassified | pref-ce-ml2048 | 62.48 | **−12.96** | **0.81** |
| unclassified | pref-ce-with-negative-construct-example | 47.39 | **−4.10** | **0.73** |

正規化は、各モデルの全ペア点差中央値で割った値（`summary.json` の `model_margin_medians` 参照）。

## 読み方

### 勝率 100% の限界

- 候補が2択（下書き vs 編集後）のみの Hard Eval 形式は、学習済み選好モデルの学習目標（編集後を選ぶ）に近い
- 3層すべてで勝率 100% は、層分けが「構成 vs 表現」を分離できている証拠にはならない
- unclassified では pref-bt 90.3%、CE モデルでも負の点差が少数発生し、勝率だけでは見えない差が出る

### 点差で見えること

**構成支配 vs unclassified（正規化中央値）**: 4モデルすべてで structure-dominant > unclassified（例: pref-bt 1.65 vs 0.80）。構成支配ペアの点差は unclassified より**小さくない**。

**構成支配 vs expression/mixed**: pref-bt では structure（1.65）< expression（2.27）< mixed（2.86）。CE モデルでも structure の正規化中央値は expression/mixed より低い傾向（0.97–1.34 vs 1.4–2.5）。表現・混合層の方が点差が大きい。

**解釈**: 層分けは「表現変化が大きいほど点差も大きい」という相関は示唆するが、構成支配層が「判定が難しい（点差が小さい）」ケースを集めているわけではない。unclassified の方が点差は小さく、判定も揺れやすい。

### 今後

- 層内で部分ウィンドウ化、表現を固定した対照
- 構成軸専用モデルの学習（`structure_train_candidates.jsonl` 27 件）
- unclassified を含めた点差集計を標準とし、勝率だけに依存しない

詳細は `outputs/structure_eval/summary.md` と `data/hard_eval/structure_eval_build_report.json` を参照。
