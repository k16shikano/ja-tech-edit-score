# 学習用データ D と GPM 選好ベクトル v(y|x) の実験計画

教師データ D（800 行）を固定し、下書き x と候補 y から **選好ベクトル** v(y|x) を学ぶ。
ペアの向きは、同じ原稿内で位置ラベルが異なる候補どうしから作る。
循環選好の有無は本実験の対象外である。

データの作り方は [DATA.md](DATA.md)。
区間損失 f(x,y) の実験（スカラー）は [plan-d-interval.md](plan-d-interval.md) と別線である。
ペア台帳の定義は [EXPERIMENT_B_CONDITIONAL_SFT.md](EXPERIMENT_B_CONDITIONAL_SFT.md) 2.2.2 節と 2.3 節に合わせる。

## 1. 測ること

**主問い**：報酬（選好）をスカラー s(x,y) ではなくベクトル v(y|x) に埋め込めるか。
D の位置ラベルから導くペア順位を、同じエンコーダ・同じ分割・同じペア台帳のもとで、GPM とスカラー BT のどちらが再現するか。

固定する性質（GPM 側）：

- 各 (x, y) に偶数次元ベクトル v(y|x) ∈ R^{2k} を置く（実装では **head_dim = 2k**）。
- 同一 x の下で二候補 a, b を比較するときのスコアは、歪対称和のみを使う：

  s(a, b | x) = Σ_{m=1}^{k} ( v_a^{(m,0)} v_b^{(m,1)} − v_a^{(m,1)} v_b^{(m,0)} )

- したがって s(b, a | x) = −s(a, b | x)、s(y, y | x) = 0。
- 学習で見るのは **向き付きペア** の符号一致。全候補を一本の実数で並べ替えることは要求しない。

本実験で **見ない** こと：

- 選好の循環を作る・検出する（D のラベル順位は推移的）。
- A1 / B / A2 を混ぜた教師。
- 位置ラベルを分類正解率だけで測る（区間所属率は副次参考に留める）。

## 2. 教師データとペア台帳

### 2.1 行データ（既存）

| 項目 | 内容 |
| --- | --- |
| 正本 | `data/d/dataset.jsonl`（800 行） |
| 下書き数 | 200 |
| 原稿あたり候補 | Qwen 生成 3 本 + 人間推敲 1 本 |
| 位置 | a < b < c < d（a=劣化、b≒下書き、c=改善、d≒人間） |
| 分割 | 5 fold（`data/d/folds/fold{0..4}_{train,valid}.jsonl`）、乱数 `--seed` 0 |

各行は `item_id`, `draft`, `y`, `position`, `y_kind` を持つ（[DATA.md](DATA.md)）。

### 2.2 ペア台帳（原稿内）

原稿 i の下書き x_i、候補 y_ij、位置 ℓ_ij ∈ {0,1,2,3}（a→0, …, d→3）とする。

観測ペア集合：

P_i = { (j,k) : j<k, y_ij ≠ x_i, y_ik ≠ x_i, ℓ_ij ≠ ℓ_ik }

- 下書きと同一本文の候補は除外する（文字列一致。strip や正規化は挟まない）。
- 同位置の組は **損失にも評価にも入れない**（同等とみなして 0 点に潰さない）。
- 別原稿の候補間には順位を作らない。

向き付き教師行（1 ペア → 1 行）：

- ℓ_ij < ℓ_ik なら、勝者 y_ik、敗者 y_ij、下書き x_i。
- 損失：softplus( −s(y_ik, y_ij | x_i) )（Bradley-Terry 型）。

原稿内損失は |P_i| で平均し、バッチは原稿間で平均する。
1 原稿あたり最大 6 ペア。

台帳の保存先（新規）：

- `data/d/pair_ledger.jsonl`（800 行から展開した directed 行）
- `data/d/pair_ledger_stats.json`（原稿数、ペア数、位置別内訳）

入口：`make pref-d-pair-ledger`（実装後）。

## 3. モデル

### 3.1 主モデル（GPM）

| 項目 | 値 |
| --- | --- |
| ベース | `sbintuitions/modernbert-ja-310m`（315M、全パラメータ更新） |
| 入力 | 下書き x と候補 y の二文入力（クロスエンコーダ。plan-d-interval と同型） |
| 出力 | v(y\|x) = Linear(h)，h は [CLS] 相当のプール出力 |
| **head_dim** | **64**（k=32 ブロック） |
| 正則化 | weight decay 0.01。v の L2 ノルムに対する弱いペナルティは任意（既定オフ） |
| 成果物 | `outputs/pref-d-gpm-modernbert/` |

head_dim=4 のような最小次元は使わない。表現力不足でスカラー BT と区別が付かなくなる。

### 3.2 対照（スカラー BT、必須）

同じ D・同じペア台帳・同じ分割・同じ最適化条件で、出力だけ 1 次元とする。

| 項目 | 値 |
| --- | --- |
| スコア | Δ(x,y) = g(x,y) − g(x,x)（plan-d-interval と同じ **f(x,y)** の形） |
| ペア損失 | softplus( −(Δ(x, y_w) − Δ(x, y_l)) ) |
| 成果物 | `outputs/pref-d-bt-modernbert/` |

GPM との差は **出力の次元と比較関数だけ** に限定する。エンコーダ・tokenizer・max_length・fold・epoch 採用規則は揃える。

### 3.3 参考（既存区間モデル）

`outputs/pref-d-modernbert/`（区間損失 f）および D epoch7 復元評価は、**別系統のスカラー** として参考記録のみに置く。
GPM 採否の主判定には使わない。

## 4. 学習

plan-d-interval と揃える項目：

| 項目 | 値 |
| --- | --- |
| 教師行 | ペア台帳からの directed 行（800 行の位置ラベルから生成） |
| 分割 | 5 fold CV。fold 0 valid で epoch を選び、fold 1–4 も同じ epoch 番号 |
| epoch 上限 | 40 |
| epoch 採用 | fold 0 valid の **ペア符号一致率（A_micro）** が最大の epoch |
| 最適化 | AdamW |
| 学習率（エンコーダ） | 5e-6 |
| 学習率（ヘッド） | 1e-4 |
| weight decay | 0.01 |
| warmup | 10% |
| batch | 16 原稿分のペア（ペア数可変。原稿内平均後にバッチ平均） |
| 精度 | bf16 |
| max_length | 1024 |
| 乱数 `--seed` | 0（分割・初期化） |

Makefile 入口（実装後）：

```text
make pref-d-pair-ledger
make pref-d-gpm-modernbert
make pref-d-bt-modernbert
make eval-pref-d-gpm-modernbert
make eval-pref-d-bt-modernbert
```

本学習は手元 GPU で回す。1 fold・1 epoch のスモークで入口を確認してから本番に入る。

## 5. 評価

### 5.1 主指標（ペア符号一致）

[EXPERIMENT_B_CONDITIONAL_SFT.md](EXPERIMENT_B_CONDITIONAL_SFT.md) 2.2.2 節と同じ式を使う。

- 原稿 i の評価ペア Q_i ⊆ P_i（有限スコアが取れた組）。
- GPM：m_ijk = (ℓ_ij − ℓ_ik) · s(y_ik, y_ij | x_i)
- BT：m_ijk = (ℓ_ij − ℓ_ik) · (Δ(x_i, y_ik) − Δ(x_i, y_ij))
- u(m) は正・零・負で 1 / ½ / 0。

報告する数：

- **A_macro**（|Q_i|>0 の原稿平均）
- **A_micro**（全ペア合算）
- 正順・逆順・同点の件数、Σ|Q_i|、長さ超過・エラー件数

**GPM と BT を同じ Q_i で並べ、A_macro / A_micro の差を主結果とする。**

部分集合（必ず別表）：

| 部分集合 | 行数・件数の目安 | 用途 |
| --- | --- | --- |
| Qwen 生成同士 | 原稿内生成 3 本の組 | 生成元差の捷経を切り分け |
| 人間 d を含むペア | 人間行が勝者側に来るペア | 著者判別の参考 |
| a vs c のみ | 385 行に対応するペア | 符号分離の核心 |

### 5.2 ベクトル固有の副次指標

| 指標 | 内容 |
| --- | --- |
| ‖v(y\|x)‖ の分布 | 位置 a/b/c/d 別。同一 x で候補間の距離が潰れていないか |
| 同一点 s≈0 の率 | 異位置ペアで |s| が閾値以下になる割合（ベクトルが比較を放棄していないか） |
| 読み出し ablation（任意） | v 上に 1 次元 Linear を足し f̂(x,y)=w·v(y\|x) を学習 **せず**、学習済み v から最小二乗で f̂ を fit したときの A_micro | v がスカラー写像を内包しているかの参考 |

位置ラベル **そのもの** への多クラス分類精度は副次参考。GPM の主目的はペア向きの再現である。

### 5.3 閾値・符号

- 同点は保存した未丸め s の **厳密一致**（float 変換後の偶然一致は同点に数えない）。
- 評価後に符号を反転しない。学習時の s の向き定義を lock ファイルに書く。

## 6. 成果物

| パス | 内容 |
| --- | --- |
| `data/d/pair_ledger.jsonl` | directed 教師行 |
| `data/d/pair_ledger_stats.json` | 件数集計 |
| `outputs/pref-d-gpm-modernbert/` | GPM 主モデル（head_dim=64） |
| `outputs/pref-d-bt-modernbert/` | スカラー BT 対照 |
| `outputs/pref-d-gpm-modernbert/eval_cv_report.json` | 5 fold 連結の主指標 |
| `outputs/pref-d-bt-modernbert/eval_cv_report.json` | 対照の同型レポート |
| `outputs/pref-d-gpm-modernbert/gpm.lock.json` | 比較関数、head_dim、符号、commit、checkpoint SHA |

## 7. 実施順

1. `make pref-d-data`（未生成なら 800 行と 5 fold）
2. `make pref-d-pair-ledger`（台帳固定）
3. スカラー BT 5 fold 学習 → eval（対照を先に通す）
4. GPM head_dim=64 の 5 fold 学習 → eval
5. A_macro / A_micro で GPM vs BT を比較
6. ベクトル副次指標を `eval_cv_report.json` に追記

## 8. 解釈

| 結果 | 読み方 |
| --- | --- |
| GPM の A_micro が BT を上回る | 同じ D ペアで、ベクトル＋歪対称比較がスカラー Δ より向きを再現している |
| 同等 | 本設定の 2k 次元 GPM は D 上でスカラー BT と差が付かない |
| BT のみ良好 | ベクトル化の追加自由度は不要（少なくとも D のペア教師では） |
| 人間 d ペアだけ GPM が高い | 著者判別の疑い。Qwen 同士部分集合を主に見る |

循環がなくても、スカラーは全ペアを一つの数値軸に投影する。
GPM は投影軸を複数持ち、比較はペアごとの歪対称スコアに限る。
この差が D で効くかを測るのが本計画の主旨である。
