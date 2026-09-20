# editor-as-distribution — 推敲評価器の構築計画

**実験名**: `editor-as-distribution`（略号 `ead`）
**対象**: コーディングエージェント
**前提文書**: `BRIEF.md`（これまでの実験記録）、`editor-as-distribution/evaluator-design.md`（理論的背景。着手前に必ず読むこと。人間向けの同内容 PDF は同ディレクトリの `evaluator-design.pdf`）
**実行環境**: NVIDIA RTX PRO 4000 Blackwell / VRAM 24GB 単体

---

## 確かめること

主仮説はこれである。

> **この編集者にとっての「良い推敲」は、質の軸上の高さではなく、この編集者の条件付き分布への近さとして測れる。**

BRIEF.md は「潜在的な推敲の質がスカラーとして埋め込めるかどうかは、検証すべき仮説。で、これはおそらく成り立っていない」と書いている。本実験はその否定側に立った上で、**では何として表現できるのか**に答えを出す。答えの候補が「分布」である。既存の `pref-*` 系列が「質を順序として表現する」試みだったことと、正面から対になる。

主仮説が真なら、次の三つが成り立つ。

| # | 下位命題 | 検証する run |
|---|---|---|
| H1 | 密度（対数尤度比）が、順序を学習したモデルを**難しい境界**で上回る | `ead-den-a`（B valid）、`ead-compose`（C） |
| H2 | 比較不可能な候補は**欠陥**を持つ。関門で落ちる件数が有意にあり、関門を外すと性能が落ちる | `ead-gate-defect`、`ead-compose` の ablation |
| H3 | 密度だけでは**負の推敲を検出できない**。絶対ラベルが要る | `ead-gate-level`、`ead-scale-diag` |

H3 が、関門と順位づけを分ける本当の理由である。密度は「近いか遠いか」しか言わず、**遠さの方向を持たない**。$s_{\mathrm{den}}$ が低い候補が、下書きより劣化しているのか、単にこの編集者らしくない別の良い推敲なのかを、密度は区別できない。この区別には原点と、原点の下方の水準が要る。ゆえに D が必要になり、ゆえに関門は順位づけとは別のデータで学習される。

**主仮説が偽になる条件**: `ead-den-a` の B valid 一致率が、既存の順序モデルの最良値を層内で上回らない場合。そのときは「順序でも分布でもない第三の表現が要る」という結論になり、本計画は棄却される。

---

## データの役割

| データ | 実体 | 件数 | 役割 |
|---|---|---|---|
| A (train) | `data/edit_sft_all/train.jsonl` | 1716 | 順位づけ（密度モデル）の学習。**`canonical.jsonl` ではない** |
| A2 | `data/revision_corpus/keep_section.jsonl` | 379 | 大局性の検証（うち 50 が section heldout） |
| B | `data/section_middle/pref_train.jsonl` / `pref_valid.jsonl` | 378 | **検証専用。学習に使わない**。難しい負例 |
| C | `data/blind_eval/items.jsonl` | 60 | 最終検証専用。最後まで触らない |
| D | `data/d/train.jsonl` / `valid.jsonl` | 800 | 関門 G2 の学習。**`ead-scale-diag` の結果で役割が確定する** |

`data/revision_corpus/canonical.jsonl`（1976）と `keep_*.jsonl` は**プール**であり、学習ファイルではない。詳細はタスク 0-1 を見よ。

---

## パスと命名規約

**相対パスの起点はリポジトリ根とする。** 本計画の成果物は、既存の慣行に合わせて `outputs/` 配下にまとめる。

```
editor-as-distribution/          # 計画と設計（本文書群）
├── plan.md
├── evaluator-design.md
└── evaluator-design.pdf

outputs/ead/                     # 本計画が生成するもの
├── reports/{run名}.md
├── adapters/{run名}/
├── scores/{run名}.jsonl
└── work/                        # 中間成果物（id 集合、分割表など）
```

run 名は実験名の略号 `ead-` を接頭辞とし、以下は kebab-case で内容を書く。**run 名は勝手に変えないこと。**

| run | 内容 | 実装 |
|---|---|---|
| `ead-audit` | スキーマと分割の検証 | 新規 `scripts/ead/audit.py` |
| `ead-scale-diag` | D の目盛りの診断（D の配分を決める） | 新規 `scripts/ead/scale_diag.py` |
| `ead-transfer-diag` | 易しい境界と難しい境界の乖離の確認 | 新規 `scripts/ead/transfer_diag.py` |
| `ead-floor` | 表層特徴のみの床 | 新規 `scripts/ead/floor.py` |
| `ead-metric-unify` | C 上の指標の統一、床と天井の確定 | 新規 `scripts/ead/metric_unify.py` |
| `ead-glob-mi` | 大局性（条件付き相互情報量）の検証 | 新規 `scripts/ead/glob_mi.py` |
| `ead-den-a` | 密度モデル（A train で学習、尤度比スコア） | 新規 `scripts/ead/den_train.py`, `den_score.py` |
| `ead-gate-defect` | 欠陥関門（規則ベース） | 新規 `scripts/ead/gate_defect.py` |
| `ead-gate-level` | 水準関門（累積リンク、D で学習） | 新規 `scripts/ead/gate_level.py` |
| `ead-compose` | 合成と C 検証 | 新規 `scripts/ead/compose.py` |

> **実装はすべて未着手である。** リポジトリには `editor-as-distribution/` の文書しかない。データと D 系評価器の重みは揃っているので、不足しているのは入口のコードだけである。既存スクリプト（`scripts/evaluate_d_epoch7.py` など）は流用してよいが、本計画が要求する出力形式はそのままでは得られない（タスク 0-2 を見よ）。

各 run に Makefile ターゲット `make ead-{run名}` を用意し、`make ead-p0` で Phase 0 を一括実行できるようにすること。

---

## 0. 設計の要点

評価器を**関門 + 順位づけ**に分解する。単一の得点関数ではない。

1. **比較不可能性は欠陥に由来する。** 致命的欠落・致命的書き足し・実質無修正を含む候補は、質の軸上のどこかに位置づくのではなく、軸に乗らない。ゆえに軸に乗せる前に落とす（関門 G1）
2. **原点の下方は密度では表現できない。** 密度は「近いか遠いか」しか言わず、遠さの方向を持たない。ゆえに絶対ラベルによる別の関門が要る（関門 G2）
3. **残った候補は密度で並べる。** 順序を学習しない。多峰性を許すので、等しく優れた相異なる推敲に優劣をつけずに済む（順位づけ R）

D の役割は**タスク 0-2（`ead-scale-diag`）の結果で確定する**。関門 G2 専用になるか、順位づけにも併用できるかが、そこで決まる。それまで Phase 3 の構成を決めてはならない。

データの役割は前掲の「データの役割」表のとおり。**密度モデルの学習に使えるのは `data/edit_sft_all/train.jsonl`（full 1716 / excl 1516）であり、`canonical.jsonl` の 1976 ではない。**

---

## 1. 環境構築

### 1.1 Blackwell 固有の注意

RTX PRO 4000 Blackwell は compute capability **sm_120**。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# 期待: (12, 0)。PyTorch は cu128 以降が必須。cu121/cu124 ビルドは動かない
```

- **flash-attn**: sm_120 向けホイールが無い場合、ソースビルドは失敗しやすい。**最初から諦めて `attn_implementation="sdpa"` を使う。** 本実験の規模では速度は律速にならない。
- **bitsandbytes**: 動かなければ `adamw_torch_fused` で代替可（メモリ +1GB 程度）。
- 全スクリプトで `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定する。

### 1.2 出力規約

すべてのスコア出力 jsonl は以下を必ず含める。**層化評価に必要なので省略不可。**

```json
{"item_id": "...", "source": "human|composer|adapter|draft|gen",
 "score": 0.0, "n_tokens": 0,
 "n_chars_draft": 0, "n_chars_cand": 0, "delta_chars": 0,
 "edit_distance_norm": 0.0}
```

seed は 0 / 1 / 2 の3通りで全実験を回し、平均と標準偏差を報告する。**単一 seed の結果を報告しないこと。** D は 640 件と小さく、seed 間分散が効果量を上回りうる。

---

## 2. Phase 0: 診断（最重要フェーズ）

**所要**: 2 日 / GPU 軽微
**目的**: 以降のデータ配分の根本判断を決める。ここを飛ばして先に進まない。

### タスク 0-1: スキーマと分割の検証

各 jsonl の件数・キー構造・文字長分布を `outputs/ead/reports/ead-audit.md` に記録する。

#### 前提: 二層構造

`revision_corpus/keep_*.jsonl` は**プール**（カテゴリ別の全量）であり、学習ファイルではない。`edit_sft_*/` の `train.jsonl` と `heldout.jsonl` が、そのプールを分割した結果である。

```
canonical.jsonl (1976)
├── keep_hunk_nopara.jsonl (1597)  ← A1 のプール
│   └── edit_sft_hunk_nopara/{train (1387), heldout (210)}
└── keep_section.jsonl (379)       ← A2 のプール
    └── edit_sft_section/{train (329), heldout (50)}

blind_eval/items.jsonl (60) ⊂ (hunk heldout ∪ section heldout)
  内訳: section heldout から 50、hunk heldout から乱択 10
```

> **したがって heldout や blind の下書きが `keep_*` や `canonical` と重なるのは設計上当然である。** ここで 0 を期待してはならない。避けるべきリークは「検証用 id が **SFT の train 側**に混ざること」であり、比較対象は `edit_sft_*/train.jsonl` である。

#### (a) 分割の整合性（構造の確認。0 を期待するものではない）

| # | 検査 | 期待 |
|---|---|---|
| S1 | `edit_sft_hunk_nopara/train` ∪ `heldout` = `keep_hunk_nopara` | 1387 + 210 = 1597 |
| S2 | `edit_sft_section/train` ∪ `heldout` = `keep_section` | 329 + 50 = 379 |
| S3 | `keep_hunk_nopara` ∪ `keep_section` = `canonical.jsonl` | 1597 + 379 = 1976 |
| S4 | `blind_eval/items` ⊆ (hunk heldout ∪ section heldout) | 60（section 50 + hunk 10） |

S1〜S4 が成り立たない場合は、上図の前提が誤っているので**停止して報告**する。件数の内訳（train 側の実数）も記録すること。

#### (b) リーク検査（0 を期待するもの。0 でなければ停止）

| # | 検査 | 期待 |
|---|---|---|
| L1 | `edit_sft_hunk_nopara/train` ∩ `edit_sft_hunk_nopara/heldout` | 0 |
| L2 | `edit_sft_section/train` ∩ `edit_sft_section/heldout` | 0 |
| L3 | `edit_sft_all/train` ∩ `blind_eval/items` | 0 |
| L4 | `pref_train` ∩ `pref_valid`（下書き単位） | 0 |
| L5 | **D の 200 件の下書き ∩ `blind_eval/items`** | 0 |

L5 は新規の検査である。D の 200 件は A1 のプール（1597）から取られており、C の 10 件も同じプールの heldout から乱択されている。両者が重なると、D で学習した関門が C の一部を既知として扱うことになる。**0 でなければ C の当該件を最終検証から除外する。**

#### (c) 配分に影響する交差（0 とは限らない。件数を記録して以降の判断に使う）

| # | 検査 | 何を決めるか |
|---|---|---|
| X1 | D の 200 件 ∩ `edit_sft_hunk_nopara/train` | 密度モデルから除外すべき id の集合（Phase 2） |
| X2 | D の 200 件 ∩ `edit_sft_hunk_nopara/heldout` | D が検証側にどれだけ食い込んでいるか |
| X3 | **`pref_valid` の下書き ∩ `edit_sft_section/heldout`** | **B 評価が清潔かどうか** |
| X4 | `pref_train` の下書き ∩ `edit_sft_section/train` | 同上 |

> **X3 が最も重要である。** B の検証 50 節が section heldout の 50 件と同一なら、密度モデルを `edit_sft_all/train` で学習する限り B 評価は清潔である。しかし `pref_valid` の下書きが section の **train 側**に入っているなら、密度モデルはその人間推敲を学習済みであり、**Phase 2 の主評価が無効になる**。
>
> その場合は、`ead-den-a-excl` の学習集合から `pref_valid` の下書きも追加で除外すること。X3 の結果が出るまで Phase 2 の学習集合は確定しない。

#### 成果物

- `outputs/ead/work/exclude_ids_D.json` — X1 で特定した、D 由来で密度モデルから除外する id
- `outputs/ead/work/exclude_ids_B.json` — X3 の結果しだいで、`pref_valid` 由来で除外する id（不要なら空）
- `outputs/ead/work/split_map.json` — 各 id がどの分割（hunk train / hunk heldout / section train / section heldout）に属するかの対応表。以降の全評価で層化と除外に使う

### タスク 0-2: 【決定的】目盛りの診断

> **これが Phase 0 の核心である。** D は原点の下方（下書きより悪い）を表現できる唯一の資源だが、目標タスクが要求するのは原点の**上方**の解像度、すなわち「良い」と「優れている」の区別である。D にその解像度があるかを測る。

**要求出力を生成する既存スクリプトは無い。** `scripts/evaluate_d_epoch7.py` は D valid の A_macro / A_micro は出すが、ラベル対別の分解は出さない。`scripts/ead/scale_diag.py` を新規実装すること。

#### 使う重み

| モデル | 重みの所在 | 備考 |
|---|---|---|
| `modernbert-d-interval-epoch7` | `outputs/pref-d-modernbert-cv/fold0/best` | epoch は `outputs/d_epoch7/d_epoch7.lock.json` で 7 に固定 |
| `pref-d-gpm-modernbert` | `outputs/pref-d-gpm-modernbert` | head_dim 16 / 32 / 64 すべて |
| `pref-d-bt-modernbert` | 同上ディレクトリ内 | スカラー BT の対照 |
| `pref-d-gpm-qwen3-8b` | `outputs/pref-d-gpm-qwen3-8b` | fold 0–2 のみ。打ち切りは BRIEF 記載どおりで問題ない |

#### ペア台帳の構成

`data/d/valid.jsonl`（160 行）から、**同一下書き内で位置ラベルが異なる候補どうし**のペアを全列挙する。下書きをまたぐペアは作らない（話題が交絡する）。各ペアに `(label_low, label_high)` を付与し、その組でグループ化する。

#### 各ラベル対について出す数値

| ラベル対 | 意味 | 注目度 |
|---|---|---|
| a vs b | 劣化 vs 下書き水準 | 関門 G2 の性能 |
| a vs c, a vs d | 劣化 vs 改善 | 関門 G2 の性能 |
| b vs c, b vs d | 下書き水準 vs 改善 | 関門 G2 の性能 |
| **c vs d** | **改善 vs 優れた改善** | **順位づけの性能。ここが本命** |

各対について、ペア数・一致率・Wilson 95% 信頼区間を出す。Wilson 区間は正規近似ではなく閉形式で計算すること（`c vs d` はペア数が少ない可能性が高く、正規近似だと区間が不正確になる）。

**判定基準:**

- `c vs d` の一致率の Wilson 下限が 0.5 を上回る → D は順位づけにも使える。Phase 3 で併用を検討する
- `c vs d` の区間が 0.5 をまたぐ → **D は関門 G2 専用とし、順位づけからは完全に外す**
- 区間が 0.5 をまたぐが下限が 0.45 以上、という境界的な場合 → **停止して報告**（§10）

さらに次を記録する。

- `c vs d` のペア数と、全ペアに占める割合。少なければ、D 全体の高い一致率（A_macro 0.79 前後）は易しい対に由来していたことになる
- 同じモデルの C 上の一致率（タスク 0-5 で算出）と、対別一致率の関係

### タスク 0-3: 転移の非対称性の確認

D の負例は汎用 Qwen3-8B の生成物、C の候補は A で SFT した適応済みアダプタの生成物である。難易度が違う。これを定量化する。

1. D の生成推敲 600 件と、`outputs/edit-sft-eval-v3/adapter_samples.jsonl` の 480 件について、表層特徴（§タスク0-4 と同じ）だけで「人間の推敲」と判別する分類器をそれぞれ学習・評価する
2. 前者の AUC が後者より明確に高ければ、**易しい境界と難しい境界が別物であることが確定する**
3. さらに、単位の違い（D は段落内 100%、C は段落またぎ 83%）を文字長分布の比較で記録する

### タスク 0-4: 床の確定

表層特徴だけの順序ロジスティック回帰を `data/d/train.jsonl` で学習し、`data/d/valid.jsonl` で評価する。

**特徴**: Δ文字数、その絶対値、文字数比、正規化 Levenshtein 距離、文数差、平均文長差、文長標準偏差差、type-token ratio 差、句読点密度差、漢字含有率差、ひらがな含有率差

**報告**: RPS、4値精度、隣接許容精度、**ラベル a の再現率と適合率**、Δ文字数十分位ごとの層内精度、各特徴の単変量 AUC、および**Δ文字数のみの1変数モデル**の性能

### タスク 0-5: 【必須】指標の統一と、床・天井の確定

> 過去の実験では、C 上の評価が「人間勝ち率」と「人との一致率」という**異なる指標**で記録されている。これらは比較可能ではない。当人のブラインド自己一致率が 100% でないため、勝率が高いことは良いことを意味しない。

#### (a) 主指標の定義

**一致率** = 評価器の選好の符号が、当人のブラインド判定と一致した件数 ÷ 分母。

分母の規約: **当人が「同等」とした件を除外する。** C では `data/blind_eval/judgments_gold_vs_adapter_selected.jsonl` の 60 件から同等 4 件を除いた **56 件**、B valid では `data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl` の 50 件から同等 3 件を除いた **47 件**。除外件数は必ず併記する。

#### (b) 天井の算出手順

「当人のブラインド自己一致率」には二つの解釈がありうる。**天井として使うのは前者である。**

| 量 | 定義 | 役割 |
|---|---|---|
| **再判定自己一致率**（test–retest） | 同一の当人が同一ペアを時間を空けて二度判定したときの符号一致率 | **これが天井。** 評価器が当人と一致できる上限 |
| 識別率 | 当人が「どちらが自分の推敲か」を当てられた割合（BRIEF では 8 割程度） | 参考値。課題の難しさを示すが、一致率の上限ではない |

**手順:**

1. `data/blind_eval/` 配下の再判定ファイル（`judgments.redo*.jsonl` 等）を全列挙し、`judgments_gold_vs_adapter_selected.jsonl` と **item_id と左右の提示順で突き合わせる**
2. 二度判定された item に限り、両回とも「同等」でないものを分母とし、符号一致率と Wilson 95% 区間を出す。これを **C の天井**とする
3. 再判定が無い、または分母が 20 件未満なら、**天井は「未測定」と記録し、暫定的に識別率を上限の代理として併記する**。この場合「天井までの残り」の報告は暫定値であることを毎回明示する
4. **B valid には再判定データが無い見込みである。** 無ければ B の天井は「未測定」とし、C の天井を流用してはならない。B の天井を確定するには 47 件の再判定（人間作業）が要る。必要性を §10 に従って報告する

再判定ファイルが期待と異なる構造（提示順が記録されていない等）だった場合は、**独自解釈で数値を出さず停止して報告する**。天井が再現不能な手順で決まると、以降の全フェーズの報告が再現不能になる。

#### (c) 再計算する既存モデルの範囲

**必須**（BRIEF に C 上の数値が記録されている、または現行採用）:

| モデル | BRIEF 記載の C 上の値 | 現在の指標 |
|---|---|---|
| `pref-sentseq-section-triples` | C 未評価（B valid のみ） | 現行採用モデルなので必ず含める |
| `pref-sentseq-a2b-modernbert-d` | 25/56 | 一致率（そのまま使える） |
| `pref-d-gpm-modernbert` (head_dim 16 / 32 / 64) | 33 / 33 / 30 ÷ 56 | 一致率（そのまま使える） |
| `pref-d-bt-modernbert` | 26/56 | 一致率（そのまま使える） |
| `a2-pdpo` (`pdpo_best`) | 27/60 | **勝率。一致率に再計算が必要** |

**任意**（重みが残っていれば含める。無ければ表に「重み欠損・再計算不能」と明記して行を残す）: `pref-bt-keep-pairsplit`、`pref-sentseq-keep-pairsplit`、`pref-setwise-*`、`pref-pair-*`、`pref-joint-*`、`pref-nce-section`、`pref-detect-section`、`pref-detect-cd-section`、`pref-d-gpm-qwen3-8b`（fold 0–2）、`modernbert-d-interval-epoch7`。

**含めない**: 生成側の実験（`b-generation-conditional-sft`）、および `a2-pdpo` の生成評価。評価器ではないため。

#### (d) 出力

`outputs/ead/reports/ead-metric-unify.md` に次を明記する。

- **床** = 0.5（偶然水準）、および `ead-floor` の表層特徴モデルの一致率
- **天井** = (b) で算出した再判定自己一致率（未測定ならその旨と識別率）
- 上記モデルの一致率を、C（分母 56）と B valid（分母 47）の二列で、床と天井の間に並べた表
- 各行に分母・除外件数・Wilson 区間を付す

**以降のすべての報告は「床からの上昇分 / 天井までの残り」の形式で行う。**

### Acceptance criteria (P0)

- [ ] 分割の整合性 S1〜S4 が成立している（train 側の実件数が記録されている）
- [ ] リーク検査 L1〜L5 がすべて 0
- [ ] X1〜X4 の交差件数が記録され、`exclude_ids_D.json` / `exclude_ids_B.json` / `split_map.json` が出力されている
- [ ] X3 の結果により、Phase 2 の学習集合が確定している
- [ ] `c vs d` の一致率と信頼区間が出ている
- [ ] D の役割（関門専用か、順位づけ併用か）が診断結果に基づいて決定され、記録されている
- [ ] 全既存モデルの C 一致率が同一指標で並んでいる
- [ ] 床と天井が数値で確定している

---

## 3. Phase 1: 大局性の検証（go / no-go）

**所要**: 2〜3 日 / GPU 5〜20 時間（上限 24 時間。超える見込みなら下記の縮退規則を適用）
**目的**: 仮説の検証。学習を一切行わない。棄却されれば以降の設計から丸ごと除去できるので、先に決着させる。

### 仮説

熟練編集者の推敲では、推敲文の離れた部分どうしが互いに情報を持つ。LLM の推敲では持たない。この量は編集量・長さ変化から独立に測定できる。

### タスク 1-1: スコアラ

- Qwen3-8B、bf16、**推論のみ**。VRAM 約 16GB + KV キャッシュ ≒ 17GB。24GB に収まる
- **学習しない。素のモデルを使う。** SFT 済みアダプタを使うと測っているものが変わる
- 載らなければ Qwen3-4B に落とす。理由を記録すること

### タスク 1-2: スコアリング

対象: **`keep_section.jsonl` の 379 節**。**段落をまたぐ推敲のみ。A1 は対象外。**

> `edit_sft_section/heldout.jsonl` の 50 節は 379 節の部分集合（id 完全一致 50/50）である。**379 + 50 = 429 ではない。** 379 節をスコアし、うち 50 節に heldout フラグを立てて層別に集計する。heldout / train の別は `outputs/ead/work/split_map.json` から引く。

推敲文を段落単位で K ブロックに分割（K < 4 の文書は除外し、除外数を記録）。各ブロック k について4条件で `log p(y_k | context)` を計算する。

| 条件 | 文脈 |
|---|---|
| A | `x` のみ |
| B | `x` + `y_{-k}`（同一文書の他ブロックの推敲結果） |
| C | `x` + `ỹ_{-k}`（**別文書**の推敲ブロック、トークン長 ±10% で整合） |
| D | `x` + 同一文書に対する Composer 推敲の対応ブロック |

推敲ソース3種: 人間推敲、Composer 推敲（`data/section_middle/revisions.jsonl`）、下書き（対照）。

#### 計算量の見積もりと縮退規則

**着手前に必ず見積もりを出すこと。** 概算は 379 節 × 平均 K ブロック × 4 条件 × 3 ソース × 3 seed。K = 6 なら約 82,000 forward になる。KV キャッシュの共有が効いていないと 24 時間を超える。

1. まず 20 節でパイロットを走らせ、1 forward あたりの実測時間から全体を外挿する
2. 外挿値が 24 時間を超えるなら、**次の順に縮退する**（勝手に対象件数を減らさないこと）

| 順 | 縮退 | 失うもの |
|---|---|---|
| 1 | 条件 A と D を 1 seed に減らす（C のサンプリングのみ 3 seed を維持） | A と D は補助的な記録なので影響は小さい |
| 2 | Qwen3-8B を Qwen3-4B に落とす | 日本語の対数尤度の質。理由を記録すること |
| 3 | K の探索を {4, 8, 16} から {4, 8} に減らす | 頑健性の確認が弱くなる |

3 まで適用しても超えるなら、**停止して報告する**（§10）。対象 379 節を減らすことで時間を稼いではならない。検定力が落ちて go/no-go の判断そのものが成立しなくなる。

**主統計量は (B − C) / トークン数。**

> **B − A を主統計量にしてはならない。** B は A より文脈が長いので内容と無関係に尤度が上がる。B − A は文脈長効果を大局性と呼んでいるにすぎない。C は同じ文脈長で無情報なので、B − C だけが「この文書が実際にどう推敲されたか」の情報を取り出す。

実装必須事項:
- `x` は全条件で共通プレフィックス。**KV キャッシュを再利用する**（しないと計算量が3倍以上）
- `ỹ` のサンプリングは seed 固定、3 seed で繰り返す
- ブロック境界のトークン位置を正確に取り、`y_k` のトークンのみを合計する

### タスク 1-3: 解析

`outputs/ead/reports/ead-glob-mi.md` に出力:

1. G の分布（人間 / Composer / 下書き）のバイオリンプロット
2. 対応のある Wilcoxon 符号順位検定と Cliff's delta
3. G と Δ文字数・編集距離の偏相関（Spearman）
4. Δ文字数十分位ごとの層内での群間差
5. **K ∈ {4, 8, 16} での符号の安定性**

### Acceptance criteria (P1)

**支持**（全項目を満たす場合）:
- [ ] G(人間) > G(Composer) が p < 0.01、Cliff's delta > 0.3
- [ ] G と Δ文字数の偏相関の絶対値 < 0.2
- [ ] Δ文字数の 7/10 以上の層で群間差の符号が一致
- [ ] K = 4, 8, 16 で符号が一致

**棄却**: 仮説を棄却し、補助信号 A を以降の設計から完全に除去する。これは失敗ではない。否定的結果として `outputs/ead/reports/ead-glob-mi.md` に明記し、Phase 2 に進む。

> **指標を作り直して再挑戦しないこと。** 反証条件を後から緩めるのは計画全体の意味を壊す。

---

## 4. Phase 2: 順位づけ R（密度モデル）

**所要**: 3 日 / GPU 10 時間
**目的**: `s_den(x,y) = log p_θ(y|x) − log p_0(y|x)` を得る。

### 設計上の注意

既存の `a2-pdpo` が同型の手法（adapter ON/OFF の対数比）であり、A2 test で良好、B valid で中程度、C で不明という結果を出している。本フェーズはその延長だが、**3点が異なる**。

1. **P-DPO ではなく素の SFT を使う。** P-DPO は chosen/rejected のペアを要求し、そこで生成過程の交絡が入る。密度モデルはペアを必要としない
2. **学習データが A2 (379) ではなく、段落内を含む A の train 全体。** 密度推定はデータ量が効く。件数は full 版 1716、excl 版 1516（= 1716 − X1 の 200）。`canonical.jsonl` の 1976 ではない
3. **B を学習に使わない。** B の「劣化なし確認済み」という性質は、難しい境界を測る唯一の資源である。学習に消費せず検証に温存する

### タスク 2-1: 二つのアダプタ

| アダプタ | 学習データ | 用途 |
|---|---|---|
| `outputs/ead/adapters/ead-den-a-excl` | `edit_sft_all/train.jsonl` − `outputs/ead/work/exclude_ids_D.json` − `outputs/ead/work/exclude_ids_B.json` | D および B の評価用 |
| `outputs/ead/adapters/ead-den-a-full` | `edit_sft_all/train.jsonl` 全量 | C の最終評価用 |

> ### ⚠ `canonical.jsonl` を学習に使ってはならない
>
> `canonical.jsonl` (1976) は heldout (210 + 50) を**含む**プールである。これで学習すると、C の 60 件の人間推敲が学習済みになり、$s_{\mathrm{den}}$ が人間側に有利に偏る。C は完全に無効になる。
>
> 学習に使えるのは `edit_sft_all/train.jsonl` である。タスク 0-1 の L3 で blind との交差が 0 であることを確認した上で使う。件数は S1〜S3 の結果から確定する（1976 − 260 = 1716 前後）。
>
> 既存の編集 SFT アダプタ（`adapter_samples.jsonl` を生成したもの）も同じ train 側で学習されているはずである。**タスク 0-1 で次の順に確認すること。**
>
> 1. チェックポイント同梱の `adapter_config.json` / `training_args.bin` / 学習ログから、学習に使った jsonl のパスを読む
> 2. パスが記録されていれば、そのファイルと `blind_eval/items.jsonl` の id 交差を取る（0 を期待）
> 3. パスが記録されていなければ、学習ログの件数と `edit_sft_all/train.jsonl` の 1716 を突き合わせる
> 4. いずれも取れなければ **「由来確認不能」と `ead-audit.md` に明記する**。この場合、Phase 4 の C 評価は「アダプタ側が C を学習済みでない保証がない」という但し書き付きの結果になる。勝手に「たぶん大丈夫」で進めないこと

> **除外版を省略すると、D の人間推敲は学習済みなので信号が楽観的に偏る。省略不可。** 除外対象はタスク 0-1 の X1 と X3 で確定する。

```yaml
base_model: Qwen3-8B          # 既存の a2-pdpo / 編集SFTアダプタと同じベース
quantization: 4bit (QLoRA)    # 8B bf16 は 16GB。LoRA 学習には 4bit が安全
dtype: bfloat16
attn_implementation: sdpa
lora:
  r: 16                        # 既存実験と揃える。r=32 も試す
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
train:
  max_seq_len: 4096
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  optim: adamw_8bit
  lr: 1e-4
  scheduler: cosine
  warmup_ratio: 0.03
  epochs: 3
  loss_mask: 推敲文のトークンのみ。下書き部分は -100
```

**VRAM**: 4bit 8B ≒ 5.5GB + LoRA/optimizer 0.3GB + 活性化 2GB + logits と勾配 2.5GB ≒ 11GB。余裕あり。OOM するなら原因は logits（vocab 151k × seq 4096）なので chunked CE を使う。

A2 は文脈が長い。**seq 長超過で truncate される件数を必ず記録**し、A2 の 20% を超えるなら報告する。

### タスク 2-2: スコアリング

`peft` の `disable_adapter()` で ON/OFF を切り替え、2回 forward して差を取る。**参照モデルを別途ロードしない。**

3通りの正規化をすべて保存する。
- `s_sum`: 差の総和
- `s_mean`: 差 / トークン数
- `s_resid`: `s_sum` を `|y|` と `Δ文字数` に回帰した残差。**回帰の fit 集合は、そのスコアを出したアダプタの学習集合と厳密に一致させる。** excl 版のスコアは excl の学習集合（1516 件）で fit、full 版は full の学習集合（1716 件）で fit する。heldout の下書きの統計が回帰係数に入ってはならない

> どれを使うかを暗黙に決めないこと。3通りすべてでタスク 2-3 の層化評価を行い、**層内性能が最も高いもの**を採用する。

### タスク 2-3: 評価（B が主戦場）

**主評価**: `data/section_middle/pref_valid.jsonl` (50節) での人間推敲 vs Composer 推敲の判別。

> B の Composer 推敲は「下書きより劣化していない」ことが確認済みである。したがってこの判別は品質の粗い差ではなく**個人語法の差**を問うている。順位づけの存在意義がここで決まる。

B の 50 節に対する人手ブラインド判定は **`data/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl`（50 行）** にある。判定プロトコルは `data/blind_eval/valid50_gold_vs_composer_protocol.json`。内訳は人間43 / Composer4 / 同等3。`docs/` には無いので探さないこと。**勝率ではなく、この判定との一致率を主指標とする。** 同等の3件は除外し、47件での符号一致率で報告する。

報告:
- 一致率（全体）と、**Δ文字数十分位ごとの層内一致率**
- 床（表層特徴モデル）および既存モデル（`pref-detect-cd-section` 等）との比較
- 正規化3通りの比較表
- 除外版と全量版の差

### Acceptance criteria (P2)

- [ ] B valid の一致率が、Δ文字数の 7/10 以上の層で床を有意に上回る
- [ ] 全体一致率と層内一致率の乖離が小さい（大きければ文字数依存が残っている）
- [ ] 採用する正規化方法が層内性能に基づいて選択され、根拠が記録されている

満たさない場合: LoRA rank {16, 32, 64}、lr {5e-5, 1e-4, 2e-4} を探索。それでも届かなければ報告して指示を仰ぐ。

---

## 5. Phase 3: 関門 G1 / G2

**所要**: 2 日 / GPU 数時間

### タスク 3-1: 欠陥関門 G1（規則ベース）

学習不要。以下を検出して候補を落とす。

| 欠陥 | 検出 |
|---|---|
| 実質無修正 | 正規化編集距離 < ε（ε は A の分布の下位1%から決める） |
| 致命的欠落 | 下書きの内容語のうち推敲文に現れない割合が閾値超、または文字数比 < 下限 |
| 致命的書き足し | 文字数比 > 上限、または推敲文の内容語のうち下書きに無いものの割合が閾値超 |

閾値は **`data/edit_sft_all/train.jsonl`（1716件）の人間推敲の分布から決める**。密度モデルと同じ母集団を使う。`canonical.jsonl` を使ってはならない（heldout の人間推敲が入り、heldout / C 上での G1 挙動が解釈できなくなる）。heldout 260 件と C 60 件に対する G1 の通過率は、**校準には使わず、別途記録するだけ**にする。人間の推敲がほぼ全件通過する（再現率 > 0.99）ように設定し、`data/d/train.jsonl` のラベル a に対する棄却率を記録する。

D の 800 件と `adapter_samples.jsonl` の 480 件に適用し、**それぞれ何件が落ちるか**を報告する。落ちる件数が極端に少ない/多い場合は閾値を再検討する。

### タスク 3-2: 水準関門 G2（累積リンク）

**Phase 0-2 の診断結果によって構成が変わる。**

- `c vs d` が偶然水準 → G2 は「a/b vs c/d」の**二値関門**として構成する。4値の順序回帰にする意味がない
- `c vs d` が有意 → 4値の累積リンクモデルとして構成し、Phase 4 で順位づけへの寄与も検討する

構成（4値の場合）:

```
入力 (x, y)
  ↓
outputs/ead/adapters/ead-den-a-excl を適用した p_θ  ← 完全に凍結
  ↓
最終層隠れ状態の mean pooling（y の部分のみ）
  ↓ concat
s_den(x,y)（Phase 2 の採用正規化）
  ↓
MLP (hidden 256, dropout 0.3) → g(x,y)
  ↓
累積リンク: P(Y ≤ k) = σ(θ_k − g),  θ_1 = b_1, θ_k = θ_{k-1} + softplus(b_k)
```

- **学習するのは MLP と θ のみ。** LLM も LoRA も凍結する
- **Δ文字数・編集距離を入力特徴に入れてはならない。** 入れれば必ずそれを使う。層化評価にのみ用いる
- 損失: 累積リンクのカテゴリ確率の NLL（proper scoring rule なので追加の較正手法は不要）
- **下書き単位の group-wise 5-fold CV**。同一下書き由来の4件が train と valid に分かれてはならない
- 3 seed × 5 fold

### タスク 3-3: 評価

`data/d/valid.jsonl` (160) 上で:

- **RPS**（主指標）
- **ラベル a の再現率と適合率**（症状「負の推敲を評価できない」の解消を示す唯一の指標）
- Δ文字数十分位ごとの層内 RPS
- `P(Y ≥ c)` の reliability diagram を全体と層別の両方で。**層別に見て予測確率が Δ文字数の単調関数になっていれば交絡が残っている**
- 床との差

### Acceptance criteria (P3)

- [ ] G1 が A の人間推敲を 99% 以上通過させ、かつ D のラベル a を有意な割合で棄却する
- [ ] G2 のラベル a 再現率が層内で床を上回る
- [ ] RPS が床を下回る
- [ ] reliability diagram が層別でも対角から大きく外れない

満たさない場合: D の 640 件では不足。**学習曲線（160/320/480/640件での性能）を添えて報告する**。これは「D の追加アノテーションが最優先の投資先である」という結論に直結する。

---

## 6. Phase 4: 合成と最終検証

**所要**: 2 日

### タスク 4-1: パイプライン

**学習で融合しない。コード側で合成する。** 融合した瞬間に全順序の仮定が復活する。

```python
def evaluate(x, y):
    # G1: 欠陥関門（軸に乗るかの判定。質の判定ではない）
    if has_defect(x, y):                  # 無修正 / 致命的欠落 / 致命的書き足し
        return Rejected("defect")

    # G2: 水準関門（絶対原点。信号2のみが担える）
    p_better = 1 - sigmoid(theta_2 - g(x, y))
    if p_better < TAU_LEVEL:              # コード側の定数
        return Rejected("degraded")

    # R: 順位づけ（多峰性を保つ）
    rank = s_den(x, y)

    # A: 大局性（P1が支持された場合のみ）
    if ENABLE_GLOBALITY and is_section(x):
        rank += W_GLOB * G(x, y)

    # C: 回送
    conf = 1 - entropy(ordinal_dist(x, y)) / log(K)
    return Decision(rank, p_better, conf)
```

`TAU_LEVEL` と `W_GLOB` は**学習せず、コード側の定数**とする。値は D valid と B valid でのみ調整し、**C では一切調整しない**。

### タスク 4-2: 最終検証（C）

> **C を選択に一度も使っていないことを、`outputs/ead/reports/ead-compose.md` の冒頭で明示的に宣言する。** ハイパーパラメータ選択・閾値調整・モデル選択のいずれにも使っていないこと。一度でも使えばそれ以降 C は検証集合ではない。

使用アダプタ: `outputs/ead/adapters/ead-den-a-full`（C の 60 件は heldout 由来なのでリークしない）。

1. `outputs/edit-sft-eval-v3/adapter_samples.jsonl` の8本から、新しい評価器で1本を選ぶ
2. 暫定評価器が選んだものと比較し、**一致件数 / 分岐件数**を集計する
3. 一致した件については、既存の人手判定を使って一致率を計算する

> **重要な制約**: 人手判定は「暫定評価器が選んだ1本」に対してのみ存在する。新しい評価器が別の本を選んだ場合、その本に対する判定は**存在しない**。したがって分岐件については追加の人手判定が必要になる。
>
> **その件数を先に見積もって報告すること。** 8本すべてに判定を取り直すのが理想だが（60×8=480件）、コストを考えて分岐分のみで足りる設計にする。
>
> **上限規則**: 分岐件数が **20 件を超えたら、判定を依頼する前に停止して報告する**（§10）。20 件以下なら、その分だけの追加判定を依頼してよい。判定依頼時は `data/blind_eval/valid50_gold_vs_composer_protocol.json` と同じ提示形式（左右をランダム化し、役割を見せず、下書きを文脈として出す）に従うこと。
>
> **追加判定が得られるまでは、C の一致率は「一致件のみで計算した暫定値」と明示して報告する。** 分岐件を評価器有利に扱ってはならない。

4. G1 が棄却した件数と、その内訳（無修正 / 欠落 / 書き足し）。棄却された候補を5件程度サンプル提示し、棄却が妥当かを人が確認できる形にする
5. G2 が棄却した件数と同様のサンプル提示

### Acceptance criteria (P4)

- [ ] C を選択に使っていないことが宣言されている
- [ ] 新旧の一致率が、床と天井の間に位置づけて報告されている
- [ ] 追加人手判定が必要な件数が見積もられている
- [ ] G1 / G2 の棄却例がサンプル提示されている

---

## 7. スケジュール

| 週 | フェーズ | GPU |
|---|---|---|
| 1 | **P0: 診断（目盛りの診断・指標統一・床天井の確定）** | 軽微 |
| 2 | **P1: 大局性の go/no-go** | 5〜20h |
| 3 | P2: 密度モデル（アダプタ2本 + B での検証） | 10h |
| 4 前半 | P3: 関門 G1 / G2 | 3h |
| 4 後半 | P4: 合成・C 検証・追加判定の見積もり | 2h |

**P0 タスク 0-2 の結果が出るまで P3 の構成を決めない。P1 の結果が出るまで P2 に着手しない。**

---

## 8. 報告フォーマット

各フェーズ終了時に `reports/{run名}.md` を出力し、以下を必ず含める。

1. 実行した設定（モデル、ハイパーパラメータ、seed、データ件数）
2. Acceptance criteria の各項目の充足状況（✓/✗ と数値）
3. **床と天井の間に位置づけた数値**
4. 層化した結果（全体値だけを報告しない）
5. 予想と異なった点と、考えられる原因
6. 次フェーズへの影響

---

## 9. やってはいけないこと

- **反証条件を後から緩めること。** P1 が棄却されたら指標を作り直さない
- **C を最終検証以外に使うこと**
- **B を学習に使うこと。** 難しい境界を測る唯一の資源であり、学習に消費してはならない
- **天井を独自解釈で算出すること。** 再判定ファイルの構造が期待と異なれば停止して報告する
- **除外版アダプタを省略して D を評価すること**
- **`canonical.jsonl` を密度モデルの学習に使うこと。** heldout を含むプールなので C が無効になる。学習に使えるのは `edit_sft_all/train.jsonl`
- **`keep_*` と heldout の交差を「リーク」と解釈すること。** `keep_*` はプールであり、heldout がその部分集合なのは設計上正しい。比較すべき相手は `edit_sft_*/train.jsonl` である
- **勝率と一致率を混ぜて比較すること。** 天井が 100% でない以上、勝率が高いことは良いことを意味しない
- **全体値だけを報告し、層化を省略すること**
- **Δ文字数や編集距離をモデルの入力特徴に入れること**
- **Spearman 相関を主指標にすること**
- **三つの信号を単一スカラーに学習で融合すること**
- **単一 seed の結果を報告すること**
- **100% の一致率を目標に据えること。** 当人のブラインド自己一致率が上限である

---

## 10. 判断を仰ぐべき分岐

- ファイル件数が期待と異なる
- 分割の整合性 S1〜S4 のいずれかが成立しない（前提の構造が誤っている）
- リーク検査 L1〜L5 のいずれかが 0 でない
- X3 で `pref_valid` の下書きが section の train 側に入っていた
- **タスク 0-2 の `c vs d` が境界的**（Wilson 区間が 0.5 をまたぐが下限 0.45 以上）
- タスク 0-5 で、再判定ファイルの構造が期待と異なり天井が算出できない
- タスク 0-1 で、既存編集 SFT アダプタの由来が確認不能
- Phase 1 の見積もりが、縮退規則を 3 段まで適用しても 24 時間を超える
- Phase 4 の分岐件数が 20 件を超える
- A2 の truncate 率が 20% を超える
- P1 の p 値が 0.01〜0.05 の間
- P2 でハイパーパラメータ探索後も acceptance criteria に届かない
- VRAM が足りずモデルサイズを落とす必要がある
- C の追加人手判定が当初見積もりを大きく超える
