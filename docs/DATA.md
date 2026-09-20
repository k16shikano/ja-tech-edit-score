# 学習データの作り方

BRIEF の A / A1 / A2 / B / C / D が、どの手順でできたか。実施済みである。作り直すときの入口もここ。
ディレクトリの置き場は [data/README.md](../data/README.md)。

## A（推敲前後ペア全体）

人がレビューして採用した、下書きと人間の推敲の対。全 1976 件。正本は `data/revision_corpus/canonical.jsonl`。

採掘は Git のブランチ対から行う。短い箇所（空行を含まない）は `make data`、段落境界を含む節は `make mine-sections`。
レビューは `make edit-sft-review`（節）と `make edit-sft-review-hunk`（断片）。採用した対を `make edit-sft-export-keeps` で書き出し、`make edit-sft-promote-reviewed` で正本へ載せる。

節か断片かは、空行（段落境界）の有無で分ける。

- 節（A2 の元）: 379 件。`data/revision_corpus/keep_section.jsonl`
- 断片（A1 の元）: 1597 件。`data/revision_corpus/keep_hunk_nopara.jsonl`

## ペア単位の分割（A1 / A2 共通）

全 1976 件を、書籍に関係なくペア単位で学習と検証に分ける。乱択のシードは 42。検証側は 260 件。節と断片の比率を保つ層化である。割当の再現用は `data/pairsplit/assignment.jsonl`。

同じ割当から、生成器の SFT 用と評価器の選好用の両方を出す。

```text
make pairsplit-data
```

既定は `--heldout-n 260 --seed 42 --also-pref`。`--also-pref` は続けて `make pref-keep-data` 相当を行う。

出てくるもの:

- SFT（チャット形式）: `data/edit_sft_section/`、`data/edit_sft_hunk_nopara/`、`data/edit_sft_all/` の `train.jsonl` と `heldout.jsonl`
- 選好: `data/pref_keep_split_section/`、`data/pref_keep_split_hunk/` の `train.jsonl` と `valid.jsonl`

選好 JSONL だけを出し直すときは `make pref-keep-data`。各行は下書きを `source_text`、人間の推敲を `candidate_a`、下書きを `candidate_b`、`label` は 1 である。

## A1（段落内の断片）

A の断片 1597 件。段落境界をまたがない推敲のみ。学習用ファイルは `data/revision_corpus/keep_hunk_nopara.jsonl`。

ペア分割後:

- 学習 1387 件: `data/edit_sft_hunk_nopara/train.jsonl`、`data/pref_keep_split_hunk/train.jsonl`
- 検証 210 件: `data/edit_sft_hunk_nopara/heldout.jsonl`、`data/pref_keep_split_hunk/valid.jsonl`

## A2（節）

A の節 379 件。段落境界を含む推敲。学習用ファイルは `data/revision_corpus/keep_section.jsonl`。

ペア分割後:

- 学習 329 件: `data/edit_sft_section/train.jsonl`、`data/pref_keep_split_section/train.jsonl`
- 検証 50 件: `data/edit_sft_section/heldout.jsonl`、`data/pref_keep_split_section/valid.jsonl`

## B（節の三つ組み選好）

A2 の下書き 379 件のそれぞれに、生成器の SFT と同じ指示文で Composer（`composer-2.5`）が推敲を 1 本出す。人間の推敲はプロンプトに見せない。

```text
次の下書きを、意味を保ったまま日本語の技術文書として推敲せよ。
行頭の # で始まる見出しと、【図】・【コード】などの置き場所を示す要素は原稿の構成要素である。推敲の一部として位置を動かすのはよいが、省いてはならない。
文体（ですます調・である調）は下書きのまま保て。変えてはならない。
出力は推敲後の本文のみ。前置き・後書き・変更点の説明・「このようにすると…」のようなメタ文言は書くな。
```

人が下書きと生成だけを見て、劣化していないかを付ける。人間の推敲とは比べない。
劣化していない件だけを、同じ下書きの三つ組みにする。

- 人間の推敲 ＞ 下書き
- 人間の推敲 ＞ 生成
- 生成 ＞ 下書き

A2 の検証 50 件は学習側に入れない。生成と劣化確認は 379 件すべてに対して行った。
人が劣化していないと付けたのは 378 件、劣化していると付けたのは 1 件である。劣化 1 件は学習側だった。学習 328 件、検証 50 件である。

```text
make section-middle-gen
make section-middle-judge
make section-middle-triples
```

成果物:

- 生成と劣化確認: `data/section_middle/revisions.jsonl`、`judgments.jsonl`
- 選好（下書きあたり 3 行）: `data/section_middle/pref_train.jsonl`（984 行）、`pref_valid.jsonl`（150 行）

検証 50 件について、人間の推敲と Composer の推敲の上下を人がブラインドで付けた結果は `data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl` と `judgments_pref_valid_gold_vs_composer.jsonl`。入口は `make pref-valid-blind-pairs` と `make pref-valid-blind-judge`。

## C（LoRA 評価用 60 件）

A1 / A2 の検証側 260 件から、人手ブラインド判定に使う 60 件を選ぶ。節 50 件、断片 10 件。節ペアを優先する。下書きが 100 文字未満のものは除く。乱択のシードは 42。

```text
make select-blind-items
```

成果物は `data/blind_eval/items.jsonl`（下書きと人間の推敲）。

各下書きについて、Qwen に A1 / A2 で学んだ SFT アダプタを載せ、温度付きサンプリングで推敲を 8 本出す。成果物は `outputs/edit-sft-eval-v3/adapter_samples.jsonl`。

人間の推敲と、8 本のうち当時の評価器が最高点を付けた 1 本とのペアを `make build-blind-pairs` で作り、人が優劣を付ける（`make blind-judge`）。

- 対象 60 件の下書きと人間の推敲: `data/blind_eval/items.jsonl`
- アダプタ生成 8 本: `outputs/edit-sft-eval-v3/adapter_samples.jsonl`
- 人間の推敲と選抜 1 本のペア: `data/blind_eval/pairs_gold_vs_adapter_selected.jsonl`
- 人間の判定: `data/blind_eval/judgments_gold_vs_adapter_selected.jsonl`

## D（区間教師 800 行）

A1 の学習側 1387 件から 200 下書きを乱択する（シード 42）。同一下書きについて Qwen3-8B が三条件で推敲を 1 本ずつ出し、人が各生成に位置ラベルを付ける。同じ 200 下書きに対する人間の推敲 200 行を足して 800 行にまとめる。

三条件（各 200 行）:

- アダプタなし。SFT と同じ指示文（`qwen_base`）
- アダプタなし。指示の前に japanese-tech-writing の本文を置く（`qwen_base_norms`）
- A1 で学んだアダプタ。SFT と同じ指示文（`qwen_adapter`）

位置ラベル（`position` 列）:

| 位置 | 意味 |
|------|------|
| a | 劣化 |
| b | 下書きに相当 |
| c | 改善だが人間ほどではない |
| d | 人間の推敲と同程度 |

生成 600 行に位置ラベル付き。人間の推敲 200 行は位置 d。

```text
make a1-probe-items
make build-a1-probe-image
make a1-probe-smoke-check
make a1-probe-check
make a1-probe-position-judge
make pref-d-data
```

中間成果物:

- 対象 200 件の id: `data/a1_probe/ids.jsonl`
- 三群生成: `outputs/a1-probe/base_samples.jsonl`、`base_norms_samples.jsonl`、`adapter_samples.jsonl`
- 位置判定: `outputs/a1-probe/position_judgments.jsonl`（600 行）

D 本体:

- 800 行: `data/d/dataset.jsonl`
- 下書き単位 8:2 分割: `train.jsonl`（640 行）、`valid.jsonl`（160 行）、`valid_item_ids.jsonl`（40 下書き）、`split_stats.json`
- 5 分割（各 640 / 160 行）: `data/d/folds/`
- スキーマ: `data/d.schema.json`
- 分割の乱数 `--seed` 0

位置ラベルの件数（800 行全体）: a 267、b 165、c 118、d 250（うち人間 200 行）。

学習手順（区間損失 f）は [plan-d-interval.md](plan-d-interval.md)。
選好ベクトル GPM は [plan-d-gpm.md](plan-d-gpm.md)。
