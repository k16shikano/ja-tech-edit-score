# 学習データの作り方

BRIEF の A / A1 / A2 / B / C が、どの手順でできたか。実施済みである。作り直すときの入口もここ。
ディレクトリの置き場は [data/README.md](../data/README.md)。

## 推敲前後ペア（A）

人がレビューして採用した、下書きと人間の推敲の対である。全 1976 件。正本は `data/revision_corpus/canonical.jsonl`。

採掘は Git のブランチ対から行う。短い箇所（空行を含まない）は `make data`、段落境界を含む節は `make mine-sections`。
レビューは `make edit-sft-review`（節）と `make edit-sft-review-hunk`（断片）。採用した対を `make edit-sft-export-keeps` で書き出し、`make edit-sft-promote-reviewed` で正本へ載せる。

節か断片かは、空行（段落境界）の有無で分ける。

- 節: 379 件。`data/revision_corpus/keep_section.jsonl`
- 断片: 1597 件。`data/revision_corpus/keep_hunk_nopara.jsonl`

これが BRIEF の A2 と A1 の元である。

## ペア単位の分割（A1 / A2、生成器と評価器で同じ）

全 1976 件を、書籍に関係なくペア単位で学習と検証に分ける。

以前は書籍単位で分けていた。ペア数の書籍分布に偏りがあり、検証側が特定の書籍の文体と分量に寄る。主張したいのは個々の推敲の質であって、書籍をまたぐ汎化ではないので、書籍単位で分ける必要はない。

乱択のシードは 42。検証側は 260 件。節と断片の比率を保つ層化である。割当の再現用は `data/pairsplit/assignment.jsonl`。旧い書籍単位の id 一覧は `data/pairsplit/booksplit_ids.json` に退避してある。

同じ割当から、生成器の SFT 用と評価器の選好用の両方を出す。分割が食い違うと、評価器が検証ペアを学習済みになる。

```text
make pairsplit-data
```

既定は `--heldout-n 260 --seed 42 --also-pref`。`--also-pref` は続けて `make pref-keep-data` 相当を行う。

出てくるもの:

- SFT（チャット形式）: `data/edit_sft_section/`、`data/edit_sft_hunk_nopara/`、`data/edit_sft_all/` の `train.jsonl` と `heldout.jsonl`
- 選好: `data/pref_keep_split_section/`、`data/pref_keep_split_hunk/` の `train.jsonl` と `valid.jsonl`
- A2 の学習 329 件・検証 50 件は、節の train / heldout である
- A1 の学習 1387 件・検証 210 件は、断片の train / heldout である

選好 JSONL だけを出し直すときは `make pref-keep-data`。各行は下書きを `source_text`、人間の推敲を `candidate_a`、下書きを `candidate_b`、`label` は 1 である。分割は canonical の `meta.split` を維持する。

## 人手判定 60 件（C の対象）

検証側 260 件から、人手のブラインド判定に使う 60 件を選ぶ。節ペアを優先する。下書きが 100 文字未満のものは除く。乱択のシードは 42。

```text
make select-blind-items
```

成果物は `data/blind_eval/items.jsonl`（下書きと人間の推敲）。
生成は BRIEF の評価用データ C。判定ペアは `make build-blind-pairs`、判定は `make blind-judge`。

## スカラー仮説の検査（C から 10 件）

評価用データ C の 60 件から 10 件を乱択する。乱択のシードは 42。
各件の候補は下書き 1 本と、SFT アダプタの 8 本のうち当時の評価器が最高点を付けた 1 本を除いた 2 本である。
総当たりは件あたり 3 対、全体で 30 対である。
問いは、左右のどちらが自分の推敲に近いかである。同等と比較できないを許す。
比較できないは同程度ではない。
比較できない対がある件は、推移的とも循環とも数えない。
人間の推敲本文は出さない。評価器が選んだ文は出さない。
下書きは文脈として出し、候補の一方にもなりうる。
10 件は循環が出るかの初回診断であり、60 件での割合の推定ではない。

```text
make scalar-transitivity-pairs
make scalar-transitivity-judge
make analyze-scalar-transitivity
```

成果物は `data/blind_eval/pairs_scalar_transitivity.jsonl` と `judgments_scalar_transitivity.jsonl`。
手順は `data/blind_eval/scalar_transitivity_protocol.json`。
集計は `data/blind_eval/scalar_transitivity_analysis.json`。
判定 Web はポート 8325、ホストは `0.0.0.0`。

## 教師データ B

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
人が劣化していないと付けたのは 378 件、劣化していると付けたのは 1 件である。劣化 1 件は学習側だった。学習 328 件、検証は A2 の 50 件である。

```text
make section-middle-gen
make section-middle-judge
make section-middle-triples
```

成果物は `data/section_middle/revisions.jsonl`、`judgments.jsonl`、`pref_train.jsonl`、`pref_valid.jsonl`。

## B の検証 50 件のブラインド判定

学習用データ B の検証 50 件について、人間の推敲と Composer の推敲の上下を、左右の役割を隠して人が付ける。下書きは文脈として出す。Composer の本文は B の生成済み 1 本を使い、新たに出さない。評価器が選んだ文は使わない。

```text
make pref-valid-blind-pairs
make pref-valid-blind-judge
```

成果物は `data/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl` と `judgments_pref_valid_gold_vs_composer.jsonl`。手順は `data/blind_eval/valid50_gold_vs_composer_protocol.json`。集計の分母は、人が同等としなかった件。分子は、人の上下とその評価器の点の上下が一致した件数である。

## 独立した人手判定の 40 件

学習 328 件と検証 50 件に使っていない下書きで、評価器が同じかを人が見るための固定一覧である。
出典は A1 の検証側。下書きは 100 文字以上。乱択のシードは 42。

```text
make freeze-8d-items
```

成果物は `data/blind_eval/8d_items.jsonl`。手順は `data/blind_eval/8d_protocol.json`。候補本文の生成は判定の直前に行い、生成後は置き換えない。判定の進め方は [PLAN.md](PLAN.md) の「独立した人手判定」。

## ブラインド判定 360 件からの選好 JSONL

`data/blind_eval/pairs.jsonl` と `judgments.jsonl` と `items.jsonl` を結合する。左右の割り付けを維持し、`preference` は `a` / `b` / `tie` である。欠陥ペアと同等も除外しない。

同じ item の 6 比較は分離しない。シード 42 の決定的 5-fold で、節 50 item を `data/generated_pref_experiment/folds_section/` に書く。各 fold の valid は 10 item、train は 40 item。fold 間で valid item は重ならない。断片 10 item は `folds_hunk/` に同じ分け方で書く。

```text
make generated-pref-data
```
