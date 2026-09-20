# EVALUATION.md

## 主評価

test A2 38件を使う。

各test draftから、学習時とは別seedでfresh generic revisionを3件生成する。

評価pairは常に、

    human_revision vs fresh_generic_revision

だけである。

generic revision同士には評価を付けない。

各pairについて、

    m_ij = s_theta(x_i,h_i) - s_theta(x_i,g_ij)

を計算する。

source単位では、3件のmarginを平均する。

primary metric:

    各A2 itemで human > generic となった割合をsource単位で平均

secondary:

- mean margin
- median margin
- per-source margin
- marginとlength ratioのPearson/Spearman相関

## blind A/B

各test draftから、

- adapter OFFのgeneric revision
- editor LoRA ONのrevision

を1件ずつ生成する。

A/Bの順番をランダム化し、mappingを別ファイルに保存する。

質問文は固定する。

> AとBのうち、あなた自身の推敲判断に近いものを選んでください。どちらとも選べない場合は「選べない」としてください。

この評価でも全順序を要求しない。
