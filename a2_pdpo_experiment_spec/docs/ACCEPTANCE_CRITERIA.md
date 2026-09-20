# ACCEPTANCE_CRITERIA.md

実装完了とみなす条件。

1. A2 379件を検証できる。
2. split manifestが303/38/38で固定される。
3. 各A2 draftから3件ずつgeneric revisionを再現可能に生成できる。
4. generationごとにmodel/prompt hash/seed/sampling parametersを保存する。
5. preference datasetにgeneric-vs-generic pairが1件もない。
6. preference datasetにdraft-as-rejectedが1件もない。
7. 各A2 sourceのtraining weight合計が1になる。
8. P-DPOを固定reference policyから再現可能に学習できる。
9. reference log-probを事前計算して再利用できる。
10. 長さリーク診断が実装される。
11. testはfresh generic revisionsで評価する。
12. test generic revisionsはtraining generationとseedが異なる。
13. source単位accuracyとmarginを出力する。
14. marginとlength ratioの相関を出力する。
15. blind A/B fileを生成できる。
16. 全runについてconfig snapshot、seed、git commitを保存する。
17. READMEのコマンドだけで再現できる。
