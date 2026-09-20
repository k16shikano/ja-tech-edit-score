# 作業ログ（エージェント管理）

公開 Git に含める。原稿本文、判定の生文、レジストリ名、DOK 起動値、実リポジトリのパスは書かない。

## いま

`BRIEF.md`：評価用データ E を冒頭に追記。§4 `ead-den-a2` 追加。271 行以降（GPM/P-DPO 等）を実験用紙＋結果要約の粒度に整理。

`plan-only-a2.md` §4–§5：E 評価の根本動機、交絡否証。詳細は plan 側。

E 整備済（498 行、`validation.json` ok）。A2 R 学習実行中：`make ead-den-a2-train`（379 節、truncate 0/379）。ログ `outputs/ead/work/ead-den-a2-train.log`。完了後 `ead-r-a2-score-e`（分布 + §5 交絡検証）。

A2 truncate（`edit_sft_section` 379 件、`Qwen3-8B` chat template、`max_seq_length=4096`）: **5/379 = 1.3%**（train 4/329、heldout 1/50）。中央値 995 tok、p99 4214。20% 閾値未満。

ead-ambiguity-probe 完了（A1 1597、GiNZA+Sudachi アラインメント失敗率 0.143）。型間 ρ 最高 T7=0.565、T8=1.0（保持 0・規則要確認）。T4=0.300。レポート `ead-ambiguity-probe.md`。

ead-readability-probe 完了（GiNZA 5.2.1、失敗 0/1976）。A1 mean 939/1597=0.588、A2 mean 184/379=0.485。expand 層 ~0.4–0.5 で stop_expand_fail。random_sign p95 未達。

C 典型性検証（A 1716 分布 → C 56）: s_den vs joint 典型 13/42、s_den vs 人手 33/56。s_den は expand 34/56・edit_distance 大 41/56。レポート `ead-typicality-c.md`。

C 天井: 再判定 `judgments.redo*.jsonl` は 22 行あるが compare5 は 5 行（同一 item 2 件の重複含む）。primary との突合は 2 件、両方一致 2/2。分母 20 未満のため**天井は未測定**。

C 56 件で excl と pref-d-gpm-modernbert-k16（head_dim 32、各 33/56）の正解重なり: 両方 24、excl のみ 9、gpm のみ 9、両方外れ 14。同じ 33 件ではない。

C margin（excl $s_\mathrm{sum}$ 差）の分散分解: プロジェクト 25 群で between 54.6% / within 45.4%（$\eta^2=0.546$）。原稿間が過半だが原稿内も 45%。websocket 5 節は margin 23〜338。

excl margin≥100 かつ不一致 3 件: picoruby（gpm は正）、websocket・typetheory（gpm も外れ）。excl margin と gpm margin の Pearson $r=0.205$。

## 前

R5 確定（B→A）。R5-B 人間勝率 0.970。R5-A 符号一致 33/56。G2 ablation `s_den_surface` RPS 0.239。

## 前

G2 R7 再学習・R11 ablation 完了。

## 前

床 RPS バグ修正。

## 前

Phase 3 G1/G2 実装・初回実行。
