# IMPLEMENTATION_ORDER.md

## Phase 0: A2検証

`inspect_a2.py`

- A2が379件であること
- id重複がないこと
- draft/human_revisionが空でないこと
- 文字数/token数分布
- 完全一致pair
- 極端に長いpair
- 同一draft重複
- 同一human_revision重複

をレポートする。

## Phase 1: split固定

`split_a2.py`

303/38/38へ分割し、manifestを保存する。
manifestが存在する場合は再生成しない。

## Phase 2: generic revision生成

`generate_generic.py`

- promptは `prompts/` から読む
- K=3
- seed固定
- 途中再開可能
- 既存sampleを上書きしない
- prompt hashを保存する
- thinking trace混入を失敗扱いにする
- 空出力、原稿の完全コピー、異常な長短にはstatusを付ける

## Phase 3: P-DPO preference dataset生成

`build_preferences.py`

各A2 itemについて、

    chosen   = human_revision
    rejected = generic_revision

の3pairだけを作る。

禁止:

- generic-vs-generic
- draft-as-rejected
- 順位ラベル追加
- 段落alignment

各sourceの3pairには `source_weight = 1/3` を付ける。

## Phase 4: 長さリーク診断

`diagnose_length_leakage.py`

human/genericの長さ分布、source比、簡単な表面特徴による識別可能性を確認する。
これは学習baselineではない。

## Phase 5: reference log-prob事前計算

`precompute_reference_logps.py`

adapter OFFのbase model `pi_0` でhumanとgenericのlog-probを保存する。

保存:

- total sequence log-prob
- completion token count
- mean token log-prob

## Phase 6: P-DPO学習

`train_pdpo.py`

実施する学習はこれだけ。

- SFTなし
- vanilla DPOなし
- base model frozen
- editor LoRAのみ更新
- source item equal weighting
- beta候補: 0.05, 0.1, 0.2
- devでbeta/checkpointを選ぶ
- testは最後まで見ない

## Phase 7: automatic evaluation

`evaluate_preferences.py`

test 38件についてfresh generic revisionを3件生成する。
学習時とseedを変える。

human vs fresh genericだけを評価する。
generic同士は比較しない。

出力:

- per-source accuracy
- per-source mean margin
- overall accuracy
- mean/median margin
- marginとlength ratioの相関

## Phase 8: blind generation

`generate_blind_ab.py`

各test draftについて、

- adapter OFF
- editor LoRA ON

の生成を1件ずつ作り、A/Bをランダム化する。
人間の判定用ファイルだけを作る。自動で勝敗を付けない。

## Phase 9: representation analysis

Phase 7と8で効果が確認できた場合だけ実施する。

adapter OFF/ONで同じdraftを読み、hidden representation差分を保存する。
この段階より前に「編集者の原稿の見え方を学習した」と主張しない。
