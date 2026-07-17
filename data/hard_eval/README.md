# Hard Eval — base pack (labeled, v1 採点済み)

Seed: edit-sft held-out 3冊（`what-is-monad`, `computer-arch-revisit`, `ir-system`）から本文段落を抽出。
Base: 同意味・別表層の言い換え（採点の source）。seed 本文は採点に使わない。

| ファイル | 内容 |
|----------|------|
| `bases_v1.jsonl` | 20件。`base_text` 済み、`candidates` は identity のみ |
| `bases_v1_with_human.jsonl` | 上記 + 全20件に `human` 候補をマージ済み |
| `bases_v1_candidates.jsonl` | human + 4モデル候補（各6案）。**ラベル付けの入力** |
| `bases_v1_candidates_preview.md` | 全候補の読み用。各サブセクションを良い順に並べ替える |
| `bases_v1_preview.md` | seed / base 対照の読み用 |
| `bases_v1_for_human_edit.md` | 人間推敲用の元ファイル |
| `base_v1_for_human_edit_edited.md` | 人間推敲の結果 |
| `prompts/CANDIDATES.md` | 候補（human / 多モデル）の生成手順 |

## モデル候補（v1）

同一指示・規範スキルなし。生成は Cursor サブエージェント経由。

| id | generator | 区分 |
|----|-----------|------|
| `model-composer` | composer | フロンティア |
| `model-fable` | fable | フロンティア |
| `model-gpt56` | gpt-5.6 | フロンティア |
| `model-grok` | grok | 癖強め |

Kimi / GMK など安価帯は API 未接続のため今回は未収録（必要なら後から追記可）。

次: `bases_v1_candidates_preview.md` の各項目内で、候補の `###` サブセクションを良い順に並べ替える。
候補見出しと本文は一緒に動かし、本文は編集しない。
`human` を自動最良にせず、`base` が最良なら先頭に置く。

並べ替え後、次のコマンドで先頭候補を `best_id`、全候補の順序を `rank` として JSONL に変換する。

```bash
make hard-eval-label
```

出力は `bases_v1_labeled.jsonl`。

## v1 採点結果（2026-07-17）

| 指標 | pref-bt | pref-ce |
|------|--------:|--------:|
| Top-1 一致 | 0.45 | **0.50** |
| ペア一致率 | 0.830 | **0.837** |

レポート: `outputs/hard_eval_report_{bt,ce}.{json,md}`。
判定と読み（CE 採用）は [docs/ROADMAP.md](../../docs/ROADMAP.md) の段階 2b。

## v2（節単位・構成軸）

held-out 実編集（`Nmonthly` の学習未使用 5 リポジトリ）から24項目を生成済み。
設計は [docs/HARD-EVAL.md](../../docs/HARD-EVAL.md) の「v2」節。

| ファイル | 内容 |
|----------|------|
| `bases_v2_labeled.jsonl` | 24項目 × 5候補。**定義順位でラベル済み**（人手並べ替え不要） |
| `bases_v2.jsonl` | 上記のコピー（互換用） |
| `bases_v2_candidates_preview.md` | 参照用。gold 順で並べてある |

定義順位: `human > deg-join > deg-split > base > deg-reverse`

採点:

```bash
make hard-eval-v2-build   # 定義ラベル付き JSONL を再生成

make hard-eval-score INPUT=data/hard_eval/bases_v2_labeled.jsonl SCORER=ce MODEL=outputs/pref-ce-beyond-para REPORT=outputs/hard_eval_v2_report_ce_beyond_para.json
make hard-eval-score INPUT=data/hard_eval/bases_v2_labeled.jsonl SCORER=ce MODEL=outputs/pref-ce-hunk-only REPORT=outputs/hard_eval_v2_report_ce_hunk_only.json
make hard-eval-score INPUT=data/hard_eval/bases_v2_labeled.jsonl SCORER=bt MODEL=outputs/pref-bt REPORT=outputs/hard_eval_v2_report_bt.json
```

新旧 CE（節ペア込み vs hunk のみ）の差は、deg-* を human より下に置けるかで見る。
