# A2 P-DPO experiment

A2 379件の「下書き / 人間推敲」ペアから、一人の編集者に固有の推敲差分を学習する実験の実装仕様。

最重要事項は `AGENTS.md` を参照。

## データの意味

A2の各項目:

```json
{
  "id": "A2-0001",
  "draft": "...",
  "human_revision": "..."
}
```

各draftからローカルLLMで3件のgeneric revisionを生成する。

学習関係は常に:

```text
human_revision > generic_revision
```

generic revision同士の順位は作らない。

## ディレクトリ

```text
.
├── AGENTS.md
├── README.md
├── config/
│   └── experiment.yaml
├── prompts/
│   ├── generic_revision_system.txt
│   └── generic_revision_user.txt
├── schemas/
│   ├── a2.schema.json
│   ├── generated_revision.schema.json
│   └── preference.schema.json
├── docs/
│   ├── DATA_CONTRACT.md
│   ├── EXPERIMENT.md
│   ├── IMPLEMENTATION_ORDER.md
│   ├── EVALUATION.md
│   └── ACCEPTANCE_CRITERIA.md
├── scripts/
│   └── README.md
├── data/
│   ├── A2/
│   ├── generated/
│   └── preferences/
├── outputs/
└── checkpoints/
```

このディレクトリは「実装エージェントへの仕様書」であり、A2本体は含まない。

## 実行手順

親リポジトリの `data/revision_corpus/keep_section.jsonl` から A2 を取り込む。作業ディレクトリは `a2_pdpo_experiment_spec/`。

```bash
cd a2_pdpo_experiment_spec
pip install -r requirements.txt   # GPU 学習・logprob 用

make a2-inspect                   # Phase 0: 379件検証 + items.jsonl 生成
make a2-split                     # Phase 1: 303/38/38 manifest 固定
make a2-generate-train            # Phase 2: train generic x3（llama.cpp 要）
make a2-generate-dev              # Phase 2: dev generic x3
make a2-preferences               # Phase 3: preference dataset
make a2-length-diag               # Phase 4: 長さリーク診断
make a2-precompute-ref            # Phase 5: reference logprob（GPU 要）
make a2-train-pdpo                # Phase 6: P-DPO 学習（GPU 要）
make a2-evaluate                  # Phase 7: test 評価（llama.cpp + GPU）
make a2-blind-ab                  # Phase 8: blind A/B 生成（GPU 要）
```

generic 生成は `config/experiment.yaml` の `generic_revision.generator.endpoint`（既定 `http://127.0.0.1:8080`）へ llama.cpp の OpenAI 互換 API が必要。
