# scripts/README.md

実装するCLI。

```text
inspect_a2.py
split_a2.py
generate_generic.py
build_preferences.py
diagnose_length_leakage.py
precompute_reference_logps.py
train_pdpo.py
evaluate_preferences.py
generate_blind_ab.py
analyze_representations.py
```

SFT用script、vanilla DPO用scriptは作らない。

全scriptは、

```bash
--config config/experiment.yaml
```

を受け取る。

run directoryへ必ず保存する。

```text
config.snapshot.yaml
environment.txt
git_commit.txt
run_metadata.json
```

## generate_generic.py

```bash
python scripts/generate_generic.py \
  --config config/experiment.yaml \
  --split train
```

llama.cpp serverのOpenAI互換APIを使ってよい。
backendは差し替え可能にする。

promptは次の2ファイルを読む。

```text
prompts/generic_revision_system.txt
prompts/generic_revision_user.txt
```

`{{DRAFT}}` をsource textで置換する。

最低限のvalidation:

- output non-empty
- output != input
- thinking markersなし
- assistant prefixや講評なし
- 異常な長短はflagする

失敗生成も履歴として残し、再生成時は別attemptにする。

## train_pdpo.py

これが唯一の学習script。

同じA2 sourceの3 negativesはloss集計時に各1/3の重みとする。
標準DPO形式のsequence log-probをP-DPO objective内で使う。
mean-token-logpへ勝手に置き換えない。

base model parameterは凍結し、editor LoRAだけを更新する。

## reproducibility

seedを明示する。

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA
- generation seed

非決定性が残る箇所はrun metadataへ記録する。
