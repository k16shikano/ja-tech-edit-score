# results/

公開してよい集計成果物（候補本文・埋め込み・活性値は含まない）。

生成手順と解釈は `docs/ADDITIONAL-ANALYSIS-2026-07-28.md` を参照。

再生成例:

```bash
python scripts/export_hard_eval_metrics.py --overwrite
python scripts/run_paired_statistics.py --overwrite --n-bootstrap 100000 --seed 0
python scripts/recheck_v1_sentseq_pairwise.py --overwrite
python scripts/analyze_length_confound.py --overwrite
python scripts/export_probe_layer_curve.py --overwrite
python scripts/eval_pref_bt_xproject.py --input data/pref_dataset.jsonl \
  --report results/raw/eval_bt_xproject_6274.json --epochs 80 --lr 1e-2
python scripts/export_pref_bt_lopo_csv.py --overwrite
```
