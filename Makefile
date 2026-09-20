ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PATH := /usr/local/bin:/usr/bin:/bin:$(PATH)
GIT ?= /usr/bin/git
export GIT
PYTHON := $(ROOT).venv/bin/python3
ifeq (,$(wildcard $(PYTHON)))
  PYTHON := python3
endif

DATA_DIR := $(ROOT)data
OUTPUT_DIR := $(ROOT)outputs/pref-static
BT_OUTPUT_DIR := $(ROOT)outputs/pref-bt
BT_KEEP_OUTPUT_DIR := $(ROOT)outputs/pref-bt-keep
BT_KEEP_PAIRSPLIT_DIR := $(ROOT)outputs/pref-bt-keep-pairsplit
RAW := $(DATA_DIR)/examples.raw.jsonl
PREF_KEEP_SPLIT_HUNK := $(DATA_DIR)/pref_keep_split_hunk
PREF_KEEP_SPLIT_SECTION := $(DATA_DIR)/pref_keep_split_section
SENTSEQ_KEEP_OUTPUT_DIR := $(ROOT)outputs/pref-sentseq-keep
SENTSEQ_KEEP_PAIRSPLIT_DIR := $(ROOT)outputs/pref-sentseq-keep-pairsplit
SENTSEQ_MIDDLE_DIR := $(ROOT)outputs/pref-sentseq-section-triples
SETWISE_MIDDLE_DIR := $(ROOT)outputs/pref-setwise-section-triples
SETWISE_HUMAN_TOP_DIR := $(ROOT)outputs/pref-setwise-section-human-top
NCE_MIDDLE_DIR := $(ROOT)outputs/pref-nce-section
PREF_INTERVAL_A1_PROBE_DIR := $(ROOT)outputs/pref-interval-a1-probe
A1_PROBE_INTERVAL_DATA := $(DATA_DIR)/a1_probe_interval
PREF_D_DATA := $(DATA_DIR)/d
PREF_D_MODERNBERT_CV := $(ROOT)outputs/pref-d-modernbert-cv
PREF_D_BT_MODERNBERT := $(ROOT)outputs/pref-d-bt-modernbert
PREF_D_GPM_MODERNBERT := $(ROOT)outputs/pref-d-gpm-modernbert
PREF_D_BT_QWEN := $(ROOT)outputs/pref-d-bt-qwen3-8b
PREF_D_GPM_QWEN := $(ROOT)outputs/pref-d-gpm-qwen3-8b
QWEN3_8B_MODEL ?= Qwen/Qwen3-8B
PREF_D_PAIR_LEDGER := $(PREF_D_DATA)/pair_ledger.jsonl
GPM_HEAD_DIM ?= 64
PREF_D_RURI_CV := $(ROOT)outputs/pref-d-ruri-cv
DETACHED_JOBS_DIR := $(ROOT)outputs/.jobs
RUN_DETACHED := $(ROOT)scripts/run_detached_job.sh
export PYTORCH_ALLOC_CONF ?= expandable_segments:True
PREF_A_SPLIT := $(DATA_DIR)/pref_a_split
PREF_D_MODERNBERT_BEST := $(PREF_D_MODERNBERT_CV)/fold0/best
BT_A_RURI_DIR := $(ROOT)outputs/pref-bt-a-ruri
BT_A_MODERNBERT_DIR := $(ROOT)outputs/pref-bt-a-modernbert
BT_A_MODERNBERT_D_DIR := $(ROOT)outputs/pref-bt-a-modernbert-d
SENTSEQ_A_RURI_DIR := $(ROOT)outputs/pref-sentseq-a-ruri
SENTSEQ_A_MODERNBERT_DIR := $(ROOT)outputs/pref-sentseq-a-modernbert
SENTSEQ_A_MODERNBERT_D_DIR := $(ROOT)outputs/pref-sentseq-a-modernbert-d
SENTSEQ_A2_MODERNBERT_D_DIR := $(ROOT)outputs/pref-sentseq-a2-modernbert-d
SENTSEQ_A2B_MODERNBERT_D_DIR := $(ROOT)outputs/pref-sentseq-a2b-modernbert-d
BT_A_OUTPUT_DIR := $(BT_A_RURI_DIR)
SENTSEQ_A_OUTPUT_DIR := $(SENTSEQ_A_RURI_DIR)
MODERNBERT_MODEL ?= sbintuitions/modernbert-ja-310m
PREF_D_BATCH_SIZE ?= 8
DETECT_MIDDLE_DIR := $(ROOT)outputs/pref-detect-section
DETECT_CD_MIDDLE_DIR := $(ROOT)outputs/pref-detect-cd-section
B_GENERATION_DATA := $(DATA_DIR)/b_generation
B_GENERATION_OUT := $(ROOT)outputs/b_generation
D_EPOCH7_OUT := $(ROOT)outputs/d_epoch7
B_GENERATION_CONFIG := $(ROOT)configs/b_generation/common.yaml

EMBED_MODEL ?= cl-nagoya/ruri-v3-30m
TRUNCATE_DIM ?= 0
# Ruri 文書プレフィックスは末尾空白が必要
null :=
space := $(null) #
TEXT_PREFIX ?= 文章:$(space)
MAX_SEQ_LENGTH ?= 512
BATCH_SIZE ?= 32

SENTSEQ_BATCH_SIZE ?= 64
MULTIGRANULAR_STAGES ?= 1,2,3,4,5,6,7
MULTIGRANULAR_SEEDS ?= 0

.PHONY: help venv data mine-sections pairsplit-data pref-keep-data pref-a-data pref-d-data pref-d-modernbert-train pref-d-modernbert-cv pref-d-ruri-cv pref-d-pair-ledger pref-d-bt-modernbert pref-d-bt-modernbert-fg pref-d-bt-modernbert-status pref-d-bt-modernbert-stop pref-d-gpm-modernbert pref-d-gpm-modernbert-fg pref-d-gpm-modernbert-status pref-d-gpm-modernbert-stop eval-pref-d-bt-modernbert eval-pref-d-gpm-modernbert eval-pref-c-d-bt-modernbert eval-pref-c-d-gpm-modernbert eval-pref-c-d-pair-all eval-pref-hard-d-bt-modernbert eval-pref-hard-d-gpm-modernbert eval-pref-hard-d-pair-all pref-d-bt-qwen3-8b pref-d-bt-qwen3-8b-fg pref-d-bt-qwen3-8b-status pref-d-bt-qwen3-8b-stop pref-d-gpm-qwen3-8b pref-d-gpm-qwen3-8b-fg pref-d-gpm-qwen3-8b-status pref-d-gpm-qwen3-8b-stop pref-d-qwen3-8b-pipeline pref-d-qwen3-8b-pipeline-status pref-d-qwen3-8b-pipeline-stop eval-pref-d-bt-qwen3-8b eval-pref-d-gpm-qwen3-8b train-bt-a train-bt-a-ruri train-bt-a-modernbert train-bt-a-modernbert-d train-sentseq-a train-sentseq-a-ruri train-sentseq-a-modernbert train-sentseq-a-modernbert-d train-sentseq-a2-modernbert-d train-sentseq-a2b-modernbert-d eval-pref-c-a2b-modernbert-d pref-d-a2b-c train-pref-a-all eval-pref-a-compare eval-pref-a-bc eval-pref-a-hard-eval eval-sentseq-a-modernbert-d-calibrated train-bt-keep train-bt-keep-pairsplit train-sentseq-keep train-sentseq-keep-pairsplit build-pref-keep-image build-generated-pref-sentseq-image build-section-middle-sentseq-image build-setwise-section-triples-image build-setwise-section-human-top-image build-section-middle-nce-image section-middle-nce build-section-middle-detect-image section-middle-detect build-section-middle-detect-cd-image section-middle-detect-cd build-pref-multigranular-image section-middle-setwise section-middle-setwise-human-top hard-eval-setwise pref-multigranular-smoke pref-multigranular freeze-8d-items a1-probe-items build-a1-probe-image a1-probe-smoke-check a1-probe-check a1-probe-position-judge pref-interval-a1-probe eval-pref-interval-a1-probe build-serve-image select-blind-items build-blind-pairs blind-judge pref-valid-blind-pairs pref-valid-blind-judge scalar-transitivity-pairs scalar-transitivity-judge analyze-scalar-transitivity b-rejudge-pairs b-rejudge-judge analyze-b-rejudge analyze-blind-judgments generated-pref-data generated-pref-sentseq-smoke generated-pref-sentseq-cv section-middle-gen section-middle-judge section-middle-triples section-middle-sentseq edit-sft-data edit-sft-review edit-sft-review-hunk edit-sft-review-section-extra edit-sft-export-keeps edit-sft-promote-reviewed edit-sft edit-sft-section-only edit-sft-hunk score-bt rank converge check calibrate-margins revise serve install-bin install-skills daemon daemon-stop test clean-model

help:
	@echo "現行（BRIEF.md）:"
	@echo "  make venv"
	@echo "  make section-middle-nce                   # InfoNCE 手元スモーク（DEVICE=cpu EPOCHS=1 等）"
	@echo "  make build-section-middle-nce-image         # InfoNCE DOK"
	@echo "  make section-middle-detect               # 人間検出 手元スモーク（DEVICE=cpu EPOCHS=1 等）"
	@echo "  make build-section-middle-detect-image     # 人間検出 DOK"
	@echo "  make section-middle-detect-cd            # 検出+Composer対下書き 手元スモーク"
	@echo "  make build-section-middle-detect-cd-image  # 検出+Composer対下書き DOK"
	@echo "  make freeze-8d-items                      # 独立人手判定の下書き 40 件を固定"
	@echo "  make pref-valid-blind-pairs               # B検証50 人間vs Composer 対"
	@echo "  make pref-valid-blind-judge               # B検証50 ブラインド判定 Web"
	@echo "  make scalar-transitivity-pairs            # Cから10件、下書き+未選抜2本の総当たり"
	@echo "  make scalar-transitivity-judge            # スカラー仮説のブラインド判定 Web"
	@echo "  make analyze-scalar-transitivity          # 件ごとの推移性を集計"
	@echo "  make b-rejudge-pairs                      # B 学習側 329 件の総当たり 3 対"
	@echo "  make b-rejudge-judge                      # B 付け直しブラインド判定 Web"
	@echo "  make analyze-b-rejudge                    # B 付け直し choice 内訳"
	@echo "  make a1-probe-items                       # 生成用に japanese-tech-writing を置く"
	@echo "  make build-a1-probe-image                 # A1 三群生成の DOK イメージを build して push"
	@echo "  make a1-probe-smoke-check                 # スモーク生成文を表示して確認する"
	@echo "  make a1-probe-check                       # 本番三群の件数・同一id・指示漏れを確認する"
	@echo "  make a1-probe-position-judge              # 生成 y に人間の推敲への近さの位置を付ける"
	@echo "  make pref-d-data                          # 学習用データ D（800 行）を組み立てる"
	@echo "  make pref-d-modernbert-cv               # D + ModernBERT 5 分割 CV"
	@echo "  make pref-d-ruri-cv                     # D + ruri 凍結 5 分割 CV"
	@echo "  make pref-d-pair-ledger                 # D ペア台帳"
	@echo "  make pref-d-bt-modernbert             # D ペア BT 5 分割 CV（SSH 切断後も継続）"
	@echo "  make pref-d-gpm-modernbert            # D ペア GPM（head_dim=64）5 分割 CV（同上）"
	@echo "  make pref-d-bt-qwen3-8b               # Qwen3-8B D ペア BT 5 分割 CV（同上）"
	@echo "  make pref-d-gpm-qwen3-8b               # Qwen3-8B D ペア GPM 5 分割 CV（同上）"
	@echo "  make pref-d-qwen3-8b-pipeline          # Qwen BT → GPM を同一ジョブで順実行（同上）"
	@echo "  make pref-d-bt-qwen3-8b-status         # 上記ジョブの稼働確認（*-status / *-stop）"
	@echo "  make pref-interval-a1-probe               # 600 対区間損失 f(x,y) を学習（SentSeq、GPU）"
	@echo "  make eval-pref-interval-a1-probe          # 区間所属・H1 を検証"
	@echo "  make test"
	@echo ""
	@echo "実施済みの入口:"
	@echo "  make data / mine-sections    # ブランチ対・節ペアの採掘"
	@echo "  make pairsplit-data          # ペア単位分割"
	@echo "  make pref-keep-data          # 節/断片の選好 JSONL"
	@echo "  make edit-sft MODEL=<hf-id>  # 推敲モデルの SFT"
	@echo "  make select-blind-items / build-blind-pairs / blind-judge / analyze-blind-judgments"
	@echo "  make section-middle-gen / section-middle-judge / section-middle-triples  # 教師データ B"
	@echo "  make train-sentseq-keep-pairsplit / train-bt-keep-pairsplit"
	@echo "  make build-section-middle-sentseq-image / build-setwise-section-triples-image / build-setwise-section-human-top-image"
	@echo "  make section-middle-setwise / section-middle-setwise-human-top"
	@echo "  make pref-multigranular-smoke / build-pref-multigranular-image"
	@echo ""
	@echo "公開 Web / 順位付け:"
	@echo "  make serve / revise / rank / check"
	@echo ""
	@echo "探索用の旧スクリプトは scripts-old/（make からは外した）。対応表は scripts/README.md"

venv:
	python3.12 -m venv $(ROOT).venv 2>/dev/null || python3 -m venv $(ROOT).venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r $(ROOT)requirements.txt

install-bin:
	@mkdir -p $(HOME)/.local/bin
	@ln -sf $(ROOT)bin/ja-tech-edit-score-check $(HOME)/.local/bin/ja-tech-edit-score-check
	@ln -sf $(ROOT)bin/ja-tech-edit-score-compare $(HOME)/.local/bin/ja-tech-edit-score-compare
	@ln -sf $(ROOT)bin/ja-tech-edit-score-rank $(HOME)/.local/bin/ja-tech-edit-score-rank
	@ln -sf $(ROOT)bin/ja-tech-edit-score-converge $(HOME)/.local/bin/ja-tech-edit-score-converge
	@echo "installed: ~/.local/bin/ja-tech-edit-score-check"
	@echo "installed: ~/.local/bin/ja-tech-edit-score-compare"
	@echo "installed: ~/.local/bin/ja-tech-edit-score-rank"
	@echo "installed: ~/.local/bin/ja-tech-edit-score-converge"

install-skills:
	@mkdir -p $(HOME)/.cursor/skills
	@rm -rf $(HOME)/.cursor/skills/ja-tech-edit-score-check
	@ln -sfn $(ROOT)skills/ja-tech-edit-score-check $(HOME)/.cursor/skills/ja-tech-edit-score-check
	@echo "installed: ~/.cursor/skills/ja-tech-edit-score-check -> $(ROOT)skills/ja-tech-edit-score-check"

# ORG は tip main ではなく fork（merge-base）。祖先でない対は miner が拒否する。
# 一括は scripts/batch_mine_hunks_premerge.sh（resolve_pre_merge_pair）を使う。
data:
	@test -n "$(DIR)" || (echo "DIR is required" && exit 1)
	@test -n "$(ORG)" || (echo "ORG is required (must be fork / ancestor of EDT)" && exit 1)
	@test -n "$(EDT)" || (echo "EDT is required" && exit 1)
	$(PYTHON) -c "from pathlib import Path; import sys; sys.path.insert(0,'scripts'); from git_pre_merge import assert_structural_edit_pair; assert_structural_edit_pair(Path('$(DIR)'), '$(ORG)', '$(EDT)')"
	$(PYTHON) scripts/mine_branch_pair.py \
	  --repo "$(DIR)" \
	  --base "$(ORG)" \
	  --edit "$(EDT)" \
	  $(if $(PROJECT_ID),--project-id "$(PROJECT_ID)",) \
	  $(if $(PATH),--path "$(PATH)",) \
	  --append "$(RAW)"

mine-sections:
	bash scripts/batch_mine_sections.sh

build-pref-keep-image:
	bash scripts/build_push_pref_keep_image.sh

build-generated-pref-sentseq-image:
	bash scripts/build_push_generated_pref_sentseq_image.sh

build-serve-image:
	docker build -f Dockerfile.serve -t ja-tech-edit-score:serve .

pref-keep-data:
	$(PYTHON) scripts/build_pref_from_keep.py \
	  --input "$(or $(KEEP_INPUT),$(DATA_DIR)/revision_corpus/canonical.jsonl)" \
	  --out-dataset "$(DATA_DIR)/pref_keep/dataset_hunk.jsonl" \
	  --out-split-dir "$(PREF_KEEP_SPLIT_HUNK)" \
	  --report "$(DATA_DIR)/pref_keep/build_report_hunk.json" \
	  --units hunk
	$(PYTHON) scripts/build_pref_from_keep.py \
	  --input "$(or $(KEEP_INPUT),$(DATA_DIR)/revision_corpus/canonical.jsonl)" \
	  --out-dataset "$(DATA_DIR)/pref_keep/dataset_section.jsonl" \
	  --out-split-dir "$(PREF_KEEP_SPLIT_SECTION)" \
	  --report "$(DATA_DIR)/pref_keep/build_report_section.json" \
	  --units section

train-bt-keep: pref-keep-data
	@test -s "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" || (echo "missing $(PREF_KEEP_SPLIT_HUNK)/train.jsonl" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" \
	  --eval-file "$(PREF_KEEP_SPLIT_HUNK)/valid.jsonl" \
	  --output-dir "$(BT_KEEP_OUTPUT_DIR)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)"

# ペア単位分割の学習側で BT を学び直す（旧 pref-bt-keep は上書きしない）
train-bt-keep-pairsplit:
	@test -s "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" || (echo "run make pairsplit-data first" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" \
	  --eval-file "$(PREF_KEEP_SPLIT_HUNK)/valid.jsonl" \
	  --output-dir "$(BT_KEEP_PAIRSPLIT_DIR)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)"

train-sentseq-keep: pref-keep-data
	@test -s "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" || (echo "missing $(PREF_KEEP_SPLIT_SECTION)/train.jsonl" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" \
	  --eval-file "$(PREF_KEEP_SPLIT_SECTION)/valid.jsonl" \
	  --output-dir "$(SENTSEQ_KEEP_OUTPUT_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

train-sentseq-keep-pairsplit:
	@test -s "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" || (echo "run make pairsplit-data first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" \
	  --eval-file "$(PREF_KEEP_SPLIT_SECTION)/valid.jsonl" \
	  --output-dir "$(SENTSEQ_KEEP_PAIRSPLIT_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

score-bt:
	@test -n "$(SOURCE)" || (echo "SOURCE is required" && exit 1)
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required" && exit 1)
	$(PYTHON) scripts/score_pref_bt.py \
	  --model "$(BT_GATE_DIR)" \
	  --source-text "$(SOURCE)" \
	  --candidate-text "$(CANDIDATE)"

# 二軸運用の既定: 主モデル=pref-sentseq-section-triples（教師データ B）、ゲート=pref-bt-keep（hunk）
# GATE_MODEL= （空）でゲートなしの一軸に戻せる
SENTSEQ_BEST_DIR := $(ROOT)outputs/pref-sentseq-section-triples
BT_GATE_DIR := $(ROOT)outputs/pref-bt-keep
RANK_MODEL ?= $(if $(wildcard $(SENTSEQ_BEST_DIR)),$(SENTSEQ_BEST_DIR),$(BT_GATE_DIR))
GATE_MODEL ?= $(if $(and $(findstring pref-sentseq,$(RANK_MODEL)),$(wildcard $(BT_GATE_DIR))),$(BT_GATE_DIR),)

# 生成→二軸判定→反復の推敲ループを1コマンドで回す（要 CURSOR_API_KEY）
revise:
	@test -n "$(FILE)" || (echo "FILE=下書き.md is required" && exit 1)
	$(PYTHON) scripts/revise_loop.py \
	  --file "$(FILE)" \
	  $(if $(OUT),--out "$(OUT)",) \
	  $(if $(N),--n-candidates $(N),) \
	  $(if $(MAX_ITERS),--max-iters $(MAX_ITERS),) \
	  $(if $(MIN_MARGIN),--min-margin $(MIN_MARGIN),) \
	  $(if $(GATE_MIN_MARGIN),--gate-min-margin $(GATE_MIN_MARGIN),) \
	  $(if $(GEN_MODEL),--model "$(GEN_MODEL)",) \
	  $(if $(ONLY_SECTIONS),--only-sections "$(ONLY_SECTIONS)",) \
	  $(if $(MIN_SECTION_CHARS),--min-section-chars $(MIN_SECTION_CHARS),) \
	  $(if $(SKILLS),--skills "$(SKILLS)",) \
	  --primary-model "$(SENTSEQ_BEST_DIR)" \
	  --gate-model "$(BT_GATE_DIR)" \
	  || { s=$$?; test $$s -eq 2 && echo "（閾値未達の節あり。詳細は上の一覧）" || exit $$s; }

# 下書きと推敲に点数を付ける Web サービス（既定 0.0.0.0:8300）
# 127.0.0.1 では起動しない。公開時は RATE_LIMIT_PER_IP（既定 10）と
# RATE_LIMIT_WINDOW_SECONDS（既定 86400）で IP 単位の試行を制限。
# 無制限にするときは RATE_LIMIT_PER_IP=0。
serve:
	$(PYTHON) scripts/score_server.py --host $(or $(HOST),0.0.0.0) --port $(or $(PORT),8300)

# 人間編集が下書きから稼ぐマージン分布を測り、合格閾値の根拠を出す
calibrate-margins:
	$(PYTHON) scripts/calibrate_acceptance_margins.py \
	  --pairs "$(or $(PAIRS),$(ROOT)data/pref_keep_split_section/valid.jsonl)" \
	  --primary-model "$(SENTSEQ_BEST_DIR)" \
	  --gate-model "$(BT_GATE_DIR)" \
	  --out "$(ROOT)outputs/acceptance_margin_calibration.json"

rank:
	@test -n "$(SOURCE)$(SOURCE_FILE)" || (echo "SOURCE is required (text or use SOURCE_FILE=)" && exit 1)
	@test -n "$(CANDIDATE_FILES)$(CANDIDATES_DIR)" || (echo "CANDIDATE_FILES or CANDIDATES_DIR is required" && exit 1)
	$(PYTHON) scripts/rank_pref_bt.py \
	  --model "$(RANK_MODEL)" \
	  $(if $(SOURCE_FILE),--source-file "$(SOURCE_FILE)",--source-text "$(SOURCE)") \
	  $(foreach f,$(CANDIDATE_FILES),--candidate-file "$(f)") \
	  $(if $(CANDIDATES_DIR),--candidates-dir "$(CANDIDATES_DIR)",) \
	  $(if $(MIN_MARGIN),--min-margin $(MIN_MARGIN),) \
	  $(if $(GATE_MODEL),--gate-model "$(GATE_MODEL)",) \
	  $(if $(GATE_MIN_MARGIN),--gate-min-margin $(GATE_MIN_MARGIN),) \
	  --format $(or $(FORMAT),text)

converge:
	@test -n "$(CURRENT)$(CURRENT_FILE)" || (echo "CURRENT or CURRENT_FILE is required" && exit 1)
	@test -n "$(REVISED)$(REVISED_FILE)" || (echo "REVISED or REVISED_FILE is required" && exit 1)
	$(PYTHON) scripts/check_convergence.py \
	  --mode $(or $(MODE),pair) \
	  --static-model "$(OUTPUT_DIR)" \
	  --bt-model "$(or $(BT_MODEL),$(BT_OUTPUT_DIR))" \
	  $(if $(CURRENT_FILE),--current-file "$(CURRENT_FILE)",--current-text "$(CURRENT)") \
	  $(if $(REVISED_FILE),--revised-file "$(REVISED_FILE)",--revised-text "$(REVISED)") \
	  $(if $(SOURCE_FILE),--source-file "$(SOURCE_FILE)",$(if $(SOURCE),--source-text "$(SOURCE)",)) \
	  --format $(or $(FORMAT),text)

check:
	@test -n "$(FILE)" || (echo "FILE is required" && exit 1)
	$(ROOT)bin/ja-tech-edit-score-check "$(FILE)" \
	  $(if $(BASE),--base "$(BASE)",) \
	  $(if $(EDIT),--edit "$(EDIT)",) \
	  $(if $(FORMAT),--format "$(FORMAT)",--format markdown)

edit-sft-data:
	@test -s "$(DATA_DIR)/examples.section.raw.jsonl" || (echo "run make mine-sections first" && exit 1)
	$(PYTHON) scripts/export_edit_sft.py \
	  --section-raw "$(DATA_DIR)/examples.section.raw.jsonl" \
	  --out-dir "$(DATA_DIR)/edit_sft"

edit-sft-review:
	@test -s "$(DATA_DIR)/edit_sft/train.jsonl" || (echo "run make edit-sft-data first" && exit 1)
	$(PYTHON) scripts/edit_sft_review_server.py \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8310) \
	  --train "$(DATA_DIR)/edit_sft/train.jsonl" \
	  --heldout "$(DATA_DIR)/edit_sft/heldout.jsonl" \
	  --state "$(DATA_DIR)/edit_sft/review_state.json"

# hunk 増分（data/edit_sft_hunk_review）。既存節 keep とは state 分離。
# pending は keep にしない（学習は明示 keep のみ）。
edit-sft-review-hunk:
	@test -s "$(DATA_DIR)/edit_sft_hunk_review/train.jsonl" || (echo "missing edit_sft_hunk_review; export hunk_trainready first" && exit 1)
	$(PYTHON) scripts/edit_sft_review_server.py \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8311) \
	  --train "$(DATA_DIR)/edit_sft_hunk_review/train.jsonl" \
	  --heldout "$(DATA_DIR)/edit_sft_hunk_review/heldout.jsonl" \
	  --state "$(DATA_DIR)/edit_sft_hunk_review/review_state.json"

# 節追加候補（data/edit_sft_section_extra_review）。keep 200 とは別キュー。
edit-sft-review-section-extra:
	@test -s "$(DATA_DIR)/edit_sft_section_extra_review/train.jsonl" || (echo "missing edit_sft_section_extra_review" && exit 1)
	$(PYTHON) scripts/edit_sft_review_server.py \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8312) \
	  --train "$(DATA_DIR)/edit_sft_section_extra_review/train.jsonl" \
	  --heldout "$(DATA_DIR)/edit_sft_section_extra_review/heldout.jsonl" \
	  --state "$(DATA_DIR)/edit_sft_section_extra_review/review_state.json"

# ペア単位の層化乱択で学習/検証を作り直し、pref-keep も同じ分割で再生成する。
pairsplit-data:
	$(PYTHON) scripts/rebuild_pairsplit_data.py \
	  --canonical "$(DATA_DIR)/revision_corpus/canonical.jsonl" \
	  --pairsplit-dir "$(DATA_DIR)/pairsplit" \
	  --heldout-n $(or $(HELDOUT_N),260) \
	  --seed $(or $(SEED),42) \
	  --also-pref

select-blind-items:
	$(PYTHON) scripts/select_blind_items.py \
	  --heldout "$(DATA_DIR)/edit_sft_all/heldout.jsonl" \
	  --out "$(DATA_DIR)/blind_eval/items.jsonl" \
	  --n $(or $(N),60) \
	  --min-chars $(or $(MIN_CHARS),100) \
	  --seed $(or $(SEED),42)

a1-probe-items:
	$(PYTHON) scripts/stage_japanese_tech_writing.py \
	  --out "$(DATA_DIR)/a1_probe/japanese-tech-writing.md"

build-a1-probe-image:
	bash scripts/build_push_a1_probe_image.sh

a1-probe-smoke-check:
	$(PYTHON) scripts/check_a1_probe_smoke.py \
	  --dir "$(or $(SMOKE_DIR),$(ROOT)outputs/a1-probe-smoke)"

a1-probe-check:
	$(PYTHON) scripts/check_a1_probe_smoke.py \
	  --dir "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)" \
	  --dump-n $(or $(DUMP_N),2) \
	  --expect-n $(or $(EXPECT_N),1384)

a1-probe-position-judge:
	$(PYTHON) scripts/a1_probe_position_server.py \
	  --dir "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)" \
	  $(if $(filter 1,$(ALL)),,--ids "$(or $(IDS),$(DATA_DIR)/a1_probe/ids.jsonl)") \
	  --judgments "$(or $(JUDGMENTS),$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)/position_judgments.jsonl)" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8327)

pref-a-data:
	@test -s "$(DATA_DIR)/revision_corpus/canonical.jsonl" || (echo "missing canonical.jsonl" && exit 1)
	$(PYTHON) scripts/build_pref_from_keep.py \
	  --input "$(DATA_DIR)/revision_corpus/canonical.jsonl" \
	  --out-dataset "$(DATA_DIR)/pref_a/dataset.jsonl" \
	  --out-split-dir "$(PREF_A_SPLIT)" \
	  --report "$(DATA_DIR)/pref_a/build_report.json"

train-bt-a: train-bt-a-ruri

train-bt-a-ruri: pref-a-data
	@test -s "$(PREF_A_SPLIT)/train.jsonl" || (echo "run make pref-a-data first" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --embed-backend sentence-transformers \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(BT_A_RURI_DIR))" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --epochs $(or $(EPOCHS),80) \
	  --lr $(or $(BT_LR),1e-2) \
	  --device "$(or $(DEVICE),cuda)"

train-bt-a-modernbert: pref-a-data
	@test -s "$(PREF_A_SPLIT)/train.jsonl" || (echo "run make pref-a-data first" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(BT_A_MODERNBERT_DIR))" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(EPOCHS),80) \
	  --lr $(or $(BT_LR),1e-2) \
	  --device "$(or $(DEVICE),cuda)"

train-bt-a-modernbert-d: pref-a-data
	@test -s "$(PREF_D_MODERNBERT_BEST)/model.safetensors" || (echo "run make pref-d-modernbert-train first" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --modernbert-checkpoint "$(PREF_D_MODERNBERT_BEST)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(BT_A_MODERNBERT_D_DIR))" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(EPOCHS),80) \
	  --lr $(or $(BT_LR),1e-2) \
	  --device "$(or $(DEVICE),cuda)"

train-sentseq-a: train-sentseq-a-ruri

train-sentseq-a-ruri: pref-a-data
	@test -s "$(PREF_A_SPLIT)/train.jsonl" || (echo "run make pref-a-data first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --embed-backend sentence-transformers \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(SENTSEQ_A_RURI_DIR))" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

train-sentseq-a-modernbert: pref-a-data
	@test -s "$(PREF_A_SPLIT)/train.jsonl" || (echo "run make pref-a-data first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(SENTSEQ_A_MODERNBERT_DIR))" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --encode-batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

train-sentseq-a-modernbert-d: pref-a-data
	@test -s "$(PREF_D_MODERNBERT_BEST)/model.safetensors" || (echo "run make pref-d-modernbert-train first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --modernbert-checkpoint "$(PREF_D_MODERNBERT_BEST)" \
	  --train-file "$(PREF_A_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(SENTSEQ_A_MODERNBERT_D_DIR))" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --encode-batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

train-sentseq-a2-modernbert-d: pref-d-modernbert-train pref-keep-data
	@test -s "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" || (echo "run make pref-keep-data first" && exit 1)
	@test -s "$(PREF_D_MODERNBERT_BEST)/model.safetensors" || (echo "run make pref-d-modernbert-train first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --modernbert-checkpoint "$(PREF_D_MODERNBERT_BEST)" \
	  --train-file "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" \
	  --eval-file "$(PREF_KEEP_SPLIT_SECTION)/valid.jsonl" \
	  --output-dir "$(or $(OUT),$(SENTSEQ_A2_MODERNBERT_D_DIR))" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --encode-batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(A2_EPOCHS),10) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

train-sentseq-a2b-modernbert-d: train-sentseq-a2-modernbert-d
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SENTSEQ_A2_MODERNBERT_D_DIR)/model.pt" || (echo "run make train-sentseq-a2-modernbert-d first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --embed-backend modernbert \
	  --model "$(MODERNBERT_MODEL)" \
	  --modernbert-checkpoint "$(PREF_D_MODERNBERT_BEST)" \
	  --init-from "$(SENTSEQ_A2_MODERNBERT_D_DIR)/model.pt" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(or $(OUT),$(SENTSEQ_A2B_MODERNBERT_D_DIR))" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --encode-batch-size $(or $(MODERNBERT_ENCODE_BATCH),8) \
	  --epochs $(or $(EPOCHS),$(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

eval-pref-c-a2b-modernbert-d:
	@test -s "$(SENTSEQ_A2B_MODERNBERT_D_DIR)/model.pt" || (echo "run make train-sentseq-a2b-modernbert-d first" && exit 1)
	@test -s "$(DATA_DIR)/blind_eval/judgments_gold_vs_adapter_selected.jsonl" || (echo "missing C judgments" && exit 1)
	$(PYTHON) scripts/eval_pref_c_human_agreement.py \
	  --model-dir "$(or $(MODEL),$(SENTSEQ_A2B_MODERNBERT_D_DIR))" \
	  --kind sentseq \
	  --out "$(or $(OUT),$(ROOT)outputs/pref-c-eval-a2b-modernbert-d.json)" \
	  --device "$(or $(DEVICE),cuda)"

pref-d-a2b-c: pref-d-data pref-d-modernbert-train train-sentseq-a2b-modernbert-d eval-pref-c-a2b-modernbert-d

train-pref-a-all: train-bt-a-ruri train-bt-a-modernbert train-bt-a-modernbert-d train-sentseq-a-ruri train-sentseq-a-modernbert train-sentseq-a-modernbert-d eval-pref-a-compare

eval-pref-a-compare:
	$(PYTHON) scripts/eval_pref_a_compare.py \
	  --valid-file "$(PREF_A_SPLIT)/valid.jsonl" \
	  --out "$(ROOT)outputs/pref-a-bt-sentseq-compare.json" \
	  --bt-ruri "$(BT_A_RURI_DIR)" \
	  --bt-modernbert "$(BT_A_MODERNBERT_DIR)" \
	  --bt-modernbert-d "$(BT_A_MODERNBERT_D_DIR)" \
	  --sentseq-ruri "$(SENTSEQ_A_RURI_DIR)" \
	  --sentseq-modernbert "$(SENTSEQ_A_MODERNBERT_DIR)" \
	  --sentseq-modernbert-d "$(SENTSEQ_A_MODERNBERT_D_DIR)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-a-bc:
	$(PYTHON) scripts/eval_pref_a_bc.py \
	  --out "$(ROOT)outputs/pref-a-bc-eval.json" \
	  --bt-ruri "$(BT_A_RURI_DIR)" \
	  --bt-modernbert "$(BT_A_MODERNBERT_DIR)" \
	  --bt-modernbert-d "$(BT_A_MODERNBERT_D_DIR)" \
	  --sentseq-ruri "$(SENTSEQ_A_RURI_DIR)" \
	  --sentseq-modernbert "$(SENTSEQ_A_MODERNBERT_DIR)" \
	  --sentseq-modernbert-d "$(SENTSEQ_A_MODERNBERT_D_DIR)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-a-hard-eval:
	$(PYTHON) scripts/eval_pref_a_hard_eval.py \
	  --out "$(ROOT)outputs/pref-a-hard-eval.json" \
	  --bt-ruri "$(BT_A_RURI_DIR)" \
	  --bt-modernbert "$(BT_A_MODERNBERT_DIR)" \
	  --bt-modernbert-d "$(BT_A_MODERNBERT_D_DIR)" \
	  --sentseq-ruri "$(SENTSEQ_A_RURI_DIR)" \
	  --sentseq-modernbert "$(SENTSEQ_A_MODERNBERT_DIR)" \
	  --sentseq-modernbert-d "$(SENTSEQ_A_MODERNBERT_D_DIR)" \
	  --device "$(or $(DEVICE),cuda)"

eval-sentseq-a-modernbert-d-calibrated:
	$(PYTHON) scripts/eval_sentseq_a_modernbert_d_calibrated.py \
	  --model "$(SENTSEQ_A_MODERNBERT_D_DIR)" \
	  --out-bc "$(ROOT)outputs/pref-a-bc-eval-sentseq-modernbert-d-delta.json" \
	  --out-hard "$(ROOT)outputs/pref-a-hard-eval-sentseq-modernbert-d-delta.json" \
	  --device "$(or $(DEVICE),cuda)"

pref-d-data:
	@test -s "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)/position_judgments.jsonl" || (echo "missing position_judgments.jsonl" && exit 1)
	$(PYTHON) scripts/build_pref_d_dataset.py \
	  --samples-dir "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)" \
	  --judgments "$(or $(JUDGMENTS),$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)/position_judgments.jsonl)" \
	  --out-dir "$(or $(OUT),$(DATA_DIR)/d)" \
	  --eval-fraction $(or $(EVAL_FRACTION),0.2) \
	  --seed $(or $(SEED),0)

pref-d-modernbert-train:
	@test -s "$(PREF_D_DATA)/folds/fold0_train.jsonl" || (echo "run make pref-d-data first" && exit 1)
	@mkdir -p "$(PREF_D_MODERNBERT_CV)/fold0"
	$(PYTHON) scripts/train_pref_d_interval.py \
	  --backend modernbert \
	  --base-model "$(or $(MODERNBERT_MODEL),sbintuitions/modernbert-ja-310m)" \
	  --train-file "$(PREF_D_DATA)/folds/fold0_train.jsonl" \
	  --valid-file "$(PREF_D_DATA)/folds/fold0_valid.jsonl" \
	  --output-dir "$(PREF_D_MODERNBERT_CV)/fold0" \
	  --fold 0 \
	  --epochs $(or $(EPOCHS),20) \
	  --batch-size $(or $(PREF_D_BATCH_SIZE),$(BATCH_SIZE),8) \
	  --seed $(or $(SEED),0)

pref-d-modernbert-cv:
	@test -s "$(PREF_D_DATA)/folds/fold0_train.jsonl" || (echo "run make pref-d-data first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_MODERNBERT_CV))"
	$(PYTHON) scripts/run_pref_d_cv.py \
	  --backend modernbert \
	  --base-model "$(or $(MODERNBERT_MODEL),sbintuitions/modernbert-ja-310m)" \
	  --data-dir "$(PREF_D_DATA)" \
	  --output-dir "$(or $(OUT),$(PREF_D_MODERNBERT_CV))" \
	  --max-epochs $(or $(EPOCHS),20) \
	  --batch-size $(or $(PREF_D_BATCH_SIZE),$(BATCH_SIZE),8) \
	  --seed $(or $(SEED),0)

pref-d-ruri-cv:
	@test -s "$(PREF_D_DATA)/folds/fold0_train.jsonl" || (echo "run make pref-d-data first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_RURI_CV))"
	$(PYTHON) scripts/run_pref_d_cv.py \
	  --backend ruri \
	  --base-model "$(EMBED_MODEL)" \
	  --data-dir "$(PREF_D_DATA)" \
	  --output-dir "$(or $(OUT),$(PREF_D_RURI_CV))" \
	  --max-epochs $(or $(EPOCHS),20) \
	  --batch-size $(or $(PREF_D_BATCH_SIZE),$(BATCH_SIZE),8) \
	  --seed $(or $(SEED),0)

pref-d-pair-ledger:
	@test -s "$(PREF_D_DATA)/dataset.jsonl" || (echo "run make pref-d-data first" && exit 1)
	$(PYTHON) scripts/build_pref_d_pair_ledger.py \
	  --dataset "$(PREF_D_DATA)/dataset.jsonl" \
	  --out-ledger "$(PREF_D_PAIR_LEDGER)" \
	  --out-stats "$(PREF_D_DATA)/pair_ledger_stats.json"

pref-d-bt-modernbert-fg: pref-d-pair-ledger
	@test -s "$(PREF_D_PAIR_LEDGER)" || (echo "run make pref-d-pair-ledger first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_BT_MODERNBERT))"
	$(PYTHON) scripts/run_pref_d_pair_cv.py \
	  --mode bt \
	  --base-model "$(or $(MODERNBERT_MODEL),sbintuitions/modernbert-ja-310m)" \
	  --data-dir "$(PREF_D_DATA)" \
	  --ledger-file "$(PREF_D_PAIR_LEDGER)" \
	  --output-dir "$(or $(OUT),$(PREF_D_BT_MODERNBERT))" \
	  --max-epochs $(or $(EPOCHS),40) \
	  --batch-items $(or $(PREF_D_BATCH_ITEMS),16) \
	  --pair-batch-size $(or $(PREF_D_PAIR_BATCH_SIZE),$(PREF_D_BATCH_SIZE),8) \
	  --seed $(or $(SEED),0)

pref-d-bt-modernbert: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-bt-modernbert \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-modernbert.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-bt-modernbert.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  -- $(MAKE) --no-print-directory pref-d-bt-modernbert-fg \
	    OUT="$(or $(OUT),$(PREF_D_BT_MODERNBERT))" \
	    EPOCHS="$(or $(EPOCHS),40)" \
	    SEED="$(or $(SEED),0)"

pref-d-bt-modernbert-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-bt-modernbert \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-modernbert.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-bt-modernbert.log)"

pref-d-bt-modernbert-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-modernbert.pid"

pref-d-gpm-modernbert-fg: pref-d-pair-ledger
	@test -s "$(PREF_D_PAIR_LEDGER)" || (echo "run make pref-d-pair-ledger first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_GPM_MODERNBERT))"
	$(PYTHON) scripts/run_pref_d_pair_cv.py \
	  --mode gpm \
	  --base-model "$(or $(MODERNBERT_MODEL),sbintuitions/modernbert-ja-310m)" \
	  --data-dir "$(PREF_D_DATA)" \
	  --ledger-file "$(PREF_D_PAIR_LEDGER)" \
	  --output-dir "$(or $(OUT),$(PREF_D_GPM_MODERNBERT))" \
	  --head-dim $(or $(GPM_HEAD_DIM),64) \
	  --max-epochs $(or $(EPOCHS),40) \
	  --batch-items $(or $(PREF_D_BATCH_ITEMS),16) \
	  --pair-batch-size $(or $(PREF_D_PAIR_BATCH_SIZE),$(PREF_D_BATCH_SIZE),8) \
	  --seed $(or $(SEED),0)

pref-d-gpm-modernbert: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-gpm-modernbert \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-modernbert.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-gpm-modernbert.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  -- $(MAKE) --no-print-directory pref-d-gpm-modernbert-fg \
	    OUT="$(or $(OUT),$(PREF_D_GPM_MODERNBERT))" \
	    GPM_HEAD_DIM="$(or $(GPM_HEAD_DIM),64)" \
	    EPOCHS="$(or $(EPOCHS),40)" \
	    SEED="$(or $(SEED),0)"

pref-d-gpm-modernbert-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-gpm-modernbert \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-modernbert.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-gpm-modernbert.log)"

pref-d-gpm-modernbert-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-modernbert.pid"

pref-d-bt-qwen3-8b-fg: pref-d-pair-ledger
	@test -s "$(PREF_D_PAIR_LEDGER)" || (echo "run make pref-d-pair-ledger first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_BT_QWEN))"
	$(PYTHON) scripts/run_pref_d_pair_cv_qwen.py \
	  --mode bt \
	  --base-model "$(or $(QWEN_MODEL),$(QWEN3_8B_MODEL))" \
	  --data-dir "$(PREF_D_DATA)" \
	  --ledger-file "$(PREF_D_PAIR_LEDGER)" \
	  --output-dir "$(or $(OUT),$(PREF_D_BT_QWEN))" \
	  --max-epochs $(or $(EPOCHS),40) \
	  --batch-items $(or $(PREF_D_QWEN_BATCH_ITEMS),8) \
	  --pair-batch-size $(or $(PREF_D_QWEN_PAIR_BATCH_SIZE),2) \
	  --max-length $(or $(PREF_D_QWEN_MAX_LENGTH),2048) \
	  --learning-rate $(or $(PREF_D_QWEN_LR),1e-4) \
	  --max-fold $(or $(MAX_FOLD),4) \
	  --seed $(or $(SEED),0)

pref-d-bt-qwen3-8b: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-bt-qwen3-8b \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-qwen3-8b.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-bt-qwen3-8b.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  -- $(MAKE) --no-print-directory pref-d-bt-qwen3-8b-fg \
	    OUT="$(or $(OUT),$(PREF_D_BT_QWEN))" \
	    EPOCHS="$(or $(EPOCHS),40)" \
	    SEED="$(or $(SEED),0)"

pref-d-bt-qwen3-8b-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-bt-qwen3-8b \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-qwen3-8b.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-bt-qwen3-8b.log)"

pref-d-bt-qwen3-8b-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-bt-qwen3-8b.pid"

pref-d-gpm-qwen3-8b-fg: pref-d-pair-ledger
	@test -s "$(PREF_D_PAIR_LEDGER)" || (echo "run make pref-d-pair-ledger first" && exit 1)
	@mkdir -p "$(or $(OUT),$(PREF_D_GPM_QWEN))"
	$(PYTHON) scripts/run_pref_d_pair_cv_qwen.py \
	  --mode gpm \
	  --base-model "$(or $(QWEN_MODEL),$(QWEN3_8B_MODEL))" \
	  --data-dir "$(PREF_D_DATA)" \
	  --ledger-file "$(PREF_D_PAIR_LEDGER)" \
	  --output-dir "$(or $(OUT),$(PREF_D_GPM_QWEN))" \
	  --head-dim $(or $(GPM_HEAD_DIM),16) \
	  --max-epochs $(or $(EPOCHS),40) \
	  --batch-items $(or $(PREF_D_QWEN_BATCH_ITEMS),8) \
	  --pair-batch-size $(or $(PREF_D_QWEN_PAIR_BATCH_SIZE),2) \
	  --max-length $(or $(PREF_D_QWEN_MAX_LENGTH),2048) \
	  --learning-rate $(or $(PREF_D_QWEN_LR),1e-4) \
	  --max-fold $(or $(MAX_FOLD),4) \
	  --seed $(or $(SEED),0)

pref-d-gpm-qwen3-8b: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-gpm-qwen3-8b \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-qwen3-8b.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-gpm-qwen3-8b.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  -- $(MAKE) --no-print-directory pref-d-gpm-qwen3-8b-fg \
	    OUT="$(or $(OUT),$(PREF_D_GPM_QWEN))" \
	    GPM_HEAD_DIM="$(or $(GPM_HEAD_DIM),16)" \
	    EPOCHS="$(or $(EPOCHS),40)" \
	    SEED="$(or $(SEED),0)"

pref-d-gpm-qwen3-8b-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-gpm-qwen3-8b \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-qwen3-8b.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-gpm-qwen3-8b.log)"

pref-d-gpm-qwen3-8b-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-gpm-qwen3-8b.pid"

pref-d-qwen3-8b-pipeline: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-qwen3-8b-pipeline \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen3-8b-pipeline.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-qwen3-8b-pipeline.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  --env GPM_HEAD_DIM="$(or $(GPM_HEAD_DIM),16)" \
	  -- $(ROOT)scripts/run_pref_d_qwen_pipeline.sh

pref-d-qwen3-8b-pipeline-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-qwen3-8b-pipeline \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen3-8b-pipeline.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-qwen3-8b-pipeline.log)"

pref-d-qwen3-8b-pipeline-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen3-8b-pipeline.pid"

pref-d-qwen-gpm-bt-fold2: pref-d-pair-ledger
	@mkdir -p "$(DETACHED_JOBS_DIR)"
	$(RUN_DETACHED) start \
	  --name pref-d-qwen-gpm-bt-fold2 \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen-gpm-bt-fold2.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-qwen-gpm-bt-fold2.log)" \
	  --workdir "$(ROOT)" \
	  --env PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" \
	  -- $(ROOT)scripts/run_pref_d_qwen_gpm-bt_fold2.sh

pref-d-qwen-gpm-bt-fold2-status:
	@$(RUN_DETACHED) status \
	  --name pref-d-qwen-gpm-bt-fold2 \
	  --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen-gpm-bt-fold2.pid" \
	  --log-file "$(or $(LOG),$(DETACHED_JOBS_DIR)/pref-d-qwen-gpm-bt-fold2.log)"

pref-d-qwen-gpm-bt-fold2-stop:
	@$(RUN_DETACHED) stop --pid-file "$(DETACHED_JOBS_DIR)/pref-d-qwen-gpm-bt-fold2.pid"

eval-pref-d-bt-qwen3-8b:
	@test -s "$(or $(OUT),$(PREF_D_BT_QWEN))/eval_cv_report.json" || (echo "run make pref-d-bt-qwen3-8b first" && exit 1)
	@$(PYTHON) -c "import json; print(json.dumps(json.load(open('$(or $(OUT),$(PREF_D_BT_QWEN))/eval_cv_report.json')), ensure_ascii=False, indent=2))"

eval-pref-d-gpm-qwen3-8b:
	@test -s "$(or $(OUT),$(PREF_D_GPM_QWEN))/eval_cv_report.json" || (echo "run make pref-d-gpm-qwen3-8b first" && exit 1)
	@$(PYTHON) -c "import json; print(json.dumps(json.load(open('$(or $(OUT),$(PREF_D_GPM_QWEN))/eval_cv_report.json')), ensure_ascii=False, indent=2))"

eval-pref-d-bt-modernbert:
	@test -s "$(or $(OUT),$(PREF_D_BT_MODERNBERT))/eval_cv_report.json" || (echo "run make pref-d-bt-modernbert first" && exit 1)
	@$(PYTHON) -c "import json; print(json.dumps(json.load(open('$(or $(OUT),$(PREF_D_BT_MODERNBERT))/eval_cv_report.json')), ensure_ascii=False, indent=2))"

eval-pref-d-gpm-modernbert:
	@test -s "$(or $(OUT),$(PREF_D_GPM_MODERNBERT))/eval_cv_report.json" || (echo "run make pref-d-gpm-modernbert first" && exit 1)
	@$(PYTHON) -c "import json; print(json.dumps(json.load(open('$(or $(OUT),$(PREF_D_GPM_MODERNBERT))/eval_cv_report.json')), ensure_ascii=False, indent=2))"

eval-pref-c-d-bt-modernbert:
	@test -s "$(PREF_D_BT_MODERNBERT)/fold0/best/config.json" || (echo "run make pref-d-bt-modernbert first" && exit 1)
	@test -s "$(DATA_DIR)/blind_eval/judgments_gold_vs_adapter_selected.jsonl" || (echo "missing C judgments" && exit 1)
	$(PYTHON) scripts/eval_pref_d_pair_c.py \
	  --cv-dir "$(or $(CV_DIR),$(PREF_D_BT_MODERNBERT))" \
	  --fold $(or $(FOLD),0) \
	  --out "$(or $(OUT),$(ROOT)outputs/pref-c-eval-d-bt-modernbert.json)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-c-d-gpm-modernbert:
	@test -s "$(or $(CV_DIR),$(PREF_D_GPM_MODERNBERT))/fold0/best/config.json" || (echo "run make pref-d-gpm-modernbert first" && exit 1)
	@test -s "$(DATA_DIR)/blind_eval/judgments_gold_vs_adapter_selected.jsonl" || (echo "missing C judgments" && exit 1)
	$(PYTHON) scripts/eval_pref_d_pair_c.py \
	  --cv-dir "$(or $(CV_DIR),$(PREF_D_GPM_MODERNBERT))" \
	  --fold $(or $(FOLD),0) \
	  --out "$(or $(OUT),$(ROOT)outputs/pref-c-eval-d-gpm-modernbert.json)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-c-d-pair-all:
	$(MAKE) eval-pref-c-d-bt-modernbert
	$(MAKE) eval-pref-c-d-gpm-modernbert CV_DIR=$(PREF_D_GPM_MODERNBERT) OUT=$(ROOT)outputs/pref-c-eval-d-gpm-modernbert.json
	$(MAKE) eval-pref-c-d-gpm-modernbert CV_DIR=$(ROOT)outputs/pref-d-gpm-modernbert-k16 OUT=$(ROOT)outputs/pref-c-eval-d-gpm-modernbert-k16.json
	$(MAKE) eval-pref-c-d-gpm-modernbert CV_DIR=$(ROOT)outputs/pref-d-gpm-modernbert-k8 OUT=$(ROOT)outputs/pref-c-eval-d-gpm-modernbert-k8.json

eval-pref-hard-d-bt-modernbert:
	@test -s "$(PREF_D_BT_MODERNBERT)/fold0/best/config.json" || (echo "run make pref-d-bt-modernbert first" && exit 1)
	@test -s "$(DATA_DIR)/hard_eval/bases_v2b_human_fable_copy.jsonl" || (echo "missing hard eval v2b" && exit 1)
	$(PYTHON) scripts/eval_pref_d_pair_hard_eval.py \
	  --cv-dir "$(or $(CV_DIR),$(PREF_D_BT_MODERNBERT))" \
	  --fold $(or $(FOLD),0) \
	  --out "$(or $(OUT),$(ROOT)outputs/pref-hard-eval-d-bt-modernbert.json)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-hard-d-gpm-modernbert:
	@test -s "$(or $(CV_DIR),$(PREF_D_GPM_MODERNBERT))/fold0/best/config.json" || (echo "run make pref-d-gpm-modernbert first" && exit 1)
	@test -s "$(DATA_DIR)/hard_eval/bases_v2b_human_fable_copy.jsonl" || (echo "missing hard eval v2b" && exit 1)
	$(PYTHON) scripts/eval_pref_d_pair_hard_eval.py \
	  --cv-dir "$(or $(CV_DIR),$(PREF_D_GPM_MODERNBERT))" \
	  --fold $(or $(FOLD),0) \
	  --out "$(or $(OUT),$(ROOT)outputs/pref-hard-eval-d-gpm-modernbert.json)" \
	  --device "$(or $(DEVICE),cuda)"

eval-pref-hard-d-pair-all:
	$(MAKE) eval-pref-hard-d-bt-modernbert
	$(MAKE) eval-pref-hard-d-gpm-modernbert CV_DIR=$(PREF_D_GPM_MODERNBERT) OUT=$(ROOT)outputs/pref-hard-eval-d-gpm-modernbert.json
	$(MAKE) eval-pref-hard-d-gpm-modernbert CV_DIR=$(ROOT)outputs/pref-d-gpm-modernbert-k16 OUT=$(ROOT)outputs/pref-hard-eval-d-gpm-modernbert-k16.json
	$(MAKE) eval-pref-hard-d-gpm-modernbert CV_DIR=$(ROOT)outputs/pref-d-gpm-modernbert-k8 OUT=$(ROOT)outputs/pref-hard-eval-d-gpm-modernbert-k8.json

pref-interval-a1-probe:
	@test -s "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)/position_judgments.jsonl" || (echo "missing position_judgments.jsonl" && exit 1)
	@test -s "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" || (echo "run make pairsplit-data first" && exit 1)
	$(PYTHON) scripts/train_pref_interval_a1_probe.py \
	  --samples-dir "$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)" \
	  --judgments "$(or $(JUDGMENTS),$(or $(A1_PROBE_DIR),$(ROOT)outputs/a1-probe)/position_judgments.jsonl)" \
	  --hunk-train-file "$(PREF_KEEP_SPLIT_HUNK)/train.jsonl" \
	  --eval-fraction $(or $(EVAL_FRACTION),0.2) \
	  --out-data-dir "$(A1_PROBE_INTERVAL_DATA)" \
	  --output-dir "$(or $(OUT),$(PREF_INTERVAL_A1_PROBE_DIR))" \
	  --model "$(EMBED_MODEL)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --encode-batch-size $(or $(ENCODE_BATCH_SIZE),64) \
	  --d-model $(or $(SENTSEQ_D_MODEL),256) \
	  --num-layers $(or $(SENTSEQ_NUM_LAYERS),2) \
	  --max-sents $(or $(SENTSEQ_MAX_SENTS),128) \
	  --batch-size $(or $(BATCH_SIZE),32) \
	  --epochs $(or $(EPOCHS),40) \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  --weight-decay $(or $(WEIGHT_DECAY),1e-2) \
	  --seed "$(or $(SEED),0)" \
	  --device $(or $(DEVICE),cuda)

eval-pref-interval-a1-probe:
	@test -s "$(or $(OUT),$(PREF_INTERVAL_A1_PROBE_DIR))/model.pt" || (echo "run make pref-interval-a1-probe first" && exit 1)
	$(PYTHON) scripts/eval_pref_interval_a1_probe.py \
	  --model "$(or $(OUT),$(PREF_INTERVAL_A1_PROBE_DIR))" \
	  --valid-probe-file "$(A1_PROBE_INTERVAL_DATA)/valid_probe.jsonl" \
	  --hunk-valid-file "$(PREF_KEEP_SPLIT_HUNK)/valid.jsonl" \
	  --out "$(or $(OUT),$(PREF_INTERVAL_A1_PROBE_DIR))/eval_report.json" \
	  --device $(or $(DEVICE),cuda)

build-blind-pairs:
	$(PYTHON) scripts/build_blind_pairs.py \
	  --items "$(DATA_DIR)/blind_eval/items.jsonl" \
	  --out "$(DATA_DIR)/blind_eval/pairs.jsonl" \
	  --seed $(or $(SEED),42) \
	  --primary-model "$(or $(PRIMARY_MODEL),$(SENTSEQ_BEST_DIR))"

blind-judge:
	$(PYTHON) scripts/blind_judge_server.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments.jsonl" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8320)

pref-valid-blind-pairs:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/build_pref_valid_gold_vs_composer_pairs.py \
	  --valid-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --out "$(DATA_DIR)/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl" \
	  --seed "$(or $(SEED),42)"

pref-valid-blind-judge:
	@test -s "$(DATA_DIR)/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl" || (echo "run make pref-valid-blind-pairs first" && exit 1)
	$(PYTHON) scripts/blind_judge_server.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs_pref_valid_gold_vs_composer.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments_pref_valid_gold_vs_composer.jsonl" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8324)

scalar-transitivity-pairs:
	@test -s "$(DATA_DIR)/blind_eval/items.jsonl" || (echo "missing data/blind_eval/items.jsonl" && exit 1)
	@test -s "outputs/edit-sft-eval-v3/adapter_samples.jsonl" || (echo "missing adapter_samples.jsonl" && exit 1)
	$(PYTHON) scripts/build_scalar_transitivity_pairs.py \
	  --items "$(DATA_DIR)/blind_eval/items.jsonl" \
	  --adapter-samples "outputs/edit-sft-eval-v3/adapter_samples.jsonl" \
	  --primary-model "$(SENTSEQ_BEST_DIR)" \
	  --out "$(DATA_DIR)/blind_eval/pairs_scalar_transitivity.jsonl" \
	  --protocol-out "$(DATA_DIR)/blind_eval/scalar_transitivity_protocol.json" \
	  --n-items $(or $(N_ITEMS),10) \
	  --n-unselected $(or $(N_UNSELECTED),2) \
	  --seed $(or $(SEED),42)

scalar-transitivity-judge:
	@test -s "$(DATA_DIR)/blind_eval/pairs_scalar_transitivity.jsonl" || (echo "run make scalar-transitivity-pairs first" && exit 1)
	$(PYTHON) scripts/blind_judge_server.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs_scalar_transitivity.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments_scalar_transitivity.jsonl" \
	  --question "A と B のどちらが、自分の推敲に近いか" \
	  --choice-a "A のほうが近い" \
	  --choice-b "B のほうが近い" \
	  --choice-tie "同程度" \
	  --choice-incomparable "比較できない" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8325)

analyze-scalar-transitivity:
	@test -s "$(DATA_DIR)/blind_eval/pairs_scalar_transitivity.jsonl" || (echo "run make scalar-transitivity-pairs first" && exit 1)
	$(PYTHON) scripts/analyze_scalar_transitivity.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs_scalar_transitivity.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments_scalar_transitivity.jsonl" \
	  --out "$(DATA_DIR)/blind_eval/scalar_transitivity_analysis.json"

b-rejudge-pairs:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/build_b_rejudge_pairs.py \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --valid-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --items "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --revisions "$(SECTION_MIDDLE_DIR)/revisions.jsonl" \
	  --judgments "$(SECTION_MIDDLE_DIR)/judgments.jsonl" \
	  --out "$(SECTION_MIDDLE_DIR)/pairs_rejudge.jsonl" \
	  --protocol-out "$(SECTION_MIDDLE_DIR)/b_rejudge_protocol.json" \
	  --seed $(or $(SEED),42)

b-rejudge-judge:
	@test -s "$(SECTION_MIDDLE_DIR)/pairs_rejudge.jsonl" || (echo "run make b-rejudge-pairs first" && exit 1)
	$(PYTHON) scripts/blind_judge_server.py \
	  --pairs "$(SECTION_MIDDLE_DIR)/pairs_rejudge.jsonl" \
	  --judgments "$(SECTION_MIDDLE_DIR)/judgments_rejudge.jsonl" \
	  --question "A と B のどちらが、自分の推敲に近いか" \
	  --choice-a "A のほうが近い" \
	  --choice-b "B のほうが近い" \
	  --choice-tie "同程度" \
	  --choice-incomparable "比較できない" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8326)

analyze-b-rejudge:
	@test -s "$(SECTION_MIDDLE_DIR)/pairs_rejudge.jsonl" || (echo "run make b-rejudge-pairs first" && exit 1)
	$(PYTHON) scripts/analyze_b_rejudge.py \
	  --pairs "$(SECTION_MIDDLE_DIR)/pairs_rejudge.jsonl" \
	  --judgments "$(SECTION_MIDDLE_DIR)/judgments_rejudge.jsonl" \
	  --out "$(SECTION_MIDDLE_DIR)/b_rejudge_analysis.json"

GENERATED_PREF_DATA_DIR := $(DATA_DIR)/generated_pref_experiment
GENERATED_PREF_OUTPUT_DIR := $(ROOT)outputs/generated-pref-sentseq
GENERATED_PREF_FOLD ?= 0
# help では FOLD= を案内。未指定時は GENERATED_PREF_FOLD にフォールバック。
PREF_CV_FOLD = $(or $(FOLD),$(GENERATED_PREF_FOLD))

analyze-blind-judgments:
	$(PYTHON) scripts/analyze_blind_judgments.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments.jsonl" \
	  --items "$(DATA_DIR)/blind_eval/items.jsonl" \
	  --adapter-greedy "$(or $(ADAPTER_GREEDY),outputs/edit-sft-eval-v3/adapter_greedy.jsonl)" \
	  --sentseq-model "$(SENTSEQ_KEEP_PAIRSPLIT_DIR)" \
	  --bt-model "$(BT_KEEP_PAIRSPLIT_DIR)" \
	  --extra-sentseq-model "$(SENTSEQ_MIDDLE_DIR)" \
	  --analysis-json "$(DATA_DIR)/blind_eval/analysis.json" \
	  --results-md docs/RESULTS.md

generated-pref-data:
	@test -s "$(DATA_DIR)/blind_eval/pairs.jsonl" || (echo "missing blind pairs" && exit 1)
	@test -s "$(DATA_DIR)/blind_eval/judgments.jsonl" || (echo "missing blind judgments" && exit 1)
	@test -s "$(DATA_DIR)/blind_eval/items.jsonl" || (echo "missing blind items" && exit 1)
	$(PYTHON) scripts/build_generated_pref_data.py \
	  --pairs "$(DATA_DIR)/blind_eval/pairs.jsonl" \
	  --judgments "$(DATA_DIR)/blind_eval/judgments.jsonl" \
	  --items "$(DATA_DIR)/blind_eval/items.jsonl" \
	  --out-dir "$(GENERATED_PREF_DATA_DIR)" \
	  --seed $(or $(SEED),42)

generated-pref-sentseq-smoke: generated-pref-data
	@test -s "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_0/train.jsonl" || (echo "run make generated-pref-data first" && exit 1)
	$(PYTHON) scripts/train_generated_pref_sentseq.py \
	  --train-file "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_0/train.jsonl" \
	  --valid-file "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_0/valid.jsonl" \
	  --output-dir "$(GENERATED_PREF_OUTPUT_DIR)-smoke" \
	  --anchor-file "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" \
	  --anchor-batch-fraction 0.25 \
	  --limit-train 16 \
	  --limit-valid 8 \
	  --epochs 1 \
	  --batch-size 8 \
	  --encode-batch-size 8 \
	  --device cpu

generated-pref-sentseq-cv: generated-pref-data
	@test -s "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_$(PREF_CV_FOLD)/train.jsonl" \
	  || (echo "missing fold $(PREF_CV_FOLD); run make generated-pref-data" && exit 1)
	$(PYTHON) scripts/train_generated_pref_sentseq.py \
	  --train-file "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_$(PREF_CV_FOLD)/train.jsonl" \
	  --valid-file "$(GENERATED_PREF_DATA_DIR)/folds_section/fold_$(PREF_CV_FOLD)/valid.jsonl" \
	  --output-dir "$(GENERATED_PREF_OUTPUT_DIR)-fold$(PREF_CV_FOLD)$(if $(filter 1,$(LENGTH_FEATURES)),-lenon,)" \
	  --anchor-file "$(PREF_KEEP_SPLIT_SECTION)/train.jsonl" \
	  --anchor-batch-fraction $(or $(ANCHOR_BATCH_FRACTION),0.25) \
	  $(if $(filter 1,$(LENGTH_FEATURES)),--length-features,) \
	  --epochs $(or $(EPOCHS),20) \
	  --batch-size $(or $(BATCH_SIZE),64) \
	  --device $(or $(DEVICE),cuda)

SECTION_MIDDLE_DIR := $(DATA_DIR)/section_middle

section-middle-gen:
	$(PYTHON) scripts/generate_section_composer_revisions.py \
	  --input "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --out "$(SECTION_MIDDLE_DIR)/revisions.jsonl" \
	  --model "$(or $(MODEL),composer-2.5)" \
	  --limit $(or $(LIMIT),0) \
	  --offset $(or $(OFFSET),0)

section-middle-judge:
	$(PYTHON) scripts/middle_degrade_server.py \
	  --items "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --revisions "$(SECTION_MIDDLE_DIR)/revisions.jsonl" \
	  --judgments "$(SECTION_MIDDLE_DIR)/judgments.jsonl" \
	  --host $(or $(HOST),0.0.0.0) \
	  --port $(or $(PORT),8321)

section-middle-triples:
	@test -s "$(SECTION_MIDDLE_DIR)/revisions.jsonl" || (echo "missing revisions: run make section-middle-gen" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/judgments.jsonl" || (echo "missing judgments: run make section-middle-judge" && exit 1)
	$(PYTHON) scripts/build_section_triples.py \
	  --items "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --revisions "$(SECTION_MIDDLE_DIR)/revisions.jsonl" \
	  --judgments "$(SECTION_MIDDLE_DIR)/judgments.jsonl" \
	  --out-dir "$(SECTION_MIDDLE_DIR)"

section-middle-sentseq:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_sentseq.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(SENTSEQ_MIDDLE_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SENTSEQ_KEEP_EPOCHS),40) \
	  --batch-size $(SENTSEQ_BATCH_SIZE) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-section-middle-sentseq-image:
	bash scripts/build_push_section_middle_sentseq_image.sh

section-middle-setwise:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_setwise.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(SETWISE_MIDDLE_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SETWISE_EPOCHS),40) \
	  --batch-size $(or $(SETWISE_BATCH_SIZE),32) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-setwise-section-triples-image:
	bash scripts/build_push_setwise_section_triples_image.sh

section-middle-setwise-human-top:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_setwise.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(SETWISE_HUMAN_TOP_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(SETWISE_EPOCHS),40) \
	  --batch-size $(or $(SETWISE_BATCH_SIZE),32) \
	  --device "$(or $(DEVICE),cuda)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  --loss-mode human_top \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-setwise-section-human-top-image:
	bash scripts/build_push_setwise_section_human_top_image.sh

section-middle-nce:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_nce.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(NCE_MIDDLE_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(NCE_EPOCHS),40) \
	  --batch-size $(or $(NCE_BATCH_SIZE),32) \
	  --device "$(or $(DEVICE),cpu)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  --tau "$(or $(NCE_TAU),0.07)" \
	  --seed "$(or $(SEED),0)" \
	  $(if $(MAX_TRAIN),--max-train $(MAX_TRAIN),) \
	  $(if $(MAX_VALID),--max-valid $(MAX_VALID),) \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-section-middle-nce-image:
	bash scripts/build_push_section_middle_nce_image.sh

section-middle-detect:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_detect.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(DETECT_MIDDLE_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(DETECT_EPOCHS),40) \
	  --batch-size $(or $(DETECT_BATCH_SIZE),64) \
	  --device "$(or $(DEVICE),cpu)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  --seed "$(or $(SEED),0)" \
	  $(if $(MAX_TRAIN),--max-train $(MAX_TRAIN),) \
	  $(if $(MAX_VALID),--max-valid $(MAX_VALID),) \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-section-middle-detect-image:
	bash scripts/build_push_section_middle_detect_image.sh

section-middle-detect-cd:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	@test -s "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/train_pref_detect.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" \
	  --eval-file "$(SECTION_MIDDLE_DIR)/pref_valid.jsonl" \
	  --output-dir "$(DETECT_CD_MIDDLE_DIR)" \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(or $(SENTSEQ_MAX_SEQ_LENGTH),256) \
	  --epochs $(or $(DETECT_EPOCHS),40) \
	  --batch-size $(or $(DETECT_CD_BATCH_SIZE),32) \
	  --device "$(or $(DEVICE),cpu)" \
	  --lr "$(or $(SENTSEQ_KEEP_LR),3e-4)" \
	  --seed "$(or $(SEED),0)" \
	  --composer-over-draft \
	  $(if $(MAX_TRAIN),--max-train $(MAX_TRAIN),) \
	  $(if $(MAX_VALID),--max-valid $(MAX_VALID),) \
	  $(if $(SENTSEQ_D_MODEL),--d-model $(SENTSEQ_D_MODEL),) \
	  $(if $(SENTSEQ_NUM_LAYERS),--num-layers $(SENTSEQ_NUM_LAYERS),) \
	  $(if $(SENTSEQ_MAX_SENTS),--max-sents $(SENTSEQ_MAX_SENTS),)

build-section-middle-detect-cd-image:
	bash scripts/build_push_section_middle_detect_cd_image.sh


freeze-8d-items:
	$(PYTHON) scripts/freeze_8d_items.py

pref-multigranular-smoke:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/run_pref_multigranular_stages.py \
	  --stages "$(or $(STAGES),2)" \
	  --device "$(or $(DEVICE),cpu)" \
	  --epochs $(or $(EPOCHS),1) \
	  --phase1-epochs $(or $(PHASE1_EPOCHS),1) \
	  --phase2-epochs $(or $(PHASE2_EPOCHS),1) \
	  --batch-size $(or $(SETWISE_BATCH_SIZE),2) \
	  --max-train $(or $(MAX_TRAIN),8) \
	  --max-valid $(or $(MAX_VALID),8) \
	  --seeds "$(or $(SEEDS),$(MULTIGRANULAR_SEEDS))" \
	  --report-dir "$(or $(REPORT_DIR),outputs/pref-multigranular-report-smoke)"

pref-multigranular:
	@test -s "$(SECTION_MIDDLE_DIR)/pref_train.jsonl" || (echo "run make section-middle-triples first" && exit 1)
	$(PYTHON) scripts/run_pref_multigranular_stages.py \
	  --stages "$(or $(STAGES),$(MULTIGRANULAR_STAGES))" \
	  --device "$(or $(DEVICE),cuda)" \
	  --epochs $(or $(EPOCHS),40) \
	  --phase1-epochs $(or $(PHASE1_EPOCHS),10) \
	  --phase2-epochs $(or $(PHASE2_EPOCHS),40) \
	  --batch-size $(or $(SETWISE_BATCH_SIZE),32) \
	  --seeds "$(or $(SEEDS),$(MULTIGRANULAR_SEEDS))" \
	  --gate-model "$(or $(GATE_MODEL),$(BT_GATE_DIR))" \
	  --report-dir "$(or $(REPORT_DIR),outputs/pref-multigranular-report)"

build-pref-multigranular-image:
	bash scripts/build_push_pref_multigranular_image.sh

hard-eval-setwise:
	@test -n "$(INPUT)" || (echo "INPUT=data/hard_eval/bases_v2b_human_fable_copy.jsonl 等が必要" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL=$(SETWISE_MIDDLE_DIR) 等が必要" && exit 1)
	$(PYTHON) scripts/score_hard_eval_setwise.py \
	  --input "$(INPUT)" \
	  --model "$(MODEL)" \
	  --report "$(or $(REPORT),outputs/hard_eval_setwise_report.json)"

test:
	$(PYTHON) -m pytest tests/ -q

# レビュー keep を学習用に書き出す。
# 空行あり → data/edit_sft_section
# 空行なし → data/edit_sft_hunk_nopara
# 合流（Qwen SFT 本線）→ data/edit_sft_all
edit-sft-export-keeps:
	$(PYTHON) scripts/export_reviewed_keeps.py \
	  --section-out "$(DATA_DIR)/edit_sft_section" \
	  --hunk-nopara-out "$(DATA_DIR)/edit_sft_hunk_nopara" \
	  --all-out "$(DATA_DIR)/edit_sft_all" \
	  --corpus-dir "$(DATA_DIR)/revision_corpus"
	$(PYTHON) scripts/build_revision_corpus.py \
	  --out-dir "$(DATA_DIR)/revision_corpus" \
	  --keep-section "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --keep-hunk-nopara "$(DATA_DIR)/revision_corpus/keep_hunk_nopara.jsonl"

# 互換: export-keeps へのエイリアス
edit-sft-promote-reviewed: edit-sft-export-keeps
	@echo "promoted: all=$$(wc -l < "$(DATA_DIR)/edit_sft_all/train.jsonl") section=$$(wc -l < "$(DATA_DIR)/edit_sft_section/train.jsonl") hunk_nopara=$$(wc -l < "$(DATA_DIR)/edit_sft_hunk_nopara/train.jsonl")"

edit-sft:
	@test -n "$(MODEL)" || (echo "MODEL=<hf-id> is required" && exit 1)
	@test -s "$(DATA_DIR)/edit_sft_all/train.jsonl" || (echo "run make edit-sft-export-keeps first" && exit 1)
	$(PYTHON) scripts/train_edit_sft.py \
	  --train "$(DATA_DIR)/edit_sft_all/train.jsonl" \
	  --model "$(MODEL)" \
	  --epochs $(or $(EPOCHS),2) \
	  $(if $(filter-out 0,$(or $(LIMIT),0)),--limit $(LIMIT),) \
	  $(if $(TRUST_REMOTE_CODE),--trust-remote-code,)

edit-sft-section-only:
	@test -n "$(MODEL)" || (echo "MODEL=<hf-id> is required" && exit 1)
	@test -s "$(DATA_DIR)/edit_sft_section/train.jsonl" || (echo "run make edit-sft-export-keeps first" && exit 1)
	$(PYTHON) scripts/train_edit_sft.py \
	  --train "$(DATA_DIR)/edit_sft_section/train.jsonl" \
	  --model "$(MODEL)" \
	  --epochs $(or $(EPOCHS),2) \
	  $(if $(filter-out 0,$(or $(LIMIT),0)),--limit $(LIMIT),) \
	  $(if $(TRUST_REMOTE_CODE),--trust-remote-code,)

edit-sft-hunk:
	@test -n "$(MODEL)" || (echo "MODEL=<hf-id> is required" && exit 1)
	@test -s "$(DATA_DIR)/edit_sft_hunk_nopara/train.jsonl" || (echo "run make edit-sft-export-keeps first" && exit 1)
	$(PYTHON) scripts/train_edit_sft.py \
	  --train "$(DATA_DIR)/edit_sft_hunk_nopara/train.jsonl" \
	  --model "$(MODEL)" \
	  --epochs $(or $(EPOCHS),2) \
	  $(if $(filter-out 0,$(or $(LIMIT),0)),--limit $(LIMIT),) \
	  $(if $(TRUST_REMOTE_CODE),--trust-remote-code,)

inspect-d-epoch7:
	$(PYTHON) scripts/inspect_d_epoch7.py \
	  --code-root "$(ROOT)" \
	  --checkpoint "$(or $(CHECKPOINT),$(PREF_D_MODERNBERT_BEST))" \
	  --out "$(D_EPOCH7_OUT)/d_epoch7.lock.json"

evaluate-d-epoch7: inspect-d-epoch7
	$(PYTHON) scripts/evaluate_d_epoch7.py \
	  --lock "$(D_EPOCH7_OUT)/d_epoch7.lock.json" \
	  --d-train "$(PREF_D_DATA)/train.jsonl" \
	  --d-valid "$(PREF_D_DATA)/valid.jsonl" \
	  --out "$(D_EPOCH7_OUT)/evaluation"

prepare-b-generation:
	$(PYTHON) scripts/prepare_b_generation.py \
	  --a2 "$(DATA_DIR)/revision_corpus/keep_section.jsonl" \
	  --composer "$(DATA_DIR)/section_middle/revisions.jsonl" \
	  --pref-train "$(DATA_DIR)/section_middle/pref_train.jsonl" \
	  --pref-valid "$(DATA_DIR)/section_middle/pref_valid.jsonl" \
	  --dev-fraction $(or $(DEV_FRACTION),0.1) \
	  --split-seed $(or $(SPLIT_SEED),20260907) \
	  --model "$(or $(MODEL),Qwen/Qwen3-8B)" \
	  --model-revision $(or $(MODEL_REVISION),b968826d9c46dd6066d109eabc6255188de91218) \
	  --config "$(B_GENERATION_CONFIG)" \
	  --out "$(B_GENERATION_DATA)"

check-b-generation-local:
	@test -s "$(B_GENERATION_DATA)/manifest.json" || (echo "run make prepare-b-generation first" && exit 1)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/check_b_generation_local.py \
	  --manifest "$(B_GENERATION_DATA)/manifest.json" \
	  --config "$(B_GENERATION_CONFIG)" \
	  --out "$(B_GENERATION_OUT)/local_preflight.json" \
	  --condition $(or $(CONDITION),xy)

train-b-generation:
	@test -s "$(B_GENERATION_DATA)/manifest.json" || (echo "run make prepare-b-generation first" && exit 1)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/train_b_generation.py \
	  --data "$(B_GENERATION_DATA)" \
	  --condition $(or $(CONDITION),x) \
	  --seed $(or $(SEED),0) \
	  --config "$(B_GENERATION_CONFIG)" \
	  --out "$(or $(OUT),$(B_GENERATION_OUT)/$(or $(CONDITION),x)/seed-$(or $(SEED),0))"

b-generation-prep: inspect-d-epoch7 evaluate-d-epoch7 prepare-b-generation check-b-generation-local

infer-b-generation:
	@test -s "$(B_GENERATION_DATA)/manifest.json" || (echo "run make prepare-b-generation first" && exit 1)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON) scripts/infer_b_generation.py \
	  --manifest "$(B_GENERATION_DATA)/manifest.json" \
	  --split $(or $(SPLIT),holdout) \
	  --condition $(or $(CONDITION),xy) \
	  --seed $(or $(SEED),0) \
	  --config "$(B_GENERATION_CONFIG)" \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",)

infer-b-generation-all:
	@set -e; \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
	HOLDOUT_N=$$($(PYTHON) -c "import json;from pathlib import Path;m=json.loads(Path('$(B_GENERATION_DATA)/manifest.json').read_text());print(sum(1 for s in m['sources'] if s.get('split')=='holdout' and s.get('eligible')))"); \
	for cond in x y xy; do \
	  for seed in 0 1 2; do \
	    out="$(B_GENERATION_OUT)/predictions/$${cond}-seed-$${seed}.jsonl"; \
	    if [ -f "$$out" ] && [ "$$(wc -l < "$$out")" -eq "$$HOLDOUT_N" ]; then \
	      echo "=== skip $$cond seed $$seed ($$HOLDOUT_N lines) ==="; \
	      continue; \
	    fi; \
	    echo "=== infer $$cond seed $$seed ==="; \
	    $(PYTHON) scripts/infer_b_generation.py \
	      --manifest "$(B_GENERATION_DATA)/manifest.json" \
	      --condition $$cond --seed $$seed --config "$(B_GENERATION_CONFIG)" \
	      --resume; \
	  done; \
	done; \
	out="$(B_GENERATION_OUT)/predictions/xy-base.jsonl"; \
	if [ -f "$$out" ] && [ "$$(wc -l < "$$out")" -eq "$$HOLDOUT_N" ]; then \
	  echo "=== skip xy-base ($$HOLDOUT_N lines) ==="; \
	else \
	  echo "=== infer xy-base ==="; \
	  $(PYTHON) scripts/infer_b_generation.py \
	    --manifest "$(B_GENERATION_DATA)/manifest.json" \
	    --condition xy-base --config "$(B_GENERATION_CONFIG)" \
	    --resume; \
	fi

make-b-generation-blind:
	@test -d "$(B_GENERATION_OUT)/predictions" || (echo "run make infer-b-generation-all first" && exit 1)
	$(PYTHON) scripts/make_b_generation_blind.py \
	  --manifest "$(B_GENERATION_DATA)/manifest.json" \
	  --predictions "$(B_GENERATION_OUT)/predictions" \
	  --out "$(B_GENERATION_DATA)/blind"

score-b-generation-blind:
	@test -s "$(B_GENERATION_DATA)/blind/pairs.jsonl" || (echo "run make make-b-generation-blind first" && exit 1)
	$(PYTHON) scripts/score_b_generation_blind.py \
	  --pairs "$(B_GENERATION_DATA)/blind/pairs.jsonl" \
	  --keys "$(B_GENERATION_DATA)/blind/keys.jsonl" \
	  --scorer "$(SENTSEQ_BEST_DIR)" \
	  --out "$(B_GENERATION_DATA)/blind/judgments_scorer_proxy.jsonl"

summarize-b-generation:
	$(PYTHON) scripts/summarize_b_generation.py \
	  --manifest "$(B_GENERATION_DATA)/manifest.json" \
	  --pairs "$(B_GENERATION_DATA)/blind/pairs.jsonl" \
	  --keys "$(B_GENERATION_DATA)/blind/keys.jsonl" \
	  --judgments "$(or $(JUDGMENTS),$(B_GENERATION_DATA)/blind/judgments_scorer_proxy.jsonl)" \
	  --out "$(B_GENERATION_OUT)/results"

b-generation-eval: infer-b-generation-all make-b-generation-blind score-b-generation-blind summarize-b-generation

clean-model:
	rm -rf "$(OUTPUT_DIR)" "$(BT_OUTPUT_DIR)"

daemon:
	@mkdir -p "$(ROOT)run"
	@if [ -S "$(ROOT)run/daemon.sock" ]; then \
	  echo "daemon already running (socket: $(ROOT)run/daemon.sock)"; \
	else \
	  nohup $(PYTHON) scripts/pref_daemon.py >/dev/null 2>&1 & \
	  echo "started pref daemon (socket: $(ROOT)run/daemon.sock)"; \
	fi

daemon-stop:
	@if [ -f "$(ROOT)run/daemon.pid" ]; then \
	  kill "$$(cat "$(ROOT)run/daemon.pid")" 2>/dev/null || true; \
	  rm -f "$(ROOT)run/daemon.pid" "$(ROOT)run/daemon.sock"; \
	  echo "stopped pref daemon"; \
	else \
	  rm -f "$(ROOT)run/daemon.sock"; \
	  echo "daemon not running"; \
	fi

EAD_OUT := $(ROOT)outputs/ead

.PHONY: ead-audit ead-scale-diag ead-floor ead-metric-unify ead-transfer-diag ead-glob-mi-pilot ead-glob-mi ead-p0 \
	ead-den-a-excl-train ead-den-a-full-train ead-den-a-excl-eval ead-den-a-full-score ead-den-a-excl \
	ead-gate-defect ead-gate-level ead-gate-level-r7 ead-gate-level-ablation \
	ead-h1-d-selfgen ead-h1-c-measure ead-r5-h1 ead-readability-probe ead-ambiguity-probe ead-p3 \
	ead-calibrate-tau ead-build-e1 ead-import-e2-from-c ead-generate-e3 ead-build-e ead-validate-e ead-e-pipeline \
	ead-den-a2-train

ead-audit:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/audit.py --root "$(ROOT)"

ead-scale-diag:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/scale_diag.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)"

ead-floor:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/floor.py --root "$(ROOT)"

ead-metric-unify: ead-floor
	@mkdir -p "$(EAD_OUT)/reports"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/metric_unify.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)" \
	  $(if $(SKIP_PDPO),--skip-pdpo,)

ead-transfer-diag:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/transfer_diag.py --root "$(ROOT)"

ead-glob-mi-pilot:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/glob_mi.py \
	  --root "$(ROOT)" \
	  --pilot 20 \
	  --pilot-only \
	  --device "$(or $(DEVICE),cuda)"

ead-glob-mi:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/glob_mi.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)"

ead-p0: ead-audit ead-scale-diag ead-transfer-diag ead-floor ead-metric-unify
	@echo "ead Phase 0 complete -> $(EAD_OUT)/reports/"

EAD_DEN_A2_MAX_SEQ ?= 5632

ead-den-a2-train:
	@mkdir -p "$(EAD_OUT)/adapters" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_train.py \
	  --root "$(ROOT)" \
	  --corpus a2 \
	  --max-seq-length "$(EAD_DEN_A2_MAX_SEQ)" \
	  $(if $(DEVICE),--device $(DEVICE),)

ead-den-a-excl-train:
	@mkdir -p "$(EAD_OUT)/adapters" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_train.py \
	  --root "$(ROOT)" \
	  --corpus all \
	  --variant excl \
	  $(if $(DEVICE),--device $(DEVICE),)

ead-den-a-full-train:
	@mkdir -p "$(EAD_OUT)/adapters" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_train.py \
	  --root "$(ROOT)" \
	  --corpus all \
	  --variant full \
	  $(if $(DEVICE),--device $(DEVICE),)

ead-den-a-excl-eval:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/scores"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_eval.py \
	  --root "$(ROOT)" \
	  --variant excl \
	  --device "$(or $(DEVICE),cuda)"

ead-den-a-excl-ablation:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_ablation.py \
	  --root "$(ROOT)" \
	  --variant excl \
	  --device "$(or $(DEVICE),cuda)"

ead-den-a-full-score:
	@mkdir -p "$(EAD_OUT)/scores"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/den_score.py \
	  --adapter-dir "$(EAD_OUT)/adapters/ead-den-a-full" \
	  --fit-train-jsonl "$(EAD_OUT)/work/ead-den-a-full-train.jsonl" \
	  --score-jsonl "$(ROOT)data/edit_sft_all/train.jsonl" \
	  --out "$(EAD_OUT)/scores/ead-den-a-full.jsonl" \
	  --device "$(or $(DEVICE),cuda)"

ead-den-a-excl: ead-den-a-excl-train ead-den-a-excl-eval
	@echo "ead Phase 2 excl complete -> $(EAD_OUT)/adapters/ead-den-a-excl/ $(EAD_OUT)/reports/"

ead-gate-defect:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/gate_defect.py --root "$(ROOT)"

ead-gate-level:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work" "$(EAD_OUT)/adapters/ead-gate-level"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/gate_level.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)" \
	  --train-device "$(or $(TRAIN_DEVICE),cpu)" \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) \
	  $(if $(REFRESH_GATE_FEATURES),--refresh-features,)

ead-gate-level-r7:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work" "$(EAD_OUT)/adapters/ead-gate-level-r7"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/gate_level.py \
	  --root "$(ROOT)" \
	  --input-mode r7 \
	  --report-stem ead-gate-level-r7 \
	  --skip-learning-curve \
	  --device "$(or $(DEVICE),cuda)" \
	  --train-device "$(or $(TRAIN_DEVICE),cpu)" \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) \
	  $(if $(REFRESH_GATE_FEATURES),--refresh-features,)

ead-gate-level-ablation:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work" "$(EAD_OUT)/adapters"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/gate_level_ablation.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)" \
	  --train-device "$(or $(TRAIN_DEVICE),cpu)" \
	  $(if $(EPOCHS),--epochs $(EPOCHS),)

ead-h1-d-selfgen:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/h1_d_selfgen.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)" \
	  $(if $(REFRESH_H1_GENERATIONS),--refresh-generations,)

ead-h1-c-measure:
	@mkdir -p "$(EAD_OUT)/reports" "$(EAD_OUT)/work"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" $(PYTHON) scripts/ead/h1_c_measure.py \
	  --root "$(ROOT)" \
	  --device "$(or $(DEVICE),cuda)"

ead-r5-h1: ead-h1-d-selfgen ead-h1-c-measure
	@echo "R5 H1 complete -> $(EAD_OUT)/reports/ead-h1-d-selfgen.md $(EAD_OUT)/reports/ead-h1-c-measure.md"

ead-readability-probe:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTHONPATH=scripts $(PYTHON) scripts/ead/readability_probe.py --root "$(ROOT)"

ead-ambiguity-probe:
	@mkdir -p "$(EAD_OUT)/reports"
	PYTHONPATH=scripts $(PYTHON) scripts/ead/ambiguity_probe.py --root "$(ROOT)"

EAD_DATA := $(EAD_OUT)/data
EAD_ADAPTER ?= outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter

ead-build-e1:
	@mkdir -p "$(EAD_DATA)"
	PYTHONPATH=scripts $(PYTHON) scripts/ead/build_e_data.py --root "$(ROOT)" --data-dir "$(EAD_DATA)" --e1-only

ead-import-e2-from-c:
	@mkdir -p "$(EAD_DATA)"
	PYTHONPATH=scripts $(PYTHON) scripts/ead/build_e_data.py \
	  --root "$(ROOT)" \
	  --data-dir "$(EAD_DATA)" \
	  --build-e2-from-c \
	  --adapter "$(EAD_ADAPTER)"

ead-generate-e3:
	@mkdir -p "$(EAD_DATA)"
	PYTORCH_ALLOC_CONF="$(PYTORCH_ALLOC_CONF)" PYTHONPATH=scripts $(PYTHON) scripts/ead/generate_e_qwen.py \
	  --root "$(ROOT)" \
	  --mode base \
	  --out "$(EAD_DATA)/e3.jsonl" \
	  --device "$(or $(DEVICE),cuda)" \
	  $(if $(LIMIT),--limit $(LIMIT),)

ead-build-e: ead-build-e1
	@test -s "$(EAD_DATA)/e2.jsonl" || (echo "missing $(EAD_DATA)/e2.jsonl — run make ead-import-e2-from-c" && exit 1)
	@test -s "$(EAD_DATA)/e3.jsonl" || (echo "missing $(EAD_DATA)/e3.jsonl — run make ead-generate-e3" && exit 1)
	PYTHONPATH=scripts $(PYTHON) scripts/ead/build_e_data.py --root "$(ROOT)" --data-dir "$(EAD_DATA)"

ead-validate-e:
	PYTHONPATH=scripts $(PYTHON) scripts/ead/validate_e_data.py --root "$(ROOT)" --data-dir "$(EAD_DATA)"

ead-e-pipeline: ead-build-e1 ead-import-e2-from-c ead-generate-e3 ead-build-e ead-validate-e

ead-p3: ead-gate-defect ead-gate-level
	@echo "ead Phase 3 complete -> $(EAD_OUT)/reports/ead-gate-defect.md $(EAD_OUT)/reports/ead-gate-level.md"

ead-calibrate-tau:
	@test -n "$(GATE_LEVEL_CHECKPOINT)" || (echo "GATE_LEVEL_CHECKPOINT=<path> is required" && exit 1)
	@test -n "$(GATE_LEVEL_PREDICTIONS)" || (echo "GATE_LEVEL_PREDICTIONS=<path> is required" && exit 1)
	$(PYTHON) scripts/ead/compose.py calibrate-tau \
	  --root "$(ROOT)" \
	  --gate-level-checkpoint "$(GATE_LEVEL_CHECKPOINT)" \
	  --predictions "$(GATE_LEVEL_PREDICTIONS)" \
	  $(if $(DEN_ADAPTER_CHECKPOINT),--den-adapter-checkpoint "$(DEN_ADAPTER_CHECKPOINT)",)
