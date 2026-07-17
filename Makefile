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
RAW := $(DATA_DIR)/examples.raw.jsonl
DB := $(DATA_DIR)/examples.db
DPO := $(DATA_DIR)/dpo_dataset.jsonl
DPO_CURATED := $(DATA_DIR)/dpo_curated.jsonl
PREF := $(DATA_DIR)/pref_dataset.jsonl
PREF_SPLIT := $(DATA_DIR)/pref_split

EMBED_MODEL ?= cl-nagoya/ruri-v3-30m
TRUNCATE_DIM ?= 0
# Ruri 文書プレフィックスは末尾空白が必要
null :=
space := $(null) #
TEXT_PREFIX ?= 文章:$(space)
MAX_SEQ_LENGTH ?= 512
BATCH_SIZE ?= 32

REVISION_PAIRS := $(DATA_DIR)/revision_pairs.jsonl
STEERING_MODEL ?=
STEERING_DEVICE ?= cuda
STEERING_LIMIT ?= 0
STEERING_BATCH_SIZE ?= 1
STEERING_MAX_LENGTH ?= 2048

CE_OUTPUT_DIR := $(ROOT)outputs/pref-ce
CE_BASE_MODEL ?= sbintuitions/modernbert-ja-130m

.PHONY: help venv data train train-bt train-ce eval-xproject eval-bt-xproject eval-ce-xproject compare score-bt rank converge check clean-model install-bin install-skills daemon daemon-stop steering-pairs steering-extract steering-probe edit-sft-data edit-sft edit-sft-score

help:
	@echo "Targets:"
	@echo "  make venv"
	@echo "  make install-bin   # symlink bin/* to ~/.local/bin"
	@echo "  make install-skills # symlink skills/* to ~/.cursor/skills"
	@echo "  make data DIR=<repo> ORG=<base-branch> EDT=<edit-branch> [PROJECT_ID=...] [PATH=...]"
	@echo "  make train         # pref-static（既定: ruri-v3-30m）"
	@echo "  make train-bt      # Bradley-Terry 報酬モデル（絶対スコア）"
	@echo "  make eval-xproject # leave-one-project-out（ペア分類）"
	@echo "  make eval-bt-xproject # leave-one-project-out（BT 報酬）"
	@echo "  make train-ce      # 段階2b: cross-encoder 報酬（GPU 推奨、DOK 可）"
	@echo "  make eval-ce-xproject [ONLY_PROJECTS=a,b] # LOPO（cross-encoder、GPU）"
	@echo "  make check FILE=<path> [BASE=...] [EDIT=...] [FORMAT=markdown|text|json]  # 選好チェック"
	@echo "  make compare SOURCE=... CANDIDATE_A=... CANDIDATE_B=..."
	@echo "  make score-bt SOURCE=... CANDIDATE=...  # BT 絶対スコア"
	@echo "  make rank SOURCE=... CANDIDATE_FILES='a.txt b.txt'  # Best-of-N"
	@echo "  make converge CURRENT=... REVISED=... [MODE=pair|bt]  # 収束判定"
	@echo "  make edit-sft-data  # 系統1フェーズ0: chat SFT データ書き出し"
	@echo "  make edit-sft MODEL=<hf-id> [LIMIT=0] [EPOCHS=2]  # 系統1フェーズ1: QLoRA SFT（GPU）"
	@echo "  make edit-sft-score  # 系統1評価: DOK 生成結果を BT 採点（CPU）"
	@echo "  make steering-pairs   # 系統3: draft/revised 対照ペア書き出し"
	@echo "  make steering-extract MODEL=<hf-id> [DEVICE=cuda] [LIMIT=0] [STEERING_PROMPT_MODE=none|reading|norms]  # 層活性抽出（GPU）"
	@echo "  make steering-probe MODEL=<hf-id> [VARIANT=reading|norms]  # LOPO 線形プローブ"
	@echo "  make daemon        # keep model loaded (auto-started by check otherwise)"
	@echo "  make daemon-stop"
	@echo "  make clean-model   # remove outputs/pref-static (before retrain)"



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

data:
	@test -n "$(DIR)" || (echo "DIR is required" && exit 1)
	@test -n "$(ORG)" || (echo "ORG is required" && exit 1)
	@test -n "$(EDT)" || (echo "EDT is required" && exit 1)
	$(PYTHON) scripts/mine_branch_pair.py \
	  --repo "$(DIR)" \
	  --base "$(ORG)" \
	  --edit "$(EDT)" \
	  $(if $(PROJECT_ID),--project-id "$(PROJECT_ID)",) \
	  $(if $(PATH),--path "$(PATH)",) \
	  --append "$(RAW)"

train: $(RAW)
	@test -s "$(RAW)" || (echo "no training data: run make data first" && exit 1)
	$(PYTHON) scripts/import_examples.py --input "$(RAW)" --db "$(DB)"
	$(PYTHON) scripts/build_dpo_dataset.py --db "$(DB)" --out "$(DPO)" --accepted-only
	$(PYTHON) scripts/curate_dpo_dataset.py \
	  --input "$(DPO)" \
	  --out "$(DPO_CURATED)" \
	  --drop-citation-only \
	  --report "$(DATA_DIR)/curate_report.json"
	$(PYTHON) scripts/build_pref_dataset.py \
	  --input "$(DPO_CURATED)" \
	  --out "$(PREF)" \
	  --augment-swap
	$(PYTHON) scripts/split_pref_dataset.py \
	  --input "$(PREF)" \
	  --out-dir "$(PREF_SPLIT)" \
	  --group-by base_id
	$(PYTHON) scripts/train_pref_static.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_SPLIT)/valid.jsonl" \
	  --output-dir "$(OUTPUT_DIR)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE)

train-bt: $(PREF_SPLIT)/train.jsonl $(PREF_SPLIT)/valid.jsonl
	@test -s "$(PREF_SPLIT)/train.jsonl" || (echo "no split: run make train first" && exit 1)
	$(PYTHON) scripts/train_pref_bt.py \
	  --model "$(EMBED_MODEL)" \
	  --train-file "$(PREF_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_SPLIT)/valid.jsonl" \
	  --output-dir "$(BT_OUTPUT_DIR)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE)

train-ce: $(PREF_SPLIT)/train.jsonl $(PREF_SPLIT)/valid.jsonl
	@test -s "$(PREF_SPLIT)/train.jsonl" || (echo "no split: run make train first" && exit 1)
	$(PYTHON) scripts/train_pref_ce.py \
	  --base-model "$(CE_BASE_MODEL)" \
	  --train-file "$(PREF_SPLIT)/train.jsonl" \
	  --eval-file "$(PREF_SPLIT)/valid.jsonl" \
	  --output-dir "$(CE_OUTPUT_DIR)" \
	  $(if $(CE_EPOCHS),--epochs $(CE_EPOCHS),) \
	  $(if $(CE_BATCH_SIZE),--batch-size $(CE_BATCH_SIZE),) \
	  $(if $(CE_LR),--lr $(CE_LR),)

eval-ce-xproject: $(PREF)
	@test -s "$(PREF)" || (echo "no pref dataset: run make train first" && exit 1)
	$(PYTHON) scripts/eval_pref_ce_xproject.py \
	  --input "$(PREF)" \
	  --base-model "$(CE_BASE_MODEL)" \
	  $(if $(CE_EPOCHS),--epochs $(CE_EPOCHS),) \
	  $(if $(CE_BATCH_SIZE),--batch-size $(CE_BATCH_SIZE),) \
	  $(if $(CE_LR),--lr $(CE_LR),) \
	  $(if $(ONLY_PROJECTS),--only-projects "$(ONLY_PROJECTS)",) \
	  --report "$(or $(REPORT),$(ROOT)outputs/eval_ce_xproject.json)"

eval-xproject: $(PREF)
	@test -s "$(PREF)" || (echo "no pref dataset: run make train first" && exit 1)
	$(PYTHON) scripts/eval_pref_xproject.py \
	  --input "$(PREF)" \
	  --model "$(EMBED_MODEL)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --report "$(or $(REPORT),$(ROOT)outputs/eval_xproject.json)"

eval-bt-xproject: $(PREF)
	@test -s "$(PREF)" || (echo "no pref dataset: run make train first" && exit 1)
	$(PYTHON) scripts/eval_pref_bt_xproject.py \
	  --input "$(PREF)" \
	  --model "$(EMBED_MODEL)" \
	  --truncate-dim $(TRUNCATE_DIM) \
	  --text-prefix "$(TEXT_PREFIX)" \
	  --max-seq-length $(MAX_SEQ_LENGTH) \
	  --batch-size $(BATCH_SIZE) \
	  --report "$(or $(REPORT),$(ROOT)outputs/eval_bt_xproject.json)"

compare:
	@test -n "$(SOURCE)" || (echo "SOURCE is required" && exit 1)
	@test -n "$(CANDIDATE_A)" || (echo "CANDIDATE_A is required" && exit 1)
	@test -n "$(CANDIDATE_B)" || (echo "CANDIDATE_B is required" && exit 1)
	$(PYTHON) scripts/compare_pref_candidates_static.py \
	  --model "$(OUTPUT_DIR)" \
	  --source-text "$(SOURCE)" \
	  --candidate-a "$(CANDIDATE_A)" \
	  --candidate-b "$(CANDIDATE_B)"

score-bt:
	@test -n "$(SOURCE)" || (echo "SOURCE is required" && exit 1)
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE is required" && exit 1)
	$(PYTHON) scripts/score_pref_bt.py \
	  --model "$(BT_OUTPUT_DIR)" \
	  --source-text "$(SOURCE)" \
	  --candidate-text "$(CANDIDATE)"

rank:
	@test -n "$(SOURCE)" || (echo "SOURCE is required (text or use SOURCE_FILE=)" && exit 1)
	@test -n "$(CANDIDATE_FILES)$(CANDIDATES_DIR)" || (echo "CANDIDATE_FILES or CANDIDATES_DIR is required" && exit 1)
	$(PYTHON) scripts/rank_pref_bt.py \
	  --model "$(BT_OUTPUT_DIR)" \
	  $(if $(SOURCE_FILE),--source-file "$(SOURCE_FILE)",--source-text "$(SOURCE)") \
	  $(foreach f,$(CANDIDATE_FILES),--candidate-file "$(f)") \
	  $(if $(CANDIDATES_DIR),--candidates-dir "$(CANDIDATES_DIR)",) \
	  $(if $(MIN_MARGIN),--min-margin $(MIN_MARGIN),) \
	  --format $(or $(FORMAT),text)

converge:
	@test -n "$(CURRENT)$(CURRENT_FILE)" || (echo "CURRENT or CURRENT_FILE is required" && exit 1)
	@test -n "$(REVISED)$(REVISED_FILE)" || (echo "REVISED or REVISED_FILE is required" && exit 1)
	$(PYTHON) scripts/check_convergence.py \
	  --mode $(or $(MODE),pair) \
	  --static-model "$(OUTPUT_DIR)" \
	  --bt-model "$(BT_OUTPUT_DIR)" \
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

edit-sft-data: $(REVISION_PAIRS)
	@test -s "$(REVISION_PAIRS)" || (echo "run make steering-pairs first" && exit 1)
	$(PYTHON) scripts/export_edit_sft.py --pairs "$(REVISION_PAIRS)"

edit-sft:
	@test -n "$(MODEL)" || (echo "MODEL=<hf-id> is required" && exit 1)
	@test -s "$(DATA_DIR)/edit_sft/train.jsonl" || (echo "run make edit-sft-data first" && exit 1)
	$(PYTHON) scripts/train_edit_sft.py \
	  --train "$(DATA_DIR)/edit_sft/train.jsonl" \
	  --model "$(MODEL)" \
	  --epochs $(or $(EPOCHS),2) \
	  $(if $(filter-out 0,$(or $(LIMIT),0)),--limit $(LIMIT),) \
	  $(if $(TRUST_REMOTE_CODE),--trust-remote-code,)

edit-sft-score:
	@test -s "$(ROOT)outputs/edit-sft-eval/adapter.jsonl" || (echo "missing outputs/edit-sft-eval/adapter.jsonl (DOK eval artifacts)" && exit 1)
	@test -s "$(ROOT)outputs/edit-sft-eval/base_norms.jsonl" || (echo "missing outputs/edit-sft-eval/base_norms.jsonl" && exit 1)
	@test -d "$(BT_OUTPUT_DIR)" || (echo "missing $(BT_OUTPUT_DIR): run make train-bt" && exit 1)
	$(PYTHON) scripts/score_edit_sft_eval.py \
	  --eval-dir "$(ROOT)outputs/edit-sft-eval" \
	  --bt-model "$(BT_OUTPUT_DIR)"

steering-pairs: $(DPO_CURATED)
	@test -s "$(DPO_CURATED)" || (echo "missing $(DPO_CURATED): run make train pipeline first" && exit 1)
	$(PYTHON) scripts/export_revision_pairs.py \
	  --input "$(DPO_CURATED)" \
	  --out "$(REVISION_PAIRS)" \
	  $(if $(filter-out 0,$(STEERING_LIMIT)),--limit $(STEERING_LIMIT),)

steering-extract: $(REVISION_PAIRS)
	@test -n "$(STEERING_MODEL)$(MODEL)" || (echo "MODEL=<hf-id> is required" && exit 1)
	@test -s "$(REVISION_PAIRS)" || (echo "run make steering-pairs first" && exit 1)
	$(PYTHON) scripts/extract_revision_activations.py \
	  --pairs "$(REVISION_PAIRS)" \
	  --model "$(or $(MODEL),$(STEERING_MODEL))" \
	  --device "$(STEERING_DEVICE)" \
	  --batch-size $(STEERING_BATCH_SIZE) \
	  --max-length $(STEERING_MAX_LENGTH) \
	  $(if $(filter-out 0,$(STEERING_LIMIT)),--limit $(STEERING_LIMIT),) \
	  $(if $(STEERING_PROMPT_MODE),--prompt-mode "$(STEERING_PROMPT_MODE)",) \
	  $(if $(TRUST_REMOTE_CODE),--trust-remote-code,)

steering-probe:
	@test -n "$(STEERING_MODEL)$(MODEL)" || (echo "MODEL=<hf-id> is required (to locate outputs/steering/<slug>)" && exit 1)
	$(PYTHON) scripts/probe_revision_activations.py \
	  --model "$(or $(MODEL),$(STEERING_MODEL))" \
	  $(if $(VARIANT),--variant "$(VARIANT)",) \
	  --activations-dir "$(ROOT)outputs/steering"

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
