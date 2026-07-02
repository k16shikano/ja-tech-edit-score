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
RAW := $(DATA_DIR)/examples.raw.jsonl
DB := $(DATA_DIR)/examples.db
DPO := $(DATA_DIR)/dpo_dataset.jsonl
DPO_CURATED := $(DATA_DIR)/dpo_curated.jsonl
PREF := $(DATA_DIR)/pref_dataset.jsonl
PREF_SPLIT := $(DATA_DIR)/pref_split

EMBED_MODEL ?= hotchpotch/static-embedding-japanese
TRUNCATE_DIM ?= 256

.PHONY: help venv data train compare check clean-model install-bin install-skills daemon daemon-stop

help:
	@echo "Targets:"
	@echo "  make venv"
	@echo "  make install-bin   # symlink bin/* to ~/.local/bin"
	@echo "  make install-skills # symlink skills/* to ~/.cursor/skills"
	@echo "  make data DIR=<repo> ORG=<base-branch> EDT=<edit-branch> [PROJECT_ID=...] [PATH=...]"
	@echo "  make train"
	@echo "  make check FILE=<path> [BASE=...] [EDIT=...] [FORMAT=markdown|text|json]  # 選好チェック"
	@echo "  make compare SOURCE=... CANDIDATE_A=... CANDIDATE_B=..."
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
	@echo "installed: ~/.local/bin/ja-tech-edit-score-check"
	@echo "installed: ~/.local/bin/ja-tech-edit-score-compare"

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
	  --truncate-dim $(TRUNCATE_DIM)

compare:
	@test -n "$(SOURCE)" || (echo "SOURCE is required" && exit 1)
	@test -n "$(CANDIDATE_A)" || (echo "CANDIDATE_A is required" && exit 1)
	@test -n "$(CANDIDATE_B)" || (echo "CANDIDATE_B is required" && exit 1)
	$(PYTHON) scripts/compare_pref_candidates_static.py \
	  --model "$(OUTPUT_DIR)" \
	  --source-text "$(SOURCE)" \
	  --candidate-a "$(CANDIDATE_A)" \
	  --candidate-b "$(CANDIDATE_B)"

check:
	@test -n "$(FILE)" || (echo "FILE is required" && exit 1)
	$(ROOT)bin/ja-tech-edit-score-check "$(FILE)" \
	  $(if $(BASE),--base "$(BASE)",) \
	  $(if $(EDIT),--edit "$(EDIT)",) \
	  $(if $(FORMAT),--format "$(FORMAT)",--format markdown)

clean-model:
	rm -rf "$(OUTPUT_DIR)"

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
