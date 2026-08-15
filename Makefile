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
DETECT_MIDDLE_DIR := $(ROOT)outputs/pref-detect-section
DETECT_CD_MIDDLE_DIR := $(ROOT)outputs/pref-detect-cd-section

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

.PHONY: help venv data mine-sections pairsplit-data pref-keep-data train-bt-keep train-bt-keep-pairsplit train-sentseq-keep train-sentseq-keep-pairsplit build-pref-keep-image build-generated-pref-sentseq-image build-section-middle-sentseq-image build-setwise-section-triples-image build-setwise-section-human-top-image build-section-middle-nce-image section-middle-nce build-section-middle-detect-image section-middle-detect build-section-middle-detect-cd-image section-middle-detect-cd build-pref-multigranular-image section-middle-setwise section-middle-setwise-human-top hard-eval-setwise pref-multigranular-smoke pref-multigranular freeze-8d-items build-serve-image select-blind-items build-blind-pairs blind-judge pref-valid-blind-pairs pref-valid-blind-judge analyze-blind-judgments generated-pref-data generated-pref-sentseq-smoke generated-pref-sentseq-cv section-middle-gen section-middle-judge section-middle-triples section-middle-sentseq edit-sft-data edit-sft-review edit-sft-review-hunk edit-sft-review-section-extra edit-sft-export-keeps edit-sft-promote-reviewed edit-sft edit-sft-section-only edit-sft-hunk score-bt rank converge check calibrate-margins revise serve install-bin install-skills daemon daemon-stop test clean-model

help:
	@echo "現行（docs/PLAN.md / BRIEF.md）:"
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

# PLAN 工程3: ペア単位分割の学習側で BT を学び直す（旧 pref-bt-keep は上書きしない）
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
