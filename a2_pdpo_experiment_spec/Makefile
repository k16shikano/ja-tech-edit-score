CONFIG := config/experiment.yaml
PYTHON := python3
SCRIPTS := scripts

.PHONY: help a2-import a2-inspect a2-split a2-generate-train a2-generate-dev \
	a2-preferences a2-length-diag a2-precompute-ref a2-train-pdpo \
	a2-evaluate a2-blind-ab a2-data-pipeline

help:
	@echo "A2 P-DPO experiment targets"
	@echo "  make a2-inspect          Phase 0: import + validate A2"
	@echo "  make a2-split            Phase 1: fixed split manifest"
	@echo "  make a2-generate-train   Phase 2: generic revisions (train)"
	@echo "  make a2-generate-dev     Phase 2: generic revisions (dev)"
	@echo "  make a2-preferences      Phase 3: preference dataset"
	@echo "  make a2-length-diag      Phase 4: length leakage diagnosis"
	@echo "  make a2-precompute-ref    Phase 5: reference log-probs"
	@echo "  make a2-train-pdpo       Phase 6: P-DPO training"
	@echo "  make a2-evaluate         Phase 7: test evaluation"
	@echo "  make a2-blind-ab         Phase 8: blind A/B files"
	@echo "  make a2-data-pipeline    Phases 0-3 (no GPU generation)"

a2-import:
	$(PYTHON) $(SCRIPTS)/inspect_a2.py --config $(CONFIG) --import

a2-inspect:
	$(PYTHON) $(SCRIPTS)/inspect_a2.py --config $(CONFIG)

a2-split:
	$(PYTHON) $(SCRIPTS)/split_a2.py --config $(CONFIG)

a2-generate-train:
	$(PYTHON) $(SCRIPTS)/generate_generic.py --config $(CONFIG) --split train

a2-generate-dev:
	$(PYTHON) $(SCRIPTS)/generate_generic.py --config $(CONFIG) --split dev

a2-preferences:
	$(PYTHON) $(SCRIPTS)/build_preferences.py --config $(CONFIG)

a2-length-diag:
	$(PYTHON) $(SCRIPTS)/diagnose_length_leakage.py --config $(CONFIG)

a2-precompute-ref:
	$(PYTHON) $(SCRIPTS)/precompute_reference_logps.py --config $(CONFIG)

a2-train-pdpo:
	$(PYTHON) $(SCRIPTS)/train_pdpo.py --config $(CONFIG)

a2-evaluate:
	$(PYTHON) $(SCRIPTS)/evaluate_preferences.py --config $(CONFIG)

a2-blind-ab:
	$(PYTHON) $(SCRIPTS)/generate_blind_ab.py --config $(CONFIG)

a2-data-pipeline: a2-inspect a2-split
