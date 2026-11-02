LRS ?= 1e-6 3e-6 1e-5
MODELS ?= HuggingFaceTB/SmolLM2-135M-Instruct Qwen/Qwen2.5-0.5B-Instruct
ARM ?= ef
SEED ?= 0
EVAL ?= 20
PY ?= python3
KL_N ?= 100

test:
	$(PY) test_quantlag.py

# Phase 1: KL at step 0 is the same for every lr, so it is measured once per model and variant
lag:
	mkdir -p results/logs
	for m in $(MODELS); do for v in $(VARIANTS); do for lr in $(LRS); do \
	  kl=50; [ $$lr = 1e-6 ] && kl=0,50; \
	  $(PY) -m quantlag.lag --model $$m --variant $$v --lr $$lr --kl-at $$kl --kl-n $(KL_N) 2>&1 | grep --line-buffered -v -i warn \
	    | tee results/logs/lag_$$(basename $$m)_$${v}_lr$$lr.log || exit 1; done; done; done

# Phase 2: one GRPO run, e.g. make repro ARM=ef SEED=0 (EVAL=0,20 adds the step-0 eval)
repro:
	mkdir -p results/logs
	$(PY) -m quantlag.grpo --arm $(ARM) --seed $(SEED) --lr $(P2_LR) --steps $(P2_STEPS) --eval-at $(EVAL) 2>&1 | grep --line-buffered -v -i warn \
	  | tee results/logs/grpo_$(ARM)_s$(SEED).log

plots:
	$(PY) scripts/plots.py

.PHONY: test lag repro plots

# Phase 2 queue as run for the committed results: seed-major so a cut-off run stays balanced.
P2_LR ?= 3e-6
VARIANTS ?= plain qurl
P2_STEPS ?= 20
P2_EVAL_N ?= 500
phase2:
	mkdir -p results/logs
	for s in 0 1 2; do for arm in rtn ef qurl $$( [ $$s = 0 ] && echo fp32 ); do \
	  [ -f results/grpo_$${arm}_int8_s$$s.json ] && grep -q wall_s results/grpo_$${arm}_int8_s$$s.json && continue; \
	  $(PY) -m quantlag.grpo --arm $$arm --seed $$s --lr $(P2_LR) --steps $(P2_STEPS) --eval-at $(P2_STEPS) --eval-n $(P2_EVAL_N) 2>&1 \
	    | grep --line-buffered -v -i warn | tee results/logs/grpo_$${arm}_s$$s.log || exit 1; done; done
	for arm in fp32 rtn qurl; do \
	  $(PY) -m quantlag.grpo --arm $$arm --seed 0 --steps 0 --eval-at 0 --eval-n $(P2_EVAL_N) --out results/grpo_step0_$$arm.json 2>&1 \
	    | grep --line-buffered -v -i warn | tee results/logs/grpo_step0_$$arm.log; done

# Qwen phase 1 as run for the committed results: one process per grid (and the QuRL variant on its own)
# so a replay fits in memory next to a phase 2 run. Same seed and data order, so every process replays
# the same AdamW trajectory (the logged losses match).
QWEN ?= Qwen/Qwen2.5-0.5B-Instruct
QWEN_STEPS ?= 30
lag-split:
	mkdir -p results/logs
	for lr in $(LRS); do kl=$(QWEN_STEPS); [ $$lr = 1e-6 ] && kl=0,$(QWEN_STEPS); \
	  for g in bf16 int8 int4; do \
	    $(PY) -m quantlag.lag --model $(QWEN) --variant plain --grids $$g --micro 1 --steps $(QWEN_STEPS) --lr $$lr --kl-at $$kl --kl-n $(KL_N) 2>&1 \
	      | grep --line-buffered -v -i warn | tee results/logs/lag_$$(basename $(QWEN))_plain_lr$${lr}_$$g.log || exit 1; done; \
	  $(PY) -m quantlag.lag --model $(QWEN) --variant qurl --micro 1 --steps $(QWEN_STEPS) --lr $$lr --kl-at $$kl --kl-n $(KL_N) 2>&1 \
	    | grep --line-buffered -v -i warn | tee results/logs/lag_$$(basename $(QWEN))_qurl_lr$$lr.log || exit 1; done
