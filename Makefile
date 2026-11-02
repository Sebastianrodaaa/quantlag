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

# Qwen phase 1 as run for the committed results: train once per lr and variant, saving the master after
# every step (`--dump`), then replay that trajectory through one grid at a time (`--replay`), so the
# replay fits in memory next to a phase 2 run. Finished pieces are skipped, so it can be rerun after a kill.
QWEN ?= Qwen/Qwen2.5-0.5B-Instruct
QWEN_STEPS ?= 30
LAG_Q = $(PY) -m quantlag.lag --model $(QWEN) --micro 1 --steps $(QWEN_STEPS) --kl-n $(KL_N)
lag-split:
	mkdir -p results/logs
	for v in plain qurl; do for lr in $(LRS); do kl=$(QWEN_STEPS); [ $$lr = 1e-6 ] && kl=0,$(QWEN_STEPS); \
	  d=results/traj_$${v}_lr$$lr; [ $$v = plain ] && gs="bf16 int8 int4" || gs="bf16,int8,int4"; \
	  for g in $$gs; do o=results/lag_$$(basename $(QWEN))_$${v}_lr$${lr}_$$(echo $$g | tr , -).json; \
	    grep -qs wall_s $$o && continue; \
	    grep -qs wall_s $$d/train.json || $(LAG_Q) --variant $$v --lr $$lr --grids "" --kl-at "" --dump $$d 2>&1 \
	      | grep --line-buffered -v -i warn > results/logs/lag_$$(basename $(QWEN))_$${v}_lr$${lr}_train.log; \
	    grep -qs wall_s $$d/train.json || exit 1; \
	    $(LAG_Q) --variant $$v --lr $$lr --grids $$g --kl-at $$kl --replay $$d --out $$o 2>&1 \
	      | grep --line-buffered -v -i warn > results/logs/lag_$$(basename $(QWEN))_$${v}_lr$${lr}_$$(echo $$g | tr , -).log; \
	  done; [ $$(grep -ls wall_s results/lag_$$(basename $(QWEN))_$${v}_lr$${lr}_*.json | wc -l) = $$(echo $$gs | wc -w) ] && rm -rf $$d; done; done
