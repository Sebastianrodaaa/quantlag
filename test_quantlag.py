"""Run: python3 test_quantlag.py  (CPU, downloads SmolLM2-135M-Instruct and GSM8K on first run)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantlag.export import Served, frozen_scale, quantize, dequant, GRIDS
from quantlag import grpo

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
W = model.model.layers[3].mlp.gate_proj.weight.detach().clone()  # a real 1536x576 weight


def step_size(w, grid, scale=None):
    """Local grid spacing for every weight (integer grids: the frozen scale)."""
    if grid == "bf16":
        e = torch.frexp(w)[1]
        return torch.ldexp(torch.ones_like(w), e - 8)
    return dequant(torch.ones_like(w, dtype=torch.int8), grid, frozen_scale(w, grid) if scale is None else scale)


def test_sr_unbiased():
    """Mean of 1,000 stochastic-rounding exports of a real weight tensor is the master."""
    g = torch.Generator().manual_seed(1)
    for grid in GRIDS:
        scale = frozen_scale(W, grid)
        acc = torch.zeros_like(W)
        for _ in range(1000):
            acc += dequant(quantize(W, grid, scale, "sr", g), grid, scale)
        err = (acc / 1000 - W) / step_size(W, grid, scale)
        # per-weight sd of the mean is <= 0.5/sqrt(1000) = 0.016 steps
        assert err.abs().mean() < 0.02 and abs(float(err.mean())) < 0.002, (grid, float(err.abs().mean()))


def test_sr_sync_tracks_small_updates():
    """Sub-half-step updates move the SR served copy by the update in expectation; RTN does not move."""
    for grid in GRIDS:
        sr, rtn = Served(W, grid, "sr", seed=2), Served(W, grid, "rtn")
        base = sr.weight()
        d = 0.1 * step_size(W, grid, sr.scale) * torch.sign(torch.randn(W.shape, generator=torch.Generator().manual_seed(3)))
        w = W.clone()
        for _ in range(3):
            w += d; sr.sync(w, d)
        moved = float(((sr.weight() - base) * d.sign()).mean() / (3 * d.abs()).mean())
        assert 0.9 < moved < 1.1, (grid, moved)


def test_ef_residual_bounded():
    for grid in GRIDS:
        s = Served(W, grid, "ef")
        w = W.clone()
        for t in range(20):
            d = 1e-4 * torch.randn_like(w)
            w += d
            target = w + s.residual
            s.sync(w, d)
            assert (s.residual.abs() <= 0.5 * (1 + 1e-5) * step_size(target, grid, s.scale)).all(), grid  # 1e-5: fp32 rounding


def test_rtn_drops_sub_half_step_update():
    for grid in GRIDS:
        s = Served(W, grid, "rtn")
        before = s.codes.clone()
        dist = (W - s.weight()).abs()  # distance to the served value
        room = 0.5 * step_size(W, grid) - dist  # how far each weight may move before it crosses a rounding boundary
        step = step_size(W, grid, s.scale)
        # skip weights within fp32 noise of a rounding tie (bf16 checkpoints put some exactly on int4 midpoints)
        upd = torch.where(room > 1e-3 * step, 0.5 * room * torch.sign(W - s.weight()), 0.0)
        assert s.sync(W + upd, upd) == 0 and torch.equal(s.codes, before), grid


def test_grpo_step_end_to_end():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    batch = load_dataset("openai/gsm8k", "main")["train"].select(range(4))
    pol = grpo.Policy(MODEL, "ef", "int8", seed=0)
    opt = torch.optim.AdamW(pol.master.parameters(), lr=1e-5)
    rec = grpo.train_step(pol, opt, tok, batch, n_samples=2, max_new=48, seed=0)
    assert 0.0 <= rec["reward"] <= 1.0 and 0.0 <= rec["changed_frac"] <= 1.0
    # the straight-through loss gives master gradients through the served copy
    x = tok(["Natalia sold clips"], return_tensors="pt")
    pol.loss_forward(**x).logits.sum().backward()
    assert pol.master.model.layers[0].mlp.up_proj.weight.grad.abs().sum() > 0


def test_extract():
    assert grpo.correct("so the total is \\boxed{1,234}.", "1234")
    assert grpo.correct("The answer is 72", "72") and not grpo.correct("\\boxed{71}", "72")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name, flush=True)
