"""Export fp32 master weights into a served copy on a bf16, int8 or int4 grid.

Rules: rtn (round to nearest), sr (stochastic rounding), ef (error feedback:
the rounding residual of each weight is carried into the next sync).
Integer scales are computed once from the master and then frozen.
QuRL's invariant scaling is a one-time reparametrisation (`invariant_scale`)
followed by rtn export.
"""
import torch

GRIDS = ("bf16", "int8", "int4")
RULES = ("rtn", "sr", "ef")
QMAX = {"int8": 127, "int4": 7}
GROUP = 32  # int4 group size along the input dimension


def _groups(w, grid):
    return w.reshape(w.shape[0], -1, GROUP) if grid == "int4" else w


def frozen_scale(w, grid):
    """Symmetric absmax scale: per output channel (int8) or per group of 32 (int4)."""
    if grid == "bf16":
        return None
    a = _groups(w, grid).abs().amax(-1, keepdim=True)
    return (a / QMAX[grid]).clamp_min(1e-12)


def _round(x, rule, gen):
    if rule == "sr":
        return torch.floor(x + torch.rand(x.shape, generator=gen, dtype=x.dtype))
    return torch.round(x)


def _bf16(x, rule, gen):
    if rule != "sr":
        return x.to(torch.bfloat16)
    # add uniform noise to the 16 bits bf16 drops, then truncate: unbiased in magnitude, so for both signs
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, generator=gen, dtype=torch.int32)
    out = ((bits + noise) & -65536).view(torch.float32)  # ponytail: no inf/nan guard; weights stay far from bf16 max
    return out.to(torch.bfloat16)


def quantize(x, grid, scale, rule="rtn", gen=None):
    """Codes of x on the grid (int8 tensor for integer grids, bf16 tensor for bf16)."""
    if grid == "bf16":
        return _bf16(x, rule, gen)
    q = _round(_groups(x, grid) / scale, rule, gen)
    return q.clamp_(-QMAX[grid], QMAX[grid]).to(torch.int8).reshape(x.shape)


def dequant(codes, grid, scale):
    if grid == "bf16":
        return codes.float()
    return (_groups(codes.float(), grid) * scale).reshape(codes.shape)


class Served:
    """A served (rollout) copy of one master tensor, starting from its round-to-nearest snapshot.

    At each sync:
      rtn: re-quantize the master to the nearest grid point (the usual export).
      sr:  add the master's update since the last sync with stochastic rounding,
           so E[served] moves by exactly the update (low churn on any grid).
      ef:  error feedback (Fair Cast) on the update: round x + residual, where
           x = served0 + (master - master0) and residual = what the last rounding
           lost (|residual| <= half a step), so the time average of the served copy
           follows x. On bf16 x is the master itself. On int grids the shift by the
           snapshot's quantization error keeps the copy from paying that error back
           by dithering (unshifted, about half the int8 weights flip every sync).
    """

    def __init__(self, master, grid, rule, seed=0):
        assert grid in GRIDS and rule in RULES
        self.grid, self.rule = grid, rule
        self.scale = frozen_scale(master, grid)
        self.gen = torch.Generator().manual_seed(seed)
        self.codes = quantize(master, grid, self.scale)
        self.residual = torch.zeros_like(master) if rule == "ef" else None
        self.offset = self.weight() - master if rule == "ef" and grid != "bf16" else 0

    def sync(self, master, delta=None):
        """Export after a master update; `delta` (master minus master at the last sync) is used by sr."""
        if self.rule == "rtn":
            codes = quantize(master, self.grid, self.scale)
        elif self.rule == "sr":
            codes = quantize(self.weight() + delta, self.grid, self.scale, "sr", self.gen)
        else:
            target = master + self.offset + self.residual
            codes = quantize(target, self.grid, self.scale)
            self.residual = target - dequant(codes, self.grid, self.scale)
            if self.scale is not None:  # no windup where the frozen absmax clamps: keep |residual| <= half a step
                h = 0.5 * self.scale
                self.residual = torch.max(torch.min(_groups(self.residual, self.grid), h), -h).reshape(master.shape)
        changed = int((codes != self.codes).sum())
        self.codes = codes
        return changed

    def weight(self):
        return dequant(self.codes, self.grid, self.scale)


def snapshot(w):
    """Copy of w, stored as bf16 when that is lossless (bf16 checkpoints), to halve its memory."""
    b = w.detach().to(torch.bfloat16)
    return b if torch.equal(b.float(), w) else w.detach().clone()


def quantized_names(model):
    """The 2-D linear weights inside transformer blocks: the tensors a CPU int8/int4 runtime quantizes.
    ponytail: embeddings/lm_head, norms and biases are served in fp32."""
    return [n for n, p in model.named_parameters() if ".layers." in n and p.dim() == 2]


@torch.no_grad()
def invariant_scale(model, s=1.5):
    """QuRL update-aware quantization: W X = (W/s)(s X), with s absorbed into the preceding RMSNorm.
    Applied to q/k/v (after input_layernorm) and gate/up (after post_attention_layernorm)."""
    for layer in model.model.layers:
        layer.input_layernorm.weight.mul_(s)
        layer.post_attention_layernorm.weight.mul_(s)
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj):
            lin.weight.div_(s)
