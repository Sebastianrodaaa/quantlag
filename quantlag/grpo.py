"""Plain PyTorch GRPO on GSM8K on CPU, rollouts from a served (quantized) copy.

The loss is computed on the served copy with a straight-through trick
(w_master + (w_served - w_master).detach()), so the sampled tokens and the
loss come from the same weights and gradients land on the fp32 master.
"""
import argparse, json, math, os, random, re, time, platform
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantlag.export import Served, quantize, dequant, quantized_names, invariant_scale, snapshot

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ARMS = {"fp32": None, "rtn": "rtn", "sr": "sr", "ef": "ef", "qurl": "rtn"}


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def extract(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = m[-1] if m else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    n = re.findall(r"-?\d+\.?\d*", s)
    return n[-1] if n else s


def correct(text, g):
    try:
        return float(extract(text)) == float(g)
    except ValueError:
        return False


def prompts(tok, questions):
    return [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False, add_generation_prompt=True) for q in questions]


def cpu_name():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip() + f" x{os.cpu_count()}"
    return platform.processor()


class Policy:
    """fp32 master + served copy (int8/int4/bf16 grid) of the block linear weights."""

    def __init__(self, model_id, arm, grid="int8", seed=0, qurl_s=1.5):
        self.master = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        if arm == "qurl":
            invariant_scale(self.master, qurl_s)
        self.rule = ARMS[arm]
        self.names = quantized_names(self.master) if self.rule else []
        params = dict(self.master.named_parameters())
        if self.rule is None:
            self.served_model = self.master
            return
        self.served_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self.served_model.load_state_dict(self.master.state_dict())
        self.served_model.requires_grad_(False)
        self.sp = dict(self.served_model.named_parameters())
        self.served = {n: Served(params[n].detach(), grid, self.rule, seed=seed + i) for i, n in enumerate(self.names)}
        self.master0 = {n: snapshot(params[n]) for n in self.names}
        # only sr exports the update itself, so only sr needs the master at the last sync
        self.prev = {n: params[n].detach().clone() for n in self.names} if self.rule == "sr" else None
        self._write()

    @torch.no_grad()
    def _write(self):
        for n, s in self.served.items():
            self.sp[n].copy_(s.weight())

    @torch.no_grad()
    def sync(self):
        """Export master -> served; returns lag stats of the served copy (None for fp32)."""
        if not self.rule:
            return None
        params = dict(self.master.named_parameters())
        for n in params:
            if n not in self.served:
                self.sp[n].copy_(params[n])
        changed = total = 0
        num = den = err = 0.0
        for n, s in self.served.items():
            w = params[n].detach()
            changed += s.sync(w, w - self.prev[n] if self.prev else None)
            if self.prev:
                self.prev[n].copy_(w)
            total += params[n].numel()
            dm = params[n] - self.master0[n]
            ds = s.weight() - dequant(quantize(self.master0[n], s.grid, s.scale), s.grid, s.scale)
            num += float((dm * ds).sum()); den += float((dm * dm).sum()); err += float(((ds - dm) ** 2).sum())
        self._write()
        return {"changed_frac": changed / total, "realized": num / max(den, 1e-30), "track_err": math.sqrt(err / max(den, 1e-30))}

    def loss_forward(self, **kw):
        if not self.rule:
            return self.master(**kw)
        params = dict(self.master.named_parameters())
        over = {n: params[n] + (self.sp[n] - params[n]).detach() for n in self.names}
        return functional_call(self.master, over, args=(), kwargs=kw, strict=False)


def generate(model, tok, texts, max_new, sample, seed=None, bs=64):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True)
        if seed is not None:
            torch.manual_seed(seed + i)
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None,
                                 top_p=1.0 if sample else None, top_k=0 if sample else None, pad_token_id=tok.pad_token_id)
        outs.append((enc["input_ids"], enc["attention_mask"], seq[:, enc["input_ids"].shape[1]:]))
    return outs


def evaluate(model, tok, test, n, max_new=256):
    rows = test.select(range(n))
    texts = prompts(tok, rows["question"])
    done = 0
    golds = iter(rows["answer"])
    for _, _, comp in generate(model, tok, texts, max_new, sample=False):
        for c in tok.batch_decode(comp, skip_special_tokens=True):
            done += correct(c, gold(next(golds)))
    return done / n


def completion_mask(comp, eos_ids, pad_id):
    """1 up to and including the first EOS, 0 after."""
    is_end = torch.isin(comp, torch.tensor(eos_ids))
    after = (is_end.cumsum(1) - is_end.long()) > 0
    return (~after).long()


def train_step(pol, opt, tok, batch, n_samples, max_new, seed, micro=2):
    t0 = time.time()
    texts = [p for p in prompts(tok, batch["question"]) for _ in range(n_samples)]
    golds = [gold(a) for a in batch["answer"] for _ in range(n_samples)]
    (ids, am, comp), = generate(pol.served_model, tok, texts, max_new, sample=True, seed=seed, bs=len(texts))
    t_gen = time.time() - t0
    eos = pol.served_model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    cm = completion_mask(comp, eos, tok.pad_token_id)
    rewards = torch.tensor([float(correct(c, g)) for c, g in zip(tok.batch_decode(comp, skip_special_tokens=True), golds)])
    r = rewards.view(-1, n_samples)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    keep = adv.abs() > 1e-6
    t1 = time.time()
    opt.zero_grad(set_to_none=True)
    idx = keep.nonzero().view(-1).tolist()
    for k in range(0, len(idx), micro):
        j = idx[k:k + micro]
        inp = torch.cat([ids[j], comp[j]], 1)
        mask = torch.cat([am[j], cm[j]], 1)
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        # trim right padding shared by the micro-batch
        L = int(mask.sum(0).nonzero().max()) + 1
        P = ids.shape[1]
        # only completion positions go through the lm head (151k vocab logits dominate memory)
        logits = pol.loss_forward(input_ids=inp[:, :L], attention_mask=mask[:, :L], position_ids=pos[:, :L], logits_to_keep=L - P + 1).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, inp[:, P:L, None]).squeeze(-1)
        m = cm[j][:, :L - P].float()
        seq_lp = (lp * m).sum(1) / m.sum(1).clamp_min(1)
        loss = -(adv[j] * seq_lp).sum() / len(texts)
        loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(pol.master.parameters(), 1.0)) if idx else 0.0
    if idx:
        opt.step()
    opt.zero_grad(set_to_none=True)  # free 2 GB of grads before the next rollout
    t_train = time.time() - t1
    t2 = time.time()
    lag = pol.sync()
    return {"reward": float(rewards.mean()), "trained_seqs": len(idx), "resp_len": float(cm.sum(1).float().mean()),
            "grad_norm": gnorm, "t_gen": t_gen, "t_train": t_train, "t_sync": time.time() - t2, **(lag or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", default="ef", choices=list(ARMS))
    ap.add_argument("--grid", default="int8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--eval-at", default="", help="comma-separated steps for held-out eval, e.g. 0,20")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    ds = load_dataset("openai/gsm8k", "main")
    order = random.Random(a.seed).sample(range(len(ds["train"])), a.steps * a.prompts)
    pol = Policy(a.model, a.arm, a.grid, a.seed)
    opt = torch.optim.AdamW(pol.master.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    eval_at = {int(s) for s in a.eval_at.split(",") if s}
    log = {"config": vars(a), "cpu": cpu_name(), "torch": torch.__version__, "threads": torch.get_num_threads(), "steps": [], "evals": []}
    out = a.out or f"results/grpo_{a.arm}_{a.grid}_s{a.seed}.json"

    def save():
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(log, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)

    t_start = time.time()
    for step in range(a.steps + 1):
        if step in eval_at:
            t = time.time()
            acc = evaluate(pol.served_model, tok, ds["test"], a.eval_n, a.max_new)
            log["evals"].append({"step": step, "n": a.eval_n, "acc": acc, "t": time.time() - t})
            print(f"eval step {step}: acc {acc:.3f} ({time.time() - t:.0f}s)", flush=True)
            save()
        if step == a.steps:
            break
        batch = ds["train"].select(order[step * a.prompts:(step + 1) * a.prompts])
        rec = {"step": step + 1, **train_step(pol, opt, tok, batch, a.samples, a.max_new, a.seed * 100003 + step)}
        log["steps"].append(rec)
        print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        save()
    log["wall_s"] = time.time() - t_start
    save()


if __name__ == "__main__":
    main()
