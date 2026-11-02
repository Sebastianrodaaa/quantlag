"""Phase 1: replay real AdamW SFT updates on GSM8K and measure how much of each
update reaches the served copy, per grid (bf16/int8/int4) and export rule.

The master trains on its own loss (a replay of real updates); every export
rule watches the same master trajectory. `--variant qurl` reparametrises the
model with QuRL invariant scaling first and exports with round-to-nearest.
"""
import argparse, json, math, os, random, re, time
import torch
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from quantlag.export import GRIDS, Served, quantize, dequant, quantized_names, invariant_scale
from quantlag.grpo import SUFFIX, cpu_name


def sft_text(tok, q, a):
    a = re.sub(r"<<[^>]*>>", "", a)
    body, ans = a.split("####")
    msgs = [{"role": "user", "content": q + SUFFIX}, {"role": "assistant", "content": body.strip() + f"\nThe final answer is \\boxed{{{ans.strip()}}}."}]
    p = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    return p, full


def encode(tok, pairs, max_len):
    """Right-padded ids, attention mask and a mask over response tokens."""
    ids, resp = [], []
    for p, full in pairs:
        pi, fi = tok(p)["input_ids"], tok(full)["input_ids"][:max_len]
        ids.append(fi); resp.append([0] * len(pi) + [1] * (len(fi) - len(pi)))
    L = max(map(len, ids))
    pad = tok.pad_token_id
    x = torch.tensor([i + [pad] * (L - len(i)) for i in ids])
    am = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids])
    rm = torch.tensor([r + [0] * (L - len(r)) for r in resp])
    return x, am, rm


@torch.no_grad()
def logprobs(model, x, am, over=None):
    kw = dict(input_ids=x, attention_mask=am)
    out = functional_call(model, over, args=(), kwargs=kw, strict=False) if over else model(**kw)
    return torch.log_softmax(out.logits[:, :-1].float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--variant", default="plain", choices=["plain", "qurl"])
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--kl-at", default="0,10,25,50")
    ap.add_argument("--kl-n", type=int, default=200)
    ap.add_argument("--kl-batch", type=int, default=4)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("openai/gsm8k", "main")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    if a.variant == "qurl":
        invariant_scale(model, 1.5)
    names = quantized_names(model)
    P = dict(model.named_parameters())
    rules = ["rtn"] if a.variant == "qurl" else ["rtn", "sr", "ef"]
    copies = {(g, r): {n: Served(P[n].detach(), g, r, seed=a.seed * 7919 + i) for i, n in enumerate(names)}
              for g in a.grids.split(",") for r in rules}
    master0 = {n: P[n].detach().clone() for n in names}
    prev = {n: P[n].detach().clone() for n in names}
    order = random.Random(a.seed).sample(range(len(ds["train"])), a.steps * a.batch)
    test = ds["test"]
    kl_rows = test.select(range(len(test) - a.kl_n, len(test)))
    kl_pairs = sorted((sft_text(tok, q, ans) for q, ans in zip(kl_rows["question"], kl_rows["answer"])), key=lambda p: len(p[1]))
    kl_data = [encode(tok, kl_pairs[i:i + a.kl_batch], a.max_len) for i in range(0, a.kl_n, a.kl_batch)]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    kl_at = {int(s) for s in a.kl_at.split(",") if s}
    key = lambda k: f"{k[0]}/{'qurl' if a.variant == 'qurl' else k[1]}"
    log = {"config": vars(a), "cpu": cpu_name(), "torch": torch.__version__, "threads": torch.get_num_threads(),
           "n_quantized": sum(P[n].numel() for n in names), "steps": [], "kl": []}
    out = a.out or f"results/lag_{a.model.split('/')[-1]}_{a.variant}_lr{a.lr:g}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def save():
        json.dump(log, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)

    def kl_eval(step):
        t = time.time()
        sums = {key(k): 0.0 for k in copies}
        sums["stuck"] = 0.0  # reference: block linears frozen at step 0, i.e. a served copy that never moves
        ntok = 0
        for x, am, rm in kl_data:
            lp_m = logprobs(model, x, am)
            m = rm[:, 1:].bool()
            ntok += int(m.sum())
            pm = lp_m[m]
            for k, c in copies.items():
                over = {n: s.weight() for n, s in c.items()}
                lp_s = logprobs(model, x, am, over)[m]
                sums[key(k)] += float((pm.exp() * (pm - lp_s)).sum(-1).double().sum())
                del over, lp_s
            lp_s = logprobs(model, x, am, master0)[m]
            sums["stuck"] += float((pm.exp() * (pm - lp_s)).sum(-1).double().sum())
        rec = {"step": step, "tokens": ntok, "t": time.time() - t, **{k: v / ntok for k, v in sums.items()}}
        log["kl"].append(rec)
        print("kl", json.dumps(rec), flush=True)

    t0 = time.time()
    for step in range(a.steps + 1):
        if step in kl_at:
            kl_eval(step); save()
        if step == a.steps:
            break
        rows = ds["train"].select(order[step * a.batch:(step + 1) * a.batch])
        x, am, rm = encode(tok, [sft_text(tok, q, ans) for q, ans in zip(rows["question"], rows["answer"])], a.max_len)
        t = time.time()
        ntok, loss = int(rm[:, 1:].sum()), 0.0
        for i in range(0, len(x), a.micro):  # micro-batches: fp32 activations of 8 x 320 tokens do not fit next to the copies
            xi, mi = x[i:i + a.micro], rm[i:i + a.micro, 1:].bool()
            logits = model(input_ids=xi, attention_mask=am[i:i + a.micro]).logits[:, :-1]
            li = torch.nn.functional.cross_entropy(logits[mi], xi[:, 1:][mi], reduction="sum") / ntok
            li.backward(); loss += li.item()
            del logits, li
        opt.step(); opt.zero_grad(set_to_none=True)
        t_train = time.time() - t
        t = time.time()
        rec = {"step": step + 1, "loss": loss, "t_train": t_train}
        with torch.no_grad():
            P = dict(model.named_parameters())
            upd = {}
            for k, c in copies.items():
                changed = num = den = err = 0.0
                for n, s in c.items():
                    w = P[n].detach()
                    changed += s.sync(w, w - prev[n])
                    dm = w - master0[n]
                    ds_ = s.weight() - dequant(quantize(master0[n], s.grid, s.scale), s.grid, s.scale)
                    num += float((dm * ds_).sum()); den += float((dm * dm).sum()); err += float(((ds_ - dm) ** 2).sum())
                upd[key(k)] = {"changed_frac": changed / log["n_quantized"], "realized": num / max(den, 1e-30),
                               "track_err": math.sqrt(err / max(den, 1e-30))}
            rec["exports"] = upd
            for n in names:
                prev[n].copy_(P[n])
        rec["t_sync"] = time.time() - t
        log["steps"].append(rec)
        print(json.dumps({"step": rec["step"], "loss": round(rec["loss"], 4), "t_train": round(t_train, 1), "t_sync": round(rec["t_sync"], 1),
                          **{k: round(v["changed_frac"], 6) for k, v in upd.items()}}), flush=True)
        save()
    log["wall_s"] = time.time() - t0
    save()


if __name__ == "__main__":
    main()
