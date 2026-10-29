"""Round 12 (advisor reach step 3): per-flow cost of the frozen MLP score, the AQT threshold and one fine-tuning step.
CPU, one thread, 77 features (32-16-1 MLP, as in the paper). Timings are medians of repeated runs, microseconds per flow.
usage: python3 round12_cost.py -> round12_cost.json"""
import json, time, platform, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.RandomState(0)
D = 77
m = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
nparams = sum(p.numel() for p in m.parameters())
X = torch.tensor(rng.randn(20000, D), dtype=torch.float32)


def med(f, reps=7):
    ts = []
    for _ in range(reps): t = time.perf_counter(); n = f(); ts.append((time.perf_counter() - t) / n * 1e6)
    return float(np.median(ts))


def score_single():
    m.eval()
    with torch.no_grad():
        for i in range(2000): m(X[i:i + 1])
    return 2000


def score_batch():
    m.eval()
    with torch.no_grad(): m(X)
    return len(X)


s = m(X).detach().numpy().flatten()


def aqt_quantile():
    buf = list(s[:500]); t = 0
    for v in s[500:3500]:
        buf.append(v); np.quantile(buf[-500:], 0.99); t += 1
    return t


def aqt_sorted():
    import bisect
    win = list(s[:500]); srt = sorted(win); k = int(np.ceil(0.99 * 500)) - 1; t = 0
    for v in s[500:20500 % len(s) if False else 20000]:
        old = win.pop(0); srt.pop(bisect.bisect_left(srt, old)); win.append(v); bisect.insort(srt, v); _ = srt[k]; t += 1
    return t


def ft_step():
    m.train(); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    for i in range(1000):
        opt.zero_grad(); l = lf(m(X[i:i + 1]).view(-1), torch.tensor([float(i % 2)])); l.backward(); opt.step()
    return 1000


def ft_step_reset():
    m.train(); lf = nn.BCEWithLogitsLoss()
    for i in range(1000):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); opt.zero_grad(); l = lf(m(X[i:i + 1]).view(-1), torch.tensor([float(i % 2)])); l.backward(); opt.step()
    return 1000


out = dict(params=nparams, model_kb=nparams * 4 / 1024, aqt_buffer_kb=500 * 4 / 1024, adam_state_kb=nparams * 8 / 1024,
           score_single_us=med(score_single), score_batch_us=med(score_batch), aqt_np_quantile_us=med(aqt_quantile),
           aqt_sorted_window_us=med(aqt_sorted), ft_persistent_us=med(ft_step), ft_reset_us=med(ft_step_reset),
           cpu=platform.processor() or platform.machine(), torch=torch.__version__)
for k, v in out.items(): print(k, round(v, 2) if isinstance(v, float) else v)
json.dump(out, open("round12_cost.json", "w"), indent=1)
