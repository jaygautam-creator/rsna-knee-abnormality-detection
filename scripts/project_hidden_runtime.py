"""Project the hidden-run timeline (N_HIDDEN studies) from a timing-probe kernel log (v90 / v91 notebook run on N train studies).

Usage: python scripts/project_hidden_runtime.py <kernel-log.json> [N_HIDDEN=1284]
Per stage: fixed overhead is estimated from the 3-study commit run of v89 (model load / compile), the rest scales with N.
"""
import json
import re
import sys

log = json.loads(open(sys.argv[1]).read())
N_HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 1284
text = [(e['time'], e['data']) for e in log]


def find(pat):
    out = []
    for t, d in text:
        m = re.search(pat, d)
        if m:
            out.append((t, m))
    return out


summ = [json.loads(m.group(1)) for _, m in find(r'PROBE_SUMMARY (\{.*\})')]
fake = [json.loads(m.group(1)) for _, m in find(r'PROBE_FAKE_READY (\{.*\})')]
print('fake root', fake[:1])
if not summ:
    for t, d in text:
        if any(k in d for k in ('STAGE_DONE', 'CONTROLLER_DONE', 'PARTB_SECONDS', 'S6_', 'Traceback', 'Error', 'PROGRESS')):
            print('%8.1f %s' % (t, d.rstrip()[:300]))
    sys.exit('no PROBE_SUMMARY (run failed or still running)')
s = summ[0]
n = s['n']
# commit-run (3 studies) stage seconds from v89: fixed costs dominate there
COMMIT = {'raptor384': 57.0, 'raptor224': 29.0, 'r6': 355.1, 'partb': 36.0, 's6': 65.1}
stages = {'raptor384': s['raptor384_s'], 'raptor224': s['raptor224_s'], 'r6': s['r6_controller_s'], 'partb': s['partb_s'],
          's6': (s['s6'].get('seconds') or 0.0)}
print(json.dumps(s, indent=1, default=str)[:4000])
tot = 0.0
proj = {}
for k, v in stages.items():
    c = COMMIT[k]
    per = max(0.0, (v - c) / max(1, n - 3))     # marginal seconds per study (whole machine)
    fixed = max(0.0, c - 3 * per)
    p = fixed + per * N_HIDDEN
    proj[k] = p
    tot += p
    print('%-10s probe %7.0f s for %d studies | marginal %6.2f s/study | fixed ~%5.0f s | hidden %d -> %5.2f h' % (k, v, n, per, fixed, N_HIDDEN, p / 3600))
chain = (proj['raptor384'] + proj['raptor224'] + proj['r6'] + proj['partb']) / 3600 + 0.08   # + setup / blend cells
print('v74 chain (incl. Part B) ends at ~%.2f h; S6 standalone needs %.2f h for all %d studies' % (chain, proj['s6'] / 3600, N_HIDDEN))
s6_per = max(1e-6, (stages['s6'] - COMMIT['s6']) / max(1, n - 3))
window = max(0.0, 8.40 - chain) * 3600
print('v90 S6 window %.2f h -> coverage ~%.0f %%' % (window / 3600, 100 * min(1.0, window / (s6_per * N_HIDDEN))))
