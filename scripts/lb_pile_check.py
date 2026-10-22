"""Compare two public-leaderboard snapshots: size of the 0.958 pile, our rank, medal lines,
and whether new pile entrants land below old ones (exact ties ordered by submission time)."""
import csv
import io
import itertools
import os
import sys
import zipfile
from collections import Counter


def load(path):
    z = zipfile.ZipFile(path)
    name = z.namelist()[0]
    return name, list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding='utf-8-sig')))


_, old = load(sys.argv[1])
name, new = load(sys.argv[2])
print(name, len(new))
print(sorted(Counter(r['Score'] for r in new).items(), reverse=True)[:14])
pile = [r for r in new if r['Score'] == '0.958']
print('pile ranks', pile[0]['Rank'], '-', pile[-1]['Rank'], 'size', len(pile))
TEAM = os.environ.get('KNEE_TEAM_NAME', '')
if TEAM:
    print([r for r in new if r['TeamName'] == TEAM][:1])
n = len(new)
for label, k in (('gold', 10 + int(0.002 * n)), ('silver', int(0.05 * n)), ('bronze', int(0.10 * n))):
    print(label, 'rank', k, 'score', new[k - 1]['Score'])

by_id = {r['TeamId']: r for r in old}
stayed = [(int(by_id[r['TeamId']]['Rank']), int(r['Rank'])) for r in pile
          if r['TeamId'] in by_id and by_id[r['TeamId']]['Score'] == '0.958']
inversions = sum(1 for (a1, b1), (a2, b2) in itertools.combinations(stayed, 2) if (a1 - a2) * (b1 - b2) < 0)
print('pile teams on both days', len(stayed), 'order inversions', inversions)
entrants = [int(r['Rank']) for r in pile if r['TeamId'] not in by_id or by_id[r['TeamId']]['Score'] != '0.958']
if entrants:
    entrants.sort()
    print('new pile entrants', len(entrants), 'rank min/median/max', entrants[0], entrants[len(entrants) // 2], entrants[-1])
