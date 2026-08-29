# -*- coding: utf-8 -*-
"""Cross-field structure of the 13-audit reporting matrix.

The paper reports per-field counts; this reports what the matrix says
ACROSS fields: reporting quality by year, whether careful audits cluster
(they do not -- instrument-careful and stimulus-careful audits are
disjoint), and which single field flips the most audits to re-runnable.
Under a minimal re-run criterion (exact prompt + pinned checkpoint +
decoding parameters), pinning the checkpoint alone flips three audits,
and it is the panel's biggest near-miss (2 report it, 10 partially).

PROVENANCE. Drafted inside an adversarial audit workflow (25 Aug 2026);
independently recomputed before adoption; all counts exact, no
statistics. Counting rule = the matrix's own _counting_rules.

    sh paper-a/src/_py.sh paper-a/src/analyze_matrix_structure.py

Writes paper-a/data/reference/matrix_structure.json.
"""
import json, re, sys, itertools
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pathlib
_R = pathlib.Path(__file__).resolve().parents[2]
P = str(_R / 'paper-a/data/reference/reporting_practice_matrix.json')
OUT = _R / 'paper-a/data/reference/matrix_structure.json'
ART = {}
d = json.load(open(P, encoding='utf-8'))
fields = [f[0] for f in d['field_order']]
pretty = dict(d['field_order'])
audits = [s for s in d['studies'] if s['kind'] == 'llm_hiring_audit']
assert len(audits) == 13

def year(label):
    return int(re.search(r'(\d{4})$', label).group(1))

def verdict(s, f):
    return s['cells'][f]['verdict']

# ---------- A. per-audit strict counts ----------
print('== A. Per-audit reporting quality (strict: reported/applicable; lenient adds partial) ==')
rows = []
for s in audits:
    rep  = sum(1 for f in fields if verdict(s, f) == 'reported')
    part = sum(1 for f in fields if verdict(s, f) == 'partial')
    na   = sum(1 for f in fields if verdict(s, f) == 'not-applicable')
    app  = 22 - na
    rows.append((s['label'], year(s['label']), rep, part, app))
rows.sort(key=lambda r: (-r[2] / r[4], r[0]))
for lab, yr, rep, part, app in rows:
    print(f'  {lab:26s} {yr}  strict {rep:2d}/{app:2d} ({rep/app:.0%})   lenient {rep+part:2d}/{app:2d} ({(rep+part)/app:.0%})')

# ---------- B. by publication year ----------
print()
print('== B. Reporting quality by publication year (strict) ==')
byyr = {}
for lab, yr, rep, part, app in rows:
    byyr.setdefault(yr, []).append((lab, rep, part, app))
for yr in sorted(byyr):
    g = byyr[yr]
    R = sum(r for _, r, _, a in g); A = sum(a for _, _, _, a in g)
    Pp = sum(p for _, _, p, _ in g)
    fr = [f'{lab.split()[0]} {r}/{a}' for lab, r, p, a in g]
    print(f'  {yr}: {len(g)} audit(s); pooled strict {R}/{A} ({R/A:.0%}); lenient {R+Pp}/{A} ({(R+Pp)/A:.0%});  per-audit: ' + ', '.join(fr))

print()
print('  Partial share by year (partials / applicable cells):')
for yr in sorted(byyr):
    g = byyr[yr]
    Pp = sum(p for _, _, p, _ in g); A = sum(a for _, _, _, a in g)
    print(f'    {yr}: {Pp}/{A} ({Pp/A:.0%})')

instr = ['checkpoint_pinned', 'quantization_reported', 'serving_stack',
         'concurrency_or_batching', 'cache_policy', 'decoding_params']
print()
print('  Instrument fields only (checkpoint, quantization, serving stack, batching, cache, decoding):')
for yr in sorted(byyr):
    labs = [lab for lab, *_ in byyr[yr]]
    ss = [s for s in audits if s['label'] in labs]
    R = sum(1 for s in ss for f in instr if verdict(s, f) == 'reported')
    A = sum(1 for s in ss for f in instr if verdict(s, f) != 'not-applicable')
    print(f'    {yr}: strict {R}/{A}')

# ---------- C. co-occurrence structure ----------
print()
print('== C. Co-occurrence: is there a careful cluster? ==')
rare = ['null_edit_control', 'name_sensitivity_reported', 'checkpoint_pinned',
        'quantization_reported', 'serving_stack', 'decoding_params', 'n_repeats',
        'resampling_unit', 'multiplicity_correction', 'code_or_data_released']
who = {}
for f in rare:
    who[f] = [s['label'] for s in audits if verdict(s, f) == 'reported']
    print(f'    {pretty[f]:34s} -> ' + (', '.join(who[f]) if who[f] else '(none)'))
print()
print('  Pairwise intersections among scarce fields (strict), pairs with any overlap:')
for f1, f2 in itertools.combinations(rare, 2):
    inter = sorted(set(who[f1]) & set(who[f2]))
    if inter:
        print(f'    {pretty[f1]} ({len(who[f1])}) & {pretty[f2]} ({len(who[f2])}): {len(inter)} -> {", ".join(inter)}')

cp = set(who['checkpoint_pinned']); ru = set(who['resampling_unit'])
print()
print(f'  Checkpoint pinned AND resampling unit stated (strict): {sorted(cp & ru)}')
print(f'  Checkpoint only: {sorted(cp - ru)};  resampling unit only: {sorted(ru - cp)}')

print()
print('  Scarce-field slots held per audit (strict, over the 10 scarce fields above):')
tot = 0
per = []
for s in audits:
    n = sum(1 for f in rare if verdict(s, f) == 'reported')
    per.append((n, s['label']))
    tot += n
per.sort(reverse=True)
for n, lab in per:
    if n:
        print(f'    {lab:26s} {n}')
print(f'    (audits reporting zero scarce fields: {sum(1 for n, _ in per if n == 0)} of 13; total scarce-field reports = {tot})')
print(f'    Top 4 audits hold {sum(n for n, _ in per[:4])} of {tot} scarce-field reports')

# ---------- D. distance to full reporting ----------
print()
print('== D. Distance to full reporting (strict missing fields per audit) ==')
missing = {}
for s in audits:
    missing[s['label']] = [f for f in fields if verdict(s, f) in ('partial', 'not-reported')]
for lab in sorted(missing, key=lambda L: len(missing[L])):
    print(f'  {lab:26s} missing {len(missing[lab]):2d} applicable fields')
print(f'  Closest audit is still {len(min(missing.values(), key=len))} fields short -> no single field flips anyone under the full-matrix criterion.')

# ---------- E. minimum-effort field ----------
print()
print("== E. Which single field, if universally adopted, closes the most audits' gap? ==")
print('  (i) Marginal gain per field = audits not strictly reporting it (applicable only):')
gain = sorted(((sum(1 for s in audits if verdict(s, f) in ('partial', 'not-reported')), f) for f in fields), reverse=True)
for n, f in gain:
    print(f'    {pretty[f]:34s} would newly cover {n:2d} audits')

print()
print('  (ii) Instrument-reproducibility criterion (paper Sec 5.2 / Sec 9 items 6-7 + decoding):')
gaps = {}
for s in audits:
    gaps[s['label']] = [f for f in instr if verdict(s, f) in ('partial', 'not-reported')]
print(f'    Audits currently reporting the full instrument set: {sum(1 for g in gaps.values() if not g)} of 13')
for lab in sorted(gaps, key=lambda L: len(gaps[L])):
    print(f'    {lab:26s} instrument gap {len(gaps[lab])}: ' + (', '.join(pretty[f] for f in gaps[lab]) or '-'))
print()
print('    Single-field flips (audits whose instrument gap is exactly that one field):')
for f in instr:
    flip = [lab for lab, g in gaps.items() if g == [f]]
    print(f'      {pretty[f]:24s} -> flips {len(flip)}: ' + (', '.join(flip) if flip else '-'))

print()
print('  (iii) Minimal re-run criterion: exact prompt + checkpoint pinned + decoding params')
rerun = ['prompt_published', 'checkpoint_pinned', 'decoding_params']
rgaps = {}
for s in audits:
    rgaps[s['label']] = [f for f in rerun if verdict(s, f) in ('partial', 'not-reported')]
print(f'    Audits currently meeting it: {sum(1 for g in rgaps.values() if not g)} of 13')
for lab in sorted(rgaps, key=lambda L: (len(rgaps[L]), L)):
    print(f'    {lab:26s} gap {len(rgaps[lab])}: ' + (', '.join(pretty[f] for f in rgaps[lab]) or '-'))
print()
print('    Single-field flips under the minimal re-run criterion:')
ART['minimal_rerun'] = dict(
    criterion=rerun,
    n_meeting=sum(1 for g in rgaps.values() if not g),
    n_audits=len(audits), flips={})
for f in rerun:
    flip = sorted(lab for lab, g in rgaps.items() if g == [f])
    ART['minimal_rerun']['flips'][f] = flip
    print(f'      {pretty[f]:24s} -> flips {len(flip)}: ' + (', '.join(flip) if flip else '-'))
ART['checkpoint_near_miss'] = dict(
    n_reported=sum(1 for s in audits
                   if verdict(s, 'checkpoint_pinned') == 'reported'),
    n_partial=sum(1 for s in audits
                  if verdict(s, 'checkpoint_pinned') == 'partial'))

# ---------- F. near-misses ----------
print()
print('== F. Near-misses: fields where partial credit exceeds strict credit ==')
for f in fields:
    r = sum(1 for s in audits if verdict(s, f) == 'reported')
    p = sum(1 for s in audits if verdict(s, f) == 'partial')
    if p > r:
        print(f'  {pretty[f]:34s} reported {r}, partial {p}')

# ---------- cross-check ----------
print()
print('== Cross-check vs the JSON counts block ==')
ok = True
for f in fields:
    c = d['counts'][f]
    r = sum(1 for s in audits if verdict(s, f) == 'reported')
    p = sum(1 for s in audits if verdict(s, f) == 'partial')
    a = sum(1 for s in audits if verdict(s, f) != 'not-applicable')
    if (r, p, a) != (c['n_reported'], c['n_partial'], c['n_applicable']):
        ok = False
        print('  MISMATCH', f, (r, p, a), (c['n_reported'], c['n_partial'], c['n_applicable']))
print('  all per-field counts recomputed from cells match the counts block:', ok)

# ---------- verify what the paper itself reports (fitz) ----------
import fitz
doc = fitz.open('paper-a/figures/paper_instrument_validity_v3.pdf')
full = ''.join(pg.get_text() for pg in doc)
for pat in ['by year', 'trend', 'co-occur', 'careful cluster']:
    print(f'  paper mentions {pat!r}:', bool(re.search(pat, full, re.I)))
print('  Table 10 per-field rows present:', 'Reporting practice across 13 LLM hiring audits' in full)

# ---------- artifact ----------
ART['_what'] = ('Cross-field structure of the reporting matrix: the minimal '
                're-run criterion and which single field flips the most '
                'audits to meeting it.')
ART['_provenance'] = ('Computed inside an adversarial audit workflow, '
                      '25 Aug 2026; counts exact, independently reproduced '
                      'before adoption.')
OUT.write_text(json.dumps(ART, indent=1), encoding='utf-8')
print()
print(f'wrote {OUT}')
