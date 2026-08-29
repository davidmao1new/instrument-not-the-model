// A round of adversarial critique by fresh model instances.
//
// WHAT THIS IS FOR. After every round of fixes, the paper should be read again
// by readers who have never seen it and did not make its mistakes. Each agent
// below starts with an empty context: it gets the built paper, the repository
// and a mandate to find defects. It does not get this conversation, the
// changelog's account of what was already fixed, or any of the reasoning that
// produced the current draft. That is the point -- a critic who knows why a
// choice was made will not notice that the paper never says why.
//
// HOW THE LOOP TERMINATES. A round returns CONFIRMED findings: raised by one
// agent, then independently reproduced by a second agent that was told to
// refute them. The loop stops when a round returns zero confirmed findings of
// severity major or critical. It does not stop on zero findings of any kind,
// because minor stylistic disagreements never run out.
//
// WHY REFUTATION IS NOT OPTIONAL. Unverified critique is worse than none: it
// costs a fix cycle and it can talk a paper out of a correct claim. Three
// times in this project a confident finding was wrong. Every finding here must
// be reproduced from the artifacts by someone who was trying to kill it.
//
//   Run:  Workflow({ scriptPath: ".../critique_round.js", args: {...} })
//   args: { version: "v4", paperText: "<abs path>", repo: "<abs path>",
//           python: "<abs path>", nCritics: 8 }

export const meta = {
  name: 'critique-round',
  description: 'Fresh-context adversarial critique of the built paper, with per-finding refutation',
  phases: [
    { title: 'Critique', detail: 'independent readers, each with an empty context' },
    { title: 'Refute', detail: 'every finding attacked by a second reader' },
  ],
}

const A = args || {}
const REPO = A.repo || 'research'
const PAPER = A.paperText
const PY = A.python || 'C:/research-toolchain/venv/Scripts/python.exe'
const VERSION = A.version || 'current'
const N = A.nCritics || 8

const PRE = `You are reviewing a quantitative research paper for a top venue. You have never seen it before. Your job is to find what is WRONG with it — not to summarise it, not to praise it, not to suggest extra experiments.

PAPER (text, page-delimited, extracted from the built PDF): ${PAPER}
REPOSITORY: ${REPO}
PYTHON: ${PY}    <-- the repo's .venv is dead; use this absolute path for everything.

The paper's governing rule is that every number in it is interpolated at build time from a JSON artifact under paper-a/data/, never typed. Raw per-call records are JSONL under paper-a/data/*/. The typesetting script is paper-a/src/build_paper_v3.py.

YOU MUST VERIFY, NOT SPECULATE. For any numerical claim you doubt, recompute it yourself from the rawest data you can reach. A finding you could not reproduce is not a finding; report it as a question instead, at low confidence.

WHAT COUNTS AS A DEFECT, in rough priority order:
 1. A statement that is false given the artifacts on disk.
 2. A claim that outruns its evidence — a mechanism asserted for a number that does not come from it, a null read as absence, a range quoted without the interval that would make it meaningful, a caption claiming more than the table contains.
 3. An internal contradiction: two places in the paper that disagree, or a figure that disagrees with its caption or with the text.
 4. A statistical error: wrong resampling unit, a ratio with a denominator not distinguishable from zero, selection on the outcome, multiplicity ignored, an estimator biased by construction.
 5. A claim about someone else's paper that their paper does not support.
 6. Something a hostile reviewer could use to reject, that the paper has not pre-empted.

DO NOT report: style preferences, requests for more experiments, "could be clearer", or anything you have not checked.

For each finding give the exact quoted sentence or table cell, what is wrong, the commands you ran and what they returned, and the minimal correct replacement.`

const FINDINGS = {
  type: 'object', additionalProperties: false, required: ['lens', 'findings'],
  properties: {
    lens: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'severity', 'quote', 'defect', 'evidence', 'fix', 'confidence'],
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          quote: { type: 'string' },
          defect: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['verified', 'likely', 'suspected'] },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['refuted', 'reasoning', 'independent_check'],
  properties: {
    refuted: { type: 'boolean' },
    reasoning: { type: 'string' },
    independent_check: { type: 'string' },
    revised_severity: { type: 'string', enum: ['critical', 'major', 'minor', 'not-a-defect'] },
  },
}

// Lenses, chosen so two critics rarely find the same thing. A pool of
// identical readers returns one finding eight times.
const LENSES = [
  { key: 'arithmetic', p: 'Recompute EVERY number in the abstract and in Tables 1 through 6 from the rawest data on disk, by your own route rather than by re-running the paper\'s scripts. Check that each number agrees with every other place the same quantity appears. Report every disagreement, however small, and say whether it is a different estimand or an error.' },
  { key: 'statistics', p: 'You are a hostile statistician. Audit the inference: resampling units, clustering, multiplicity across the whole paper and not only within a family, ratios whose denominators may not be distinguishable from zero, nulls reported without a minimum detectable effect, estimators that could be biased by their own construction, intervals on quantities that are themselves estimated (a standard deviation, a ratio). Check whether any headline rests on a model or cell selected on its own outcome.' },
  { key: 'claims-vs-evidence', p: 'Take every load-bearing sentence in the abstract, Section 1 and the Conclusion, and find the artifact that licenses it. Report every sentence whose evidence is weaker than its phrasing — universals ("none", "every", "always"), causal language where only a correlation was measured, and mechanisms asserted for numbers that do not come from that mechanism.' },
  { key: 'captions-figures', p: 'Check every table and figure caption against what the table or figure actually contains. Captions are rendered into the PNGs, so read the figure images themselves (render with PyMuPDF) and not only the text layer. Check figure numbering against in-text references, that every figure is referenced, and that no caption claims completeness ("every contrast", "all models") it does not have.' },
  { key: 'related-work', p: 'For every citation, check what the cited work actually says against the copy in lit/ and lit/text/. Report any claim about another paper that its text does not support. Then look for prior work the paper should be positioned against and is not — search the web for close antecedents of its central claims. Also check that every reference is cited in the body and every body citation is in the list.' },
  { key: 'reproducibility', p: 'Try to reproduce the paper\'s claims from the repository as a stranger would. Are the artifacts the paper cites actually present and do they contain what it says? Are any quarantined or superseded artifacts still being read by a live script? Is anything in the paper hand-typed that should be interpolated? Run the test suite and the consistency and integrity audits and report anything they flag.' },
  { key: 'self-consistency', p: 'Hunt for internal contradictions. Read the whole paper and list every pair of statements that cannot both be true, every section that describes the design differently from another section, and every place a number, count or scope is stated two ways. Pay attention to counts of studies, models, conditions, cells and name pairs.' },
  { key: 'reviewer-attack', p: 'You are Reviewer 2 and you want to reject this paper. Write the strongest honest case against it: what is the weakest link in the central argument, what would a reader with a different prior conclude, and what does the paper claim that its design cannot support? Then check each of your attacks against the data, and report only the ones that survive your own checking.' },
]

phase('Critique')

// SPLIT THE ROUND, BECAUSE A WHOLE ROUND IS TOO BIG TO LAND IN ONE PIECE.
// Round 1 spawned eight critics and up to ten refuters each, and roughly half
// the agents died before returning -- which silently narrows the round rather
// than failing it, so the paper looks cleaner than it was checked. Running two
// invocations of four lenses each keeps every agent's result, and the two
// halves are merged afterwards by merge_critique_rounds.py.
const OFFSET = A.lensOffset || 0

const results = await pipeline(
  LENSES.slice(OFFSET, OFFSET + N),
  l => agent(`${PRE}\n\nYOUR LENS — ${l.key}:\n${l.p}`,
    { label: `critique:${l.key}`, phase: 'Critique', schema: FINDINGS }),
  (r, l) => {
    // A DEAD CRITIC IS NOT A CLEAN CRITIC. `r` is null when the lens agent
    // failed -- a session limit, a dropped connection -- and returning an
    // empty `verified` for that case made it identical to a critic that ran
    // and found nothing. Round 6 hit the session limit on all eight lenses and
    // the round reported `converged: true` on zero findings examined, which is
    // one merge away from tripping the loop's stop condition on a round where
    // nothing looked at the paper. The distinction is recorded here and
    // enforced in the return value.
    if (!r) return { lens: l.key, verified: [], critic_failed: true }
    if (!r.findings || !r.findings.length) return { lens: l.key, verified: [] }
    return parallel(r.findings.slice(0, 10).map(f => () =>
      agent(`You are refuting a claimed defect in a research paper. Default to refuted=true. Return refuted=false ONLY if you personally reproduced the defect from the artifacts.

PAPER: ${PAPER}
REPOSITORY: ${REPO}
PYTHON: ${PY}

CLAIMED DEFECT (${f.severity}, reporter confidence ${f.confidence}), lens ${l.key}:
  quote:    ${f.quote}
  defect:   ${f.defect}
  evidence: ${f.evidence}
  fix:      ${f.fix}

Do not trust the reporter's evidence. Re-derive it yourself. Consider: is there a definition elsewhere in the paper that makes the sentence correct? Is the reporter using the wrong subset, dedup rule or denominator? Is this a Monte Carlo difference rather than a defect? Does the caption's actual scope already cover it? Is the quantity a deliberately different estimand?`,
      { label: `refute:${f.id}`, phase: 'Refute', schema: VERDICT })
      .then(v => ({ ...f, lens: l.key, verdict: v }))
    )).then(vs => ({ lens: l.key, verified: vs.filter(Boolean) }))
  }
)

const all = results.filter(Boolean).flatMap(r => r.verified || [])
const confirmed = all.filter(f => f.verdict && f.verdict.refuted === false)
const serious = confirmed.filter(f =>
  ['critical', 'major'].includes(f.verdict.revised_severity || f.severity))

// DID THE ROUND ACTUALLY RUN? A lens that died returns nothing, and nothing is
// what a lens that found nothing returns too. Convergence may only be claimed
// when every lens reported -- otherwise "no serious findings" means "no serious
// findings were looked for".
const nLenses = LENSES.slice(OFFSET, OFFSET + N).length
const failed = results.filter(r => !r || r.critic_failed).length
const ran = nLenses - failed

log(`${VERSION}: ${ran}/${nLenses} lenses reported, ${all.length} examined, ` +
    `${confirmed.length} confirmed, ${serious.length} serious`)
if (failed) log(`  WARNING: ${failed} lens(es) failed; this round is INCOMPLETE and cannot show convergence`)

return {
  version: VERSION,
  n_examined: all.length,
  n_confirmed: confirmed.length,
  n_serious: serious.length,
  n_lenses: nLenses,
  n_lenses_reported: ran,
  n_lenses_failed: failed,
  complete: failed === 0,
  // Convergence requires that the critics ran AND found nothing serious.
  converged: failed === 0 && ran > 0 && serious.length === 0,
  _incomplete_warning: failed
    ? `${failed} of ${nLenses} lenses failed to report; zero findings here `
      + 'means zero were examined, not zero exist. Re-run before drawing any '
      + 'conclusion about convergence.'
    : undefined,
  confirmed: confirmed.map(f => ({
    lens: f.lens, id: f.id,
    severity: f.verdict.revised_severity || f.severity,
    quote: f.quote, defect: f.defect, evidence: f.evidence, fix: f.fix,
    refuter_check: f.verdict.independent_check,
  })),
  refuted_count: all.length - confirmed.length,
}
