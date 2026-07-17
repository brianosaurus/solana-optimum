# The Optimum Study — Comparison Table & Study Guide

*For: MIDS graduate student prep. Everything here maps to concepts from W203
(statistics) and W241 (experiments & causal inference): potential outcomes,
randomization, DiD, regression discontinuity, clustered errors, bootstrap.
Learn the numbers cold, the mechanisms in your own words, and the scripts
verbatim.*

---

# Part 1 · The Comparison Table

## Where THEIR approach (the APR Calculator) is better

| dimension | their approach | why it beats ours |
|---|---|---|
| Direct observation of the treatment | They measure `Q_mump2p(80%)` from their own gateways — actual mump2p delivery times | We cannot observe mump2p at all (shadow mode is invisible in public data). Our Δ is an assumed floor/multiplier; theirs is measured on the real product |
| Outcome completeness | Reduced-form: realized operator APR captures every revenue channel at once, including ones we didn't model | A structural model is only as complete as its author's imagination |
| Baseline anchor | 2.91% network APR from Rated — realized yield | Realized APR is the better denominator for a "% uplift" claim than our spec-derived issuance |
| Reach threshold | One spec-motivated criterion: p80 propagation ≈ 66% attestation supermajority + margin | Cleaner than our patchwork (p90 spread, 40% proposer-boost, headroom caps) |
| Auditability | Δ × slope × stake — auditable in five minutes | Our pipeline takes days and a statistician. For a *calculator*, theirs is right |
| Updateability | Δ re-measures continuously from live telemetry | Our calibration is frozen to study windows |

## Where OUR study is better

| dimension | their approach | why ours beats it |
|---|---|---|
| Causal identification | Cross-sectional APR-vs-latency across operators — confounded | Deadline dose-response + RANDAO random assignment: same validators, random exposure |
| Venue transfer | Hoodi Δ transplanted raw onto a mainnet slope | Floor model `min(observed, floor)`: 6.7× on Hoodi → 2.2× at mainnet's median |
| Equilibrium / adoption | "Mean-field assumptions" | Measured adoption curve: ~40% receiver ramp, (1−α) arms-race decay, sweet spot 40–55%, public-goods floor at saturation |
| Behavioral honesty | "Immediate revenue improvements" | 92% of value requires re-tuned publication timing; passive install ≈ $2–3/yr |
| Physical constraints | Unbounded multiplier extrapolation | Light-speed floor; deadline headroom; 6×→20× moves the total only 11% |
| Falsification | None — no placebo, control, or null ever published | Five independent designs on the deployed network; placebos; 1× calibration checks; ~60 unit tests |
| Channel structure | One black-box slope | A/B/C/D separable with ownership; B/C exclusivity; orphan values validated vs relay payloads (1.02×) |
| Uncertainty | Point estimates | Scenarios everywhere: 6×/3×/2×, floors 150/250/400ms, cloud/metal |
| Reproducibility | Private operator panel, unseen figure, gateway telemetry | End-to-end public data, open pipeline |
| Framing integrity | Present-tense revenue for an undeployed product | Counterfactual labeling enforced in code, dashboard, report |
| Bandwidth channel | Absent | Modeled: ~3.8 TB/node/yr, priced per hosting regime |

**Synthesis line (say this):** *"The approaches are complements. They hold the
one thing I can't get — direct measurement of the product. I hold everything
that makes a number believable — identification, transfer, equilibrium,
falsification. And both methods converge on about thirty dollars per validator
per year, so the fight isn't over the answer; it's over which method survives
scrutiny getting there."*

---

# Part 2 · Numbers You Must Know Cold

| number | what it is | one-line derivation |
|---|---|---|
| **4,000 ms** | attestation deadline (1/3 of a 12s slot) | consensus spec; attesters vote for whatever head they see at that instant |
| **99.9% → 67.4% → 0.1%** | correct-head rate: on-time / 4–5s / >6s | dose-response over 216,000 mainnet slots, attester-weighted |
| **0.94%** | measured head-miss rate | 1 − weighted mean correct_head_rate |
| **61%** | share of head misses on ON-TIME blocks | tail nodes; the reason "rescue late blocks" is the wrong model |
| **3,656 vs 1,811 ms** | median *publication* time, late vs normal blocks | late blocks are published late, not transported slowly |
| **331 ms / 976 ms** | mainnet median-node transit / p90 spread | median arrival − first sighting; p90 − min across sentries |
| **0.027 → 0.051 ETH** | block value at t=0 vs t≈3.5s (then flat) | MEV relay bid traces; V(t) plateaus when a block can't beat the deadline |
| **7.04e-6 ETH/ms** | dV/dt — value of one ms of delay | slope of V(t) over 1–4s |
| **188 / 214,712 = 0.088%** | orphan rate; hazard 1.6%→9.9%→28.5% across 4.0–6.0s | seen-blocks anti-join vs canonical chain |
| **$35.10** | modeled uplift per validator/yr at 6× (A 2.48 + C 32.76 + D 0.68; B 0.14 loses to C) | ETH @ $1,805.50, post-Pectra 46.2 ETH validators |
| **~$4.8M / yr** | all seven named partners at 6× | Kiln $1.55M, P2P $1.04M, Everstake $0.96M… |
| **40% / 55%** | adoption sweet spots (per-adopter / total value) | ramp `min(1, α/40%)` × decay `(1−α)`; θ=40% from proposer boost |
| **16.8%** | the seven partners' share of the network | 147,737 / 880,550 validators — below the sweet spot |
| **6.7× vs 2.2×** | what a 150ms floor buys on Hoodi vs mainnet median | floor model: same product, different baselines |
| **+11%** | value gain going 6× → 20× | deadline headroom caps C; publication floor caps A |
| **5.8%** | proposals that capture max block value safely today | publish ≥95% of plateau AND arrive <4s |
| **2.91% / +1.7%** | their baseline APR / their own calculator's uplift | 0.0159 ETH per 32 ETH — converges with our number |
| **5 designs, 0 detections** | the Hoodi identification record | ITS-vs-control, operator stepping, peer races, size-penalty DiD, mixture |

---

# Part 3 · The Statistics, MIDS-Style

## 3.1 Why the mainnet study is causal (the W241 story)

Two ingredients make this a natural experiment, not a correlation:

1. **A sharp exogenous threshold.** The spec — not any actor — sets the
   attestation deadline at 4,000ms. Blocks arriving at 3,950 vs 4,050ms are
   alike in every covariate; only votability changes. That's the regression
   discontinuity logic: identification comes from *local* comparison at a
   cutoff nobody can precisely manipulate (proposers can't steer gossip to the
   millisecond — and we ran a McCrary-style density check).
2. **Random assignment of the exposed.** RANDAO assigns attesters to slots by
   lottery. In potential-outcomes terms: treatment (being on duty when a late
   block appears) is independent of potential outcomes. So we compare *the same
   validators* across slots that randomly had early vs late blocks — not fast
   operators vs slow operators, which is selection on everything.

**Say it like this:** *"I never compare operators to each other — that's
confounded to death. I compare the same validators across slots that, by
lottery, got early or late blocks. RANDAO is my randomizer; the spec deadline
is my discontinuity."*

**The honest limitation (volunteer it before they find it):** our running
variable is a *sentry-median* arrival, not each attester's own arrival.
Classical measurement error in an RD's running variable smooths the jump —
so sharp-RD point estimates are bandwidth-sensitive (τ ranged −0.005 to −0.124
across bandwidths). That's why we **lead with the dose-response** (the full
measured curve), which needs no sharpness assumption, and treat the RD as
supporting. Measurement error attenuates → our effects are lower bounds.

## 3.2 Why their regression is not causal

Their APR-uplift comes from a cross-section: operators with lower latency have
higher APR. That's `selection on unobservables`: latency correlates with
operator quality (relay connections, MEV tuning, timing strategy, geography).
The latency coefficient absorbs all of it — classic omitted-variable bias.
Then they multiply a Hoodi-measured Δ into that biased slope. Two errors,
multiplied.

## 3.3 The DiD story (tell it as a journey)

1. **The dream:** staggered DiD of mump2p adoption on mainnet, Callaway–
   Sant'Anna estimator. C-S generalizes 2×2 DiD to staggered adoption: it
   estimates ATT(g,t) per adoption-cohort g and time t, using not-yet-treated
   units as controls, then aggregates — avoiding the negative-weighting bug in
   two-way fixed effects when effects vary over time.
2. **Why it died:** no staggering (all seven partners named in ONE press
   release, 2025-06-24) and no treatment (mump2p never reached mainnet — the
   treatment indicator is identically zero in the data). You cannot estimate
   the effect of a treatment that never occurred in your sample.
3. **The honest substitutes on Hoodi** (the only treated network), five designs:
   - **ITS + comparison series** (Hoodi vs never-treated Sepolia). One unit per
     arm — so *not* a real DiD; no cross-sectional variance, parallel trends
     untestable. The control still earns its keep: the biggest "effect"
     (−56% transit, 2025-09-29) appeared on the *control* 3 days later, bigger
     and cleaner (−72%, r² 0.84 vs 0.34) → common shock, not treatment.
   - **Per-operator stepping**: segment proposers by fee-recipient (operators
     deposited in contiguous index batches), estimate a least-squares
     changepoint per cluster. Adoption = staggered breaks. Result: all 14
     clusters break the *same day* — span 0 days → common shock.
   - **Peer-level races**: libp2p first-delivery wins per sending peer.
     Gateways would be new, fast, persistent winners. Result: young peers are
     no faster (−84 to +14ms edge); dominant winners born *after* the window.
   - **Size-penalty DiD (label-free)**: second difference = block difficulty,
     not identity. A constant-time overlay must compress the big-block
     penalty. Result: **inverted** — penalty tripled (225→1,346ms) in the
     deployment window. That's a congestion signature (consistent with
     shadow-mode traffic duplication), the opposite of offloading.
   - **Mixture emergence (label-free)**: if an unknown fraction adopted,
     log-transit becomes a 2-component mixture; the mixing weight *is* the
     adoption share. Fit by EM, gated on ΔBIC>10. Result: bimodality exists
     *before* deployment (the placebo fires) and no overlay-speed component
     ever appears.

**Say it like this:** *"When treatment assignment is unobservable, you move
the second difference onto something you can observe — block difficulty, or
the shape of the distribution. Five designs, including two that need no labels
at all, and none finds the treatment. The absence isn't a labeling problem."*

## 3.4 Inference: why not just OLS standard errors?

Slots within a day share network conditions (blob demand, client releases), so
errors are correlated within day — 216k slots are far fewer *effective*
observations. Fixes, in order:

- **Cluster-robust (CR1) sandwich**: allow arbitrary correlation within
  cluster, independence across; G=30 day-clusters.
- **Wild cluster bootstrap (WCR, Rademacher)**: with few clusters, CRVE
  over-rejects. Impose the null, flip cluster residual signs at random,
  rebuild the t-distribution empirically (Cameron-Gelbach-Miller). Caveat we
  cite: MacKinnon-Webb — even WCB misbehaves with few *treated* clusters.
- **Test discipline**: our suite includes a *size* test (under a true null,
  p-values ≈ uniform; nominal 10% test rejects ≈10%) and planted-effect
  recovery. An estimator you haven't watched fail on nulls isn't tested.

## 3.5 The counterfactual model — and the three bugs (tell these!)

Channels: **A** attester head votes (receive-side), **B** reorgs avoided,
**C** MEV delay budget, **D** bandwidth. Total = A + max(B,C) + D.

The three bugs make you credible because you found them yourself:

1. **The tail-attester bug.** First model: a block only costs attesters if its
   *median* arrival is past 4s. Wrong — arrival is a *distribution* across
   nodes; 61% of real misses occur on on-time blocks (tail nodes). Fix: the
   measured dose-response f(arrival) *is* the tail-mass function; acceleration
   moves each slot along the measured curve. Lesson: never collapse a
   distribution to a point when the outcome lives in the tail.
2. **Baseline ≠ counterfactual estimator mismatch.** We briefly compared a
   *measured* network-average baseline to a *modeled* single-node counterfactual
   — different populations, invalid delta. Fix: both sides through the same
   estimator; speedup=1 must reproduce the measured rate (it does: 1.05% vs
   0.94%). Lesson: a counterfactual that can't recover reality at the no-op is
   untrustworthy everywhere.
3. **The silent-zero Channel B.** Reorg pricing keyed on a column that didn't
   exist in the cached panel; a fallback returned $0.00 into a results table.
   Physically impossible (188 orphans measured!). Fix: measure the orphan
   hazard from *seen* blocks (winners AND losers — canonical-only data has
   survivorship bias), and add tests forbidding exact zeros. Lesson: silent
   fallbacks are how impossible numbers reach tables.

Also: **B and C are mutually exclusive** uses of the same saved milliseconds —
bank them as earlier arrival (safety) or spend them as later publication (MEV),
never both. Summing would double-count.

## 3.6 The transfer model (floor, not multiplier)

An RLNC overlay delivers in near-constant time; gossipsub scales with mesh
depth. So the product is an **absolute floor**, and it runs in parallel:
`transit_cf = min(observed, floor)`. The same 150ms floor = 6.7× on Hoodi
(baseline ~1,000ms) but 2.2× at mainnet's median node (331ms) — and 0.8×,
a *downgrade*, if the realized floor is 400ms. Multipliers don't transfer
across baselines; floors do. Bonus: 20× multiplicative on mainnet implies
17ms global propagation — faster than light in fiber.

## 3.7 The adoption sweep (the game theory)

Channel C per adopter = C_max × min(1, α/θ) × (1−α):

- **Ramp**: a proposer can only *spend* delay if ~θ=40% of attesters receive
  fast (proposer-boost reorg threshold). Below that, delaying is suicide.
- **Decay**: the captured MEV is flow stolen from the *next* slot; with
  probability α your predecessor steals from you the same way. Red-queen race.
- Peaks: per-adopter at α=θ (40%), total value at ~55%, → public-goods floor
  ($3.30) at 100%. The adopt-vs-not *spread* saturates at ~$32 and never
  decays — late adoption is bought to stop bleeding, not to gain.

---

# Part 4 · Spoken Scripts (memorize verbatim)

## The 30-second version

*"Optimum markets 6-to-20× faster block propagation. I measured what that's
actually worth on Ethereum mainnet using public data: about thirty-five
dollars per validator per year — and their own APR calculator, buried in a
Notion doc, agrees with me within twenty percent. Ninety-two percent of that
value is MEV timing, not reliability, and it only pays if you re-tune your
block publication. On Hoodi — the only network where the product runs — I
tried five independent ways to detect it, including two that need no adopter
labels, and found nothing. The value is real but modest, front-loaded to
partial adoption, and their headline number was benchmarked against the
slowest gossipsub conditions on their slowest network."*

## The 2-minute version (add these beats)

1. **The physics**: latency is free until 4,000ms, then a cliff — 99.9% to
   67% correct-head votes. Mean latency is the wrong KPI; the tail is
   everything. 61% of attestation misses happen on blocks that arrived *on
   time* — they're nodes in the propagation tail.
2. **The reframe**: late blocks are *published* late (3.65s vs 1.8s), not
   transported slowly. Even an infinitely fast network rescues only ~80% of
   them. So Optimum isn't selling throughput — it's selling *delay budget*:
   publish later at the same arrival, catch a richer MEV bid. dV/dt ≈ 7
   micro-ETH per millisecond.
3. **The economics**: $35/validator/yr at their 6×; going to their 20× adds
   only 11% because the deadline caps the budget. Sweet spot at 40–55%
   adoption; their seven partners are at 16.8% — below it. At saturation the
   MEV channel cannibalizes to zero and only public goods remain.
4. **The identification**: mainnet effects identified off the spec deadline
   plus RANDAO's random committee assignment — never operator-vs-operator.
   Hoodi tested with five designs; every clean break belonged to the
   never-treated control; the one non-null (size-penalty DiD) was *inverted* —
   congestion during their benchmark window, not offloading.
5. **The close**: *"their calculator and my study agree on the magnitude; the
   difference is my number survives cross-examination."*

## Answers to the questions you WILL get

**"Isn't your RD invalid? Blob count jumps at the cutoff."**
*"Correct — which is why I don't lead with the sharp RD. The running variable
is a sentry median, so measurement error smooths the discontinuity and makes τ
bandwidth-sensitive; covariate balance fails on blob count. I report that
openly and rest the claim on the dose-response, which needs no sharpness
assumption, and on RANDAO for exogeneity. Attenuation also means my estimates
are conservative."*

**"Why is the attester channel so small? Faster propagation helps attestations!"**
*"It helps them by pulling your node out of the propagation tail — worth about
$2.50 a year per validator, because source and target rewards, 74% of the
attestation slice, are structurally immune to latency. Only timely-head is
exposed, the miss rate is under 1%, and most of what remains is late
publication, which no transport fixes."*

**"Your Hoodi nulls just mean shadow mode. The product still works."**
*"Agreed — I say exactly that: absence of evidence under shadow mode, not
proof of absence. But it cuts both ways: there is then no deployment evidence
anywhere, and every quantitative claim rests on gateway self-telemetry from a
degraded testnet window. I priced the claim generously assuming it holds;
that's the $35."*

**"Why should anyone believe your counterfactual over their measurement?"**
*"Because their measurement is of a confounded cross-section and mine is of a
randomized natural experiment — and because when I force my model to predict
reality with the treatment turned off, it reproduces the measured baseline.
Their slope can't run that check. And note we agree on the magnitude anyway."*

**"Mean-field is standard. Why does adoption dynamics matter?"**
*"Because the value is a relative advantage in a timing race. Below ~40%
adoption you can't safely spend the delay; above it, adopters steal from each
other one-for-one. Mean-field prices the one world that can't exist — a lone
adopter with full network support. The observable consequence: their partners,
at 16.8% of stake, are below the sweet spot — which is actually a *sales*
argument they're missing."*

---

# Part 5 · Flash Cards

| Q | A |
|---|---|
| Attestation deadline? | 4,000ms — 1/3 of the slot, spec-set |
| Correct-head at 4–5s? | 67.4% (vs 99.9% on time) |
| Share of misses on on-time blocks? | 61% — tail nodes |
| Late blocks' median publication? | 3,656ms (normal: 1,811ms) |
| dV/dt? | ~7.0e-6 ETH/ms; V plateaus 0.051 ETH at ~3.5s |
| Orphan rate / hazard cliff? | 0.088%; 1.6% → 9.9% → 28.5% over 4.0–6.0s |
| Per-validator value at 6×? | $35.10 (C=32.76, A=2.48, D=0.68, B loses to C) |
| 6× → 20× gains? | +11% — deadline headroom binds |
| Floor model? | transit = min(observed, floor); 150ms = 6.7× Hoodi, 2.2× mainnet median |
| Adoption sweet spots? | 40% per-adopter ($20.82), 55% total ($7.95M); partners at 16.8% |
| Hoodi designs & result? | 5 designs, 0 detections; size-penalty DiD inverted (congestion) |
| Their calculator's uplift? | 0.0159 ETH per 32 ETH ≈ +1.7% APR ≈ our number ±20% |
| Why not compare operators? | Selection: latency proxies operator quality → OVB |
| Why WCR bootstrap? | Few clusters → CRVE over-rejects; impose null, Rademacher flips |
| B/C rule? | Mutually exclusive uses of the same ms: total = A + max(B,C) + D |
| The three bugs? | Tail-attester (distribution ≠ point); estimator mismatch (1× must recover baseline); silent-zero B (survivorship + fallback) |

---

*Sources: 216k-slot mainnet panel (Xatu, 2025-06), 487 network-days
Hoodi/Sepolia, MEV relay bid traces, 1.6M-proposal operator scan, libp2p peer
races. Pipeline: this repo; live: brian.biz/optimum. Every dollar figure is a
modeled counterfactual — mump2p is not deployed on mainnet.*
