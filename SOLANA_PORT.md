# Porting the propagation-value study to Solana

## The one-paragraph verdict

The **economic question ports perfectly** — "what is faster block propagation
worth to a validator?" is arguably *more* alive on Solana than Ethereum
(Firedancer, DoubleZero, Jito ShredStream, Turbine tuning are all selling
exactly this). The **outcome side ports well and is free**: vote latency, skip
rate, forks, and rewards are all derivable from the ledger via public RPC. But
the **identification core does not survive the move**, and the **treatment
variable has no public dataset**. There is no Xatu for Solana. The 4-second
attestation deadline (our sharp RD cutoff) and RANDAO (our randomizer) are
Ethereum-specific and have no Solana equivalent. So this is not a rewrite of the
SQL against a new URL — it is a re-identification of the same economic estimand
against a different consensus machine. Below is the mapping, the new
identification strategy, and exactly where each byte of data comes from.

---

## 1. Why a naive port fails — the three load-bearing Ethereum facts

The whole study stands on three things that Solana does not have:

| Ethereum fact | Role in the study | Solana equivalent |
|---|---|---|
| **4,000 ms attestation deadline** (1/3 of a 12 s slot) | The sharp, spec-mandated **RD cutoff** in `src/estimators/rdd.py`. Blocks at 3,950 vs 4,050 ms are identical except votability. | **None.** Solana has no per-slot vote deadline. Votes land continuously; the reward penalty for lateness is a *smooth kinked ramp* (Timely Vote Credits), not a cliff. → the RDD becomes a **Regression Kink Design (RKD)**. |
| **RANDAO committee assignment** | The **randomizer**. Attesters are lottery-drawn into one committee per epoch, so exposure to a late block is as-good-as-random. | **None.** Every Solana validator votes on *every* slot, and the leader schedule is **public an epoch ahead** (stake-weighted, deterministic). No committee lottery → the "random exposure" argument must be rebuilt (see §4). |
| **Xatu** (ethPandaOps free parquet, per-sentry block sighting times) | The **treatment variable**: `propagation_slot_start_diff`, median across sentries = network arrival time. | **None public.** Nothing publishes per-node shred arrival times. This is the single biggest lift — you either self-collect it (run a sentry fleet) or proxy it from on-chain signals (§3, §5). |

Everything else — the revenue model, the counterfactual channels, the inference
stack (cluster-robust + wild bootstrap), the dose-response idea — ports with
edits, not redesigns.

---

## 2. Concept mapping (Ethereum → Solana)

| Study concept | Ethereum (current code) | Solana port |
|---|---|---|
| Slot | 12 s, `SECONDS_PER_SLOT` | **400 ms** nominal; ~432,000 slots/epoch (~2–3 days) |
| Unit of analysis | canonical slot | **leader slot** (and the validator-slot for the vote outcome) |
| Proposer | RANDAO-selected, 1 slot | **Leader**, deterministic schedule, **4 consecutive slots** each |
| Block propagation | gossipsub mesh | **Turbine** (FEC-shredded, stake-weighted retransmit tree) |
| Treatment (running var) | `arrival_ms` = median sentry sighting | **shred/block arrival time** — must be self-measured (§3), OR proxy: `vote_latency` (on-chain), `block_completion_slot` |
| The cutoff | 4,000 ms deadline (sharp RD) | **Timely Vote Credit ramp** — full 16 credits if vote lands ≤2 slots late, −1 credit per extra slot to a floor of 1 at ≥17 (SIMD-0033). A **kink at latency=2**, not a cliff → **RKD**. |
| `correct_head_rate` | voted for right head at 4 s | **vote credit earned** = `max(1, 16 − (landed_slot − voted_slot − 2))`; the analog of "timely head" is *timely vote latency* |
| `inclusion_distance` | `block_slot − slot` | **vote latency** = `landed_slot − voted_slot`, parsed from vote txns (identical shape, on-chain) |
| `missed_attestation` | assigned but never included | **missed/late vote** — validator's vote for slot s never lands or lands past the credit floor |
| `missed_proposal` | proposer duty, no canonical block | **skipped slot** — leader assigned but produced no rooted block (`getBlockProduction`) |
| `orphaned` (Channel B) | block seen, lost fork choice | **abandoned/forked leader block** — produced but not rooted; leader loses priority fees + Jito tips |
| Block value V(t) / MEV (Channel C) | relay bid trace, dV/dt | **priority fees + Jito tips**; Solana's timing game is weaker (400 ms slots, Gulf Stream pre-forwarding) but nonzero |
| Bandwidth (Channel D) | ~3.8 TB/node/yr gossip | Turbine + repair + gossip egress; larger (higher throughput) |
| Revenue units | ETH, base_reward spec formula | **SOL**; inflation reward ∝ **vote credits earned** (this is the clean lever — see §6) |
| The "product" being valued | Optimum / mump2p | **Firedancer, DoubleZero, Jito ShredStream** — real, shipping propagation accelerators |
| Never-treated control (Hoodi/Sepolia analog) | testnets | Solana testnet/devnet, or pre/post a DoubleZero/Firedancer rollout window |

---

## 3. The data problem, stated honestly

The study is cheap on Ethereum for one reason only: **Xatu already ran the sentry
fleet for you** and publishes per-node block sighting times as free parquet.
Solana has no such public dataset. Your options, cheapest first:

### Path A — On-chain proxies only (free, runnable this week)
Derive everything that lives in the ledger. You get a *complete outcome panel*
and a *proxy treatment* without any infrastructure:

- **Vote latency** (the RKD running variable) — parse every vote transaction:
  each `TowerSync`/`VoteStateUpdate` instruction lists the slots voted on; the
  block it landed in is the landing slot. `latency = landed − voted`. This is
  the Solana `inclusion_distance`, and it is *directly* the input to the credit
  schedule — a better-instrumented running variable than Ethereum's sentry
  median.
- **Skip rate / forks** — `getBlockProduction` (assigned vs produced per leader),
  plus walking `getBlocks` for gaps and `getBlock().parentSlot` for fork skips.
- **Rewards** — `getInflationReward` (per validator per epoch) and `getBlock`
  `rewards[]` (fee/voting/staking split).
- **Leader schedule** — `getLeaderSchedule` (deterministic, an epoch ahead).
- **Stake / credits** — `getVoteAccounts` gives `epochCredits` and stake.

What Path A **cannot** give you: true network transit time. Vote latency
conflates "the block was slow to reach me" with "my vote was slow to land in the
next leader's block" — publication vs transit, the exact decomposition
`src/panel.py` fought to keep separate. So Path A identifies the *value of vote
latency*, not cleanly the *value of transit*. Good enough for the RKD and the
credit-revenue channel; not enough for a clean transit counterfactual.

### Path B — A Geyser vantage point (infrastructure ALREADY EXISTS in ../memeorator)
To recover arrival time you timestamp block/entry arrivals, as Xatu does with
beacon blocks. **The sibling repo `../memeorator` already has a working
capture stack for this** — reuse it rather than building from scratch:

- **Yellowstone Geyser gRPC client, in Python, live.** `../memeorator` subscribes
  to a Yellowstone endpoint (`parallel-titan.fleet.hellomoon.io:889`, HelloMoon)
  at `PROCESSED` commitment and **already stamps first-seen arrival**
  (`time.perf_counter()` / `time.time()` per signature — see
  `grad_shred_shadow.py`, `grad_shred_micro_collector.py`). Compiled protos are in
  `grpc_proto/` (`geyser_pb2.py`, `solana_storage_pb2.py`).
- **The proto exposes exactly what we need.** `SubscribeUpdate.created_at`
  (server-side observation timestamp on every message), plus subscribable
  streams for **slots** (`SubscribeUpdateSlot`, with `interslot_updates`),
  **blocks / block-meta** (`SubscribeUpdateBlockMeta.parent_slot` → fork/skip
  detection), **entries** (`SubscribeUpdateEntry` — groups of shreds, the closest
  thing to shred-level arrival granularity), and **transactions** (for the vote
  latencies of §3 Path A, live instead of by RPC replay).
- So a single-vantage **arrival-relative-to-slot-start** series — the direct
  analog of one Xatu sentry's `propagation_slot_start_diff` — is essentially
  already collectible. Point the existing client at slot+entry subscriptions and
  log `created_at` + local recv per slot.

**The one caveat that governs the whole design — independent vantage points.**
Xatu's power is the *spread* across independent sentries (min/median/p90), which
is what isolates transit from publication. HelloMoon's endpoint is **one
vantage, and not your node** — its `created_at` is HelloMoon's fleet observing
the slot, plus delivery latency to you. That gives a clean *single-vantage
arrival* (good for a first-cut dose-response and for crossing against on-chain
vote latency / skips), but **not** the transit spread. To reconstruct the Xatu
fleet you run the same memeorator client against **N independent Geyser
endpoints in different regions** (HelloMoon + Helius + Triton + optionally a
self-hosted agave+Yellowstone node). The spread across *independent* providers ≈
transit; the spread across sockets to the *same* provider is just delivery
jitter and must not be used. That distinction is the Solana equivalent of Xatu's
"require ≥5 distinct `meta_client_name`" gate.

- **`solana-gossip spy`** — observes gossip (votes, contact info), *not* block
  shreds (Turbine), so not sufficient alone for transit.
- **Jito ShredStream** — an alternative low-latency single vantage; same
  one-geography limitation as a single Geyser endpoint.

Budget reality, revised down: because the client already exists, the marginal
cost is **N Geyser subscriptions in distinct regions** (several are free-tier or
cheap; a self-hosted node is the expensive option) plus a small amount of
capture/alignment glue. The treatment variable Ethereum got for free is now a
few days of wiring, not a validator buildout.

### STATUS: sentry #1 is deployed and collecting (2026-07-15)
`deploy/sentry_collect.py` is running on the **frankfurt** box against HelloMoon's
`parallel-titan.fleet.hellomoon.io:889` Geyser feed (plaintext on that host — the
endpoint resolves to the box's own IP, i.e. HelloMoon fleet infra is local/LAN to
this node, which makes it an unusually good vantage point). It subscribes to the
slot-status stream with `interslot_updates` and logs, per slot, the `created_at`
timestamp of every transition to `~/solana-optimum/data/slots_YYYYMMDD.jsonl`.

First 20 s of live data already yields the core metric:

    shred-download duration  = t(SLOT_COMPLETED) − t(SLOT_FIRST_SHRED_RECEIVED)
    → median 357 ms, p10 297, p90 391 (capped near the 400 ms slot)
    inter-slot cadence median 399 ms (nominal 400 → clock is sane)

This interval is **delivery-latency-free** (both stamps are the same node's
`created_at`), so it is a clean per-slot propagation-download measurement — the
Solana analog of Xatu's transit, from one real vantage point. `SLOT_DEAD` and
`parent`-gap give skips/forks.

Two collectors now run persistently on frankfurt (under `run_service.sh`,
restart-on-crash, `setsid` so they survive logout):
- **`sentry_collect.py`** (`SENTRY_NAME=titan`) → `data/slots_titan_*.jsonl`.
  Live: shred-download median ~350 ms, p90 ~400 ms.
- **`jito_tip_collect.py`** → `data/jitotips_titan_*.jsonl` — the Channel-C
  outcome. Live: tips/slot median ~0.006 SOL, p90 ~0.02, max ~0.14 SOL; ~230
  tip-txs/s. Joins to the sentry on **slot number** (not timestamp — see below).

**Two operational notes learned in deployment:**
- The `elite-cache:2096` shred feed is **transaction-only** — it does NOT carry
  the slot-status stream, so it cannot be a second slot-status vantage. A genuine
  independent transit spread therefore needs a **non-HelloMoon** provider
  (Helius/Triton in a different region), not a second HelloMoon socket.
- HelloMoon's `created_at` clock runs ~95 s **ahead** of frankfurt's wall clock.
  Harmless: the COMPLETED−FIRST_SHRED interval is same-clock (cancels), and the
  sentry↔jito join is on slot number. But never compare `srv_ms` across
  different feeds, and prefer slot-number joins over timestamp joins.
- Reboot persistence is NOT yet set up (processes are `setsid nohup`, not a
  systemd unit / `@reboot` cron) — add that if the box may reboot.

---

## 8. Channel C on Solana — the Jito-tip timing hypothesis

**Hypothesis (yours):** Jito tips are MEV tips; a leader that produces its block
*later* accumulates more of them. This is the direct heir to Ethereum's Channel C
("faster transport buys delay budget; delay buys MEV"), and on Solana it is
**more directly testable**, for one reason: **Jito tips are on-chain.** Where
Ethereum needed private relay bid traces to reconstruct V(t), Solana records
every tip as a transfer to one of the 8 known Jito tip-payment accounts, inside
the block itself.

How the mechanism differs from Ethereum (state these, they matter):
- **No single publish instant.** A Solana leader holds its slot(s) for a 4-slot
  (~1.6 s) window and *streams* shreds continuously as it packs. "Delay" is not
  "wait then publish" — it is producing/finalizing the block later in the window,
  seeing more of Jito's continuous bundle auction before you seal.
- **The auction is the value curve.** Jito's block engine runs an off-chain
  blockspace auction (~200 ms ticks) and forwards top-tip bundles to the leader.
  Later sealing ⇒ strictly more bundles seen ⇒ weakly higher captured tips. That
  is your V(t), and it is observable.
- **The binding constraint is the same as Ethereum's.** Seal too late and the
  block's shreds don't propagate before the next leader/committee moves on →
  the block is skipped or forked out (`SLOT_DEAD`), and the leader loses
  everything (priority fees + tips). So there is a real, bounded delay budget,
  just a much shorter one (hundreds of ms inside a 400 ms slot).

**The test (all data now in reach):**
1. Per leader block: total Jito tips (on-chain, sum of transfers to the tip
   accounts) — the outcome.
2. Block production timing — from the sentry (`FIRST_SHRED`→`COMPLETED`), plus
   intra-block tx/CU position — the running variable ("how late was it sealed").
3. Regress tips on sealing-lateness, controlling for network load (total CU,
   bundle-auction depth). The hypothesis predicts **d(tips)/d(lateness) > 0**.

**The identification trap (same shape as the Ethereum study):** the cross-section
is confounded — busy slots have both more tips *and* later sealing, so a naive
positive slope is mostly congestion, not causal delay budget. You want the causal
`d(tips)/d(lateness)`. Cleanest leverage available on Solana:
- **Within-leader-window variation:** a leader owns 4 consecutive slots; compare
  tips captured across positions 1–4 of the *same* leader in the *same* window
  (leader fixed, load nearly fixed) — later positions have seen more auction.
- **The skip/fork hazard as the cost side:** measure P(`SLOT_DEAD` | lateness)
  from the sentry — this is Channel B, and it is exactly what caps how far a
  rational leader spends the delay budget. Tips gained vs skip-risk lost is the
  Solana version of "B and C are mutually exclusive uses of the same ms."

This is a genuinely sharper Channel C than the Ethereum original — the value
curve and the timing are both observed rather than modeled. It is worth treating
as the headline result of the Solana port, not a footnote.

### Path C — Indexed / third-party (fills gaps, some paid)
- **Trillium (trillium.so)** — per-epoch validator MEV (Jito tips), priority
  fees, rewards, skip rate. The best single source for the *revenue* side.
- **Jito APIs** — MEV tips, bundle data, ShredStream.
- **validators.app, stakewiz.com, solanabeach, Solana Foundation reports** —
  validator metadata, skip rates, stake.
- **Dune / Flipside / Helius / Triton / QuickNode** — indexed blocks, votes,
  rewards at scale so you are not walking `getBlock` yourself for a whole epoch
  (that is the Solana equivalent of Xatu's 800 MB/day attestation table — one
  epoch is ~432k blocks; parsing every vote txn is the heavy job).

**Recommended sequencing:** Path A first (it is free and proves the RKD + credit
channel end-to-end), Path C for the revenue channel, Path B only if you need a
publishable *transit* counterfactual rather than a *vote-latency* one.

---

## 4. Re-identification: what replaces the RD + RANDAO

Ethereum's clean natural experiment (sharp cutoff × committee lottery) has to be
rebuilt from Solana's different primitives. Three candidate designs, best first:

1. **Regression Kink Design on the Timely-Vote-Credit ramp.** The credit
   schedule is a deterministic, spec-mandated piecewise-linear function of vote
   latency with a **kink at latency = 2 slots**. RKD identifies the causal effect
   of latency on credits (hence SOL) off the *change in slope* at the kink,
   under the assumption that the density of latency is smooth there (the RKD
   analog of the McCrary test — Card, Lee, Pei, Weber). This is the direct
   heir to `rdd.py`: same local-linear + triangular-kernel + cluster-robust
   machinery, but the estimand is the kink (Δslope) not the jump (Δlevel).
   The mechanism is *sharper* than Ethereum's because the credit-latency
   relationship is written into the protocol, not inferred.

2. **Skip-rate dose-response** (the heir to the attestation dose-response
   `_dose_response_fn`). Regress P(leader block is skipped/forked) on measured
   arrival/transit (needs Path B) or on block-completion latency (Path A). No
   sharpness assumption — this is the curve the whole counterfactual rides on.

3. **Leader-geography as the exogeneity argument.** Since there is no committee
   lottery, exogeneity comes from a different place: a *voter* does not choose
   which leader is scheduled when, and the leader schedule is fixed by stake a
   full epoch ahead — orthogonal to any single voter's transient network state.
   You compare the *same validator* across slots whose leaders happened to be
   near vs far, fast vs slow. The threat is fixed geography (a validator
   co-located with big leaders systematically sees early blocks); control for it
   with validator and leader fixed effects, and cluster by leader. This is
   weaker than RANDAO and must be stated as such — the honesty section of
   `STUDY_GUIDE.md` has the template.

**Say it like this:** *"Ethereum handed me a cliff and a lottery. Solana hands me
a protocol-defined kink in the reward function and a schedule fixed a day in
advance. I trade the committee lottery for validator/leader fixed effects, and I
trade the sharp RD for an RKD on the credit ramp — which is actually written into
the consensus rules rather than inferred from behavior."*

---

## 5. Concrete architecture

Mirror the existing layout; swap the data layer and the identification layer,
keep inference and reporting.

```
config.py                 # + SOLANA_RPC_URL, epoch range instead of date range
src/solana.py             # NEW — replaces xatu.py: RPC client, slot/epoch math,
                          #        TVC constants, leader schedule. (seed written)
src/panel.py              # rewrite build_slot_panel() → build_leader_slot_panel()
                          #   + build_vote_latency_panel()
src/estimators/rdd.py     # → rkd.py: kink at latency=2, estimand = Δslope
src/estimators/inference.py  # KEEP AS-IS (cluster-robust + wild bootstrap port 1:1)
src/revenue.py            # rewrite: SOL, inflation ∝ vote credits, no 64-weight
src/counterfactual.py     # keep channel structure; re-fit A=credit,
                          #   B=skip/fork, C=Jito-tip timing, D=Turbine bandwidth
src/sentry/               # NEW (Path B only) — shred-arrival capture fleet
tests/                    # port; TVC schedule + skip-rate have clean unit tests
```

Constants you will need (verify against current agave source before publishing):

- `SLOTS_PER_EPOCH = 432_000`, slot ≈ 400 ms → `SLOTS_PER_YEAR ≈ 78.8M`
- Timely Vote Credits: `MAX_CREDITS = 16`, `GRACE_SLOTS = 2`, floor `= 1`;
  `credit(latency) = max(1, 16 − max(0, latency − 2))`
- `NUM_CONSECUTIVE_LEADER_SLOTS = 4`
- Inflation: ~4.4%/yr decaying 15%/yr toward 1.5% long-term; validator reward ∝
  (their credits / total credits) × stake share — credits are the lever latency
  actually moves.

---

## 6. Revenue: the clean part

Solana makes one thing *cleaner* than Ethereum. On Ethereum, latency hits
exactly the `TIMELY_HEAD` 14/64 slice. On Solana, latency moves **vote credits
directly**, and inflation rewards are distributed in proportion to credits
earned. So the marginal SOL cost of one extra slot of vote latency is:

```
Δcredits = 1 (per vote, in the ramp region 2 < latency ≤ 17)
share_lost ≈ Δcredits / MAX_CREDITS = 1/16 per affected vote
SOL/yr lost ≈ (fraction of votes in ramp region) × (1/16) × validator_inflation_SOL/yr
```

No `base_reward` spec formula, no 64-weight bookkeeping — it collapses to
"credits are money, latency burns credits." The proposer/leader side (skip +
fork) maps onto lost **priority fees + Jito tips**, which is where the real money
is on Solana, just as Channel C was on Ethereum.

---

## 7. What to build first (proposed order)

1. `src/solana.py` seed (done — RPC client, epoch/slot math, TVC schedule).
2. Vote-latency panel from Path A → validates the credit schedule end-to-end on
   one epoch, zero infrastructure.
3. RKD at the credit kink (`inference.py` ports unchanged).
4. Skip-rate dose-response + `getBlockProduction`.
5. Revenue in SOL (credit channel + Jito-tip channel via Trillium).
6. Only if a *transit* (not vote-latency) claim is needed: stand up the Path B
   sentry fleet.

Open decision for you: **do you want a publishable transit counterfactual
(requires the Path B fleet), or is a vote-latency / skip-rate study on free
on-chain data enough?** That fork determines whether step 6 happens and roughly
whether this costs $0 or ~$X00/month in nodes.
