# optimum — What Does Block Propagation Latency Actually Cost an Ethereum Validator?

A causal study of **block propagation latency → validator performance → staking revenue**, on **Ethereum mainnet**, built entirely from public data.

> **This program is on Ethereum, not Solana.** It is a read-only econometric analysis of Ethereum beacon-chain data. It holds no private keys, connects to no wallet, and **submits no transactions of any kind**. If you came here looking for transaction submission, SwQOS, Jito, or Jupiter — wrong repository. See [Scope](#scope-and-what-this-is-not).

---

## The question

Every propagation-acceleration pitch in Ethereum infrastructure — mump2p, RLNC overlays, faster gossip, better relays — rests on an unstated premise:

> *Blocks arriving faster makes validators earn more.*

That premise is almost never measured on mainnet. Vendors benchmark propagation latency on testnets and report *milliseconds*. Nobody converts milliseconds into **ETH**.

This repo measures the missing link:

**How much does a validator actually lose when the block arrives late — and what is that worth per day, week, month, and year?**

## The identification strategy

The Ethereum consensus spec tells an attester to cast its vote **4,000 ms into the slot** (1/3 of a 12-second slot), for whatever chain head it can see *at that instant*.

That single sentence gives us a natural experiment.

1. **A sharp, exogenous threshold.** If a block arrives at 3,950 ms, attesters can vote for it. At 4,050 ms, they cannot — they vote for the *previous* block instead (a "wrong head vote") and forfeit the `TIMELY_HEAD` reward. The cutoff is mandated by the protocol; nobody chose it, and no proposer can steer their block's gossip propagation to land on one side of it with millisecond precision. This is a textbook **regression discontinuity**.

2. **Random assignment of the exposed.** Attesters are assigned to slots by **RANDAO**. An operator cannot arrange to be assigned only to slots whose blocks arrive early. So *exposure to a late block is as-good-as-random from the attester's point of view.*

This is what makes the study causal rather than correlational. We are **not** comparing fast operators to slow operators — that comparison is hopelessly confounded (good operators are good at many things at once). We compare **the same validators across slots that, by lottery, happened to have an early vs. a late block.**

### What we find

The relationship is flat, and then it is a cliff — exactly where the spec says it should be. From one day of mainnet (2025-06-01):

| block arrival | slots | **correct head vote** | mean inclusion distance |
|---|---:|---:|---:|
| < 1s | 40 | 99.92% | 1.12 |
| 1–2s | 2,208 | 99.93% | 1.20 |
| 2–3s | 3,815 | 99.71% | 1.24 |
| 3.0–3.5s | 612 | 98.71% | 1.37 |
| 3.5–4.0s | 403 | 96.08% | 1.33 |
| **4–5s — past deadline** | 54 | **71.64%** | 1.79 |
| **> 5s** | 11 | **13.71%** | 1.66 |

Latency is nearly free right up to the deadline, and then catastrophic. That non-linearity is the whole story, and it is why "average propagation latency" is a misleading KPI: what matters is the *tail* — the share of your blocks that cross 4,000 ms.

## Outcomes measured

| Outcome | Definition | Source |
|---|---|---|
| **Correct head vote** | Attester voted for slot *s*'s block as head, vs. voting for *s−1* because the block hadn't arrived. Forfeits 14/64 of the base reward. | `canonical_beacon_elaborated_attestation.beacon_block_root` vs. the canonical root |
| **Inclusion distance** | `block_slot − slot`: how many slots before the attestation landed on chain | same table |
| **Missed attestation** | Assigned to a committee but never included in any block | `canonical_beacon_committee` (the duty roster — the denominator) |
| **Missed proposal** | Had a proposer duty but produced no canonical block (i.e. got reorged out) | `canonical_beacon_proposer_duty` vs. `canonical_beacon_block` |

**Treatment** — block arrival time — is the **median** `propagation_slot_start_diff` across ≥5 distinct Xatu sentry nodes, matched on the *canonical block root* (so a reorged competitor block can't pollute the timing).

## Pricing it

Latency destroys `TIMELY_HEAD` specifically, and only that:

| component | weight | survives a late block? |
|---|---|---|
| `TIMELY_SOURCE` | 14/64 | ✅ yes — it's about checkpoints, not the head |
| `TIMELY_TARGET` | 26/64 | ✅ yes — same |
| **`TIMELY_HEAD`** | **14/64** | ❌ **no — this is what a late block destroys** |

So a missed head vote costs **14/54 ≈ 25.9%** of an attester's income for that epoch.

The **proposer** side is far worse: a block that propagates too slowly gets **reorged out entirely**, forfeiting the consensus reward *and* the priority fees *and* the MEV. `run_study.py` reports both, converted into ETH and USD per **day / week / month / year**, for a single validator and for 10k / 100k-validator fleets.

## The counterfactual: what would adopting a propagation accelerator actually earn?

`src/counterfactual.py` prices the uplift from a claimed speedup through **three channels**, reported separately because they differ enormously in size *and* in credibility.

### First, the decomposition that keeps this honest

```
arrival_ms  =  t_publish            +  t_transit
               proposer's own          network transit —
               timing-game delay       the ONLY part a p2p
               (most of arrival!)      product can compress
```

A propagation product shrinks **only `t_transit`**. Applying a "6× faster" claim to the whole of `arrival_ms` would overstate the benefit by roughly an order of magnitude, because most of arrival time is the proposer *deliberately waiting* to accrue MEV — which no networking upgrade removes.

We recover `t_transit` from the spread of Xatu's geographically distributed sentries: `prop_spread_ms = p90_arrival − min_arrival`. (Since `min` is the first *sentry* sighting, it already contains one hop — so this is a **lower** bound on true transit, making our uplift estimates conservative.)

### The three channels

| channel | mechanism | captured by adopter? |
|---|---|---|
| **A. Attester (receive-side)** | mump2p accelerates blocks *into my* nodes → *my* attesters miss fewer head votes on *everyone else's* late blocks, every epoch | **100%** |
| **B. Proposer reorgs** | my blocks arrive sooner → fewer orphaned → I keep the block (consensus + tips + MEV) | **100%** |
| **C. MEV — delay budget** | **the big one, see below** | **100%** |

**Channel C is the actual product.** Measured from Xatu relay bid traces (`src/mev.py`), the value of a block to its proposer **nearly doubles across the slot** — 0.027 ETH at t=0, 0.051 ETH by 3.5s — and then **plateaus dead**, because builders stop bidding on a block that cannot beat the attestation deadline.

Proposers already delay publication to harvest this. What stops them delaying further is reorg risk. So:

> **If transit is Δ ms faster, you publish Δ ms later and land at the same arrival time — identical reorg risk, strictly better bid.**
>
> Gain = `dV/dt × Δ` per block proposed. **Optimum is not selling throughput. It is selling delay budget.**

### Two arithmetic traps, both guarded by tests

1. **Channel A runs the *other* way than you'd think.** The obvious model — "my blocks reach attesters faster" — is a **public good**: the attesters of my slots are a random RANDAO committee, i.e. overwhelmingly *other operators'* validators. An operator with 3% of stake would capture 3% of it. The *private* benefit is **receive-side**, and it applies on every epoch rather than only the ~0.003% of slots I propose. Getting this backwards overstates Channel A by ~30× for a large operator.

2. **B and C cannot be added.** They are mutually exclusive uses of the same saved milliseconds — *bank* them as earlier arrival (B), or *spend* them as later publication (C). Not both.
   **Total = `A + max(B, C)`, never `A + B + C`.**

### Scenarios, not a vendor number

Optimum's stated improvement is **6×** — but that figure compares mump2p on the **Hoodi testnet** against an ethPandaOps gossipsub baseline **on a different network**. It is a marketing claim, not a mainnet measurement. So the study reports **6× / 3× / 2×** side by side and lets the reader discount.

## Robustness

Because a discontinuity you can't defend is worthless, the run reports:

- **Bandwidth sensitivity** — the estimate must not be an artifact of one window (500 → 2000 ms).
- **Density (McCrary-style) test** — if proposers could *manipulate* arrival time around the deadline, the density would jump at the cutoff and the design would be invalid.
- **Covariate balance** — blob count, tx count, and payload bytes must *not* jump at 4,000 ms. If they did, the "effect" might be theirs.
- **Placebo cutoffs** — the same RD run at 2000/2500/3000/5500 ms, where no spec threshold exists. These must come back ≈ 0. If the estimator finds discontinuities in noise, nothing else it says can be trusted.
- **Cluster-robust SEs + wild cluster bootstrap** (WCR, Rademacher weights) — errors are correlated within a day (shared network conditions, blob demand), so treating ~650k slots as independent draws would produce absurdly tight intervals.

### The estimate is a lower bound (and why that's fine)

The running variable is measured with error. We observe the **median arrival across Xatu's sentry nodes**, not the arrival at each individual attester — and the attester is what the deadline actually binds on. Classical measurement error in an RD's running variable **smears the discontinuity**, attenuating the estimated jump toward zero.

So the reported τ is a **conservative lower bound** on the true causal effect of a block missing the deadline. The real effect is at least this large. We state this rather than quietly hoping nobody asks.

The `MIN_SENTRIES` setting (default 5) is the lever here: requiring more independent observers shrinks the measurement error in the median, at the cost of dropping slots.

## Frameworks and stack

| | |
|---|---|
| **Chain** | **Ethereum** (beacon chain / consensus layer). Not Solana. |
| **Data** | [Xatu](https://github.com/ethpandaops/xatu-data) — ethPandaOps' public beacon-chain dataset, plain parquet over HTTPS, **no authentication** |
| **Query engine** | [DuckDB](https://duckdb.org) — reads the parquet directly, no warehouse, no ETL |
| **Analysis** | Python 3.12+, NumPy, pandas, pyarrow |
| **Econometrics** | Hand-rolled: sharp RD with local-linear + triangular kernel, CR1 cluster-robust sandwich, wild cluster bootstrap. No black boxes — every estimator is in `src/estimators/` and unit-tested against planted effects. |
| **Tests** | pytest |

**Zero infrastructure.** There is no database to stand up, no node to sync, no API key to obtain. The entire study is SQL pointed at public URLs.

## Layout

```
config.py              .env loading (no secrets — by design there are none)
run_study.py           orchestrator: panel -> estimates -> robustness -> ETH/USD
src/
  xatu.py              Xatu URL scheme, DuckDB connection, consensus constants
  panel.py             slot-level panel construction (the SQL)
  revenue.py           consensus reward weights -> ETH -> USD per day/week/month/year
  estimators/
    inference.py       CR1 cluster-robust SEs + wild cluster bootstrap
    rdd.py             sharp RD at the 4s deadline, density + placebo tests
tests/                 unit tests (see below)
```

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # optional: every setting has a working default
./.venv/bin/python run_study.py --start 2025-04-01 --end 2025-06-30
```

Configuration is read from `.env` (see `.env.example`). All of it is public, non-secret tuning: date range, DuckDB memory/threads, bootstrap replications, seed, and an ETH price used *only* to render USD in the report — it never enters an estimate.

### A note on disk

Raw Xatu is ~1.5 GB/day (the attestation table alone is ~800 MB, nearly all of it the `validators` array column). A 90-day study touches ~140 GB.

The pipeline therefore **streams**: download one day → aggregate to one row per slot → **delete the raw file** → next day. Peak disk stays around 2 GB. This is not an optimisation, it's a requirement — the deployment host has 15 GB free.

### A note on DuckDB + httpfs

Do **not** query the fat Xatu tables directly over HTTP. DuckDB's `httpfs` range requests corrupt large ZSTD column chunks and fail with `ZSTD Decompression failure`. Download first, then query the local file. `src/panel.py` does this.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

The tests that matter are the **negative** ones — an estimator that finds effects is worthless unless it also *fails* to find them when they are absent:

- `test_no_false_positive_when_no_jump` — plant **zero** discontinuity (but *different slopes* either side) and confirm the RD reports none. Catches the classic bug of mistaking a kink for a jump.
- `test_wild_bootstrap_size` — under a true null, p-values must be roughly uniform, in the few-cluster regime where the plain cluster-robust t-test is known to over-reject. If this fails, every p-value in the study is a lie.
- `test_clustered_se_exceeds_naive_se` — if clustering were silently a no-op, every interval would be too tight.
- `test_recovers_planted_discontinuity` — plant τ = −0.25 and recover it.
- `test_url_does_not_zero_pad_month_and_day` — Xatu serves `/2025/6/1.parquet`; `/2025/06/01.parquet` is a **silent 404**, which mid-study looks identical to "no data that day".

## The Hoodi postscript: we tested the network where mump2p actually runs

mump2p never reached mainnet, but it did deploy on the **Hoodi testnet**. So the
pipeline was re-run there (`event_study.py`), with **Sepolia as a never-treated
control**: 487 network-days of daily propagation physics bracketing the
deployment window. Result, in one line:

> **No positive propagation signature exists on the only network where mump2p is
> deployed** — every clean structural break in eight months of data belongs to
> the *control* network, and the deployment window itself coincides with Hoodi's
> worst transit of the study period. Details in [`FINDINGS.md`](FINDINGS.md).

Testnet dollars are fiction (free ETH, no MEV market), so the pricing layer
gates itself off for any `NETWORK` other than mainnet.

## Scope, and what this is *not*

This repository began life as a different study: a **staggered difference-in-differences** estimating the effect of **mump2p adoption** on the seven publicly-named Optimum partner operators (Everstake, P2P.org, Kiln, Luganodes, Ebunker, InfStones, Blockdaemon).

**That study cannot be run, and this repo does not pretend otherwise.** The reasons are documented in [`FINDINGS.md`](FINDINGS.md), but in short:

1. **No staggering.** All seven operators were named in a *single press release on 2025-06-24*. Callaway–Sant'Anna is an estimator *for staggered adoption*; with one cohort it degenerates to a 2×2 DiD. There is no timing variation to exploit.
2. **No treatment in the data.** mump2p has never run on Ethereum **mainnet** — it is a Hoodi/private-testnet product. Xatu is mainnet. So the treated operators' mainnet validators, the only ones observable, were **never treated**. The treatment indicator is identically zero across the entire panel.

Rather than estimate the effect of a press release, this repo measures the thing that propagation-acceleration products *claim to improve*, on the network where the money actually is. It is the **first stage** that any such claim depends on.

### It is not a trading bot

No keys. No RPC. No wallet. No `SWQOS_ENDPOINT`, no Jito, no Jupiter, no slippage. Nothing here signs or broadcasts anything. It reads public parquet files and runs regressions.

## Data source & credit

All data from **[Xatu](https://github.com/ethpandaops/xatu-data)** by **[ethPandaOps](https://ethpandaops.io)**, released publicly and freely. This study is only possible because they publish it.
