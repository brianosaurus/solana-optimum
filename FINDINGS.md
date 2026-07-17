# Why the mump2p staggered difference-in-differences cannot be run

This repository was originally scoped as a **staggered difference-in-differences** study estimating the causal effect of **mump2p adoption** on attestation effectiveness, inclusion distance, and missed proposals — using the seven publicly-named Optimum partner operators (Everstake, P2P.org, Kiln, Luganodes, Ebunker, InfStones, Blockdaemon) as treated units against comparable non-adopters, with a Callaway–Sant'Anna estimator, operator-clustered standard errors, wild bootstrap, event-study plots and placebo tests.

**That study cannot be run.** Not "is underpowered", not "has a confound to manage" — cannot be run. This document records why, because the reasoning is worth more than the study would have been.

---

## 1. There is no staggering

All seven operators were announced in a **single press release on 2025-06-24**, in a single sentence:

> "seven of the top Ethereum validators and node operators—**Kiln, P2P.org, Everstake, Blockdaemon, Infstones, Luganodes, and Ebunker**—have partnered to test and deploy its first product to market: OptimumP2P"

Carried same-day by [CryptoSlate](https://cryptoslate.com/press-releases/optimum-debuts-optimump2p-testnet-with-support-from-ethereums-largest-node-operators/), [The Defiant](https://thedefiant.io/news/press-releases/optimum-debuts-optimump2p-testnet-with-support-from-ethereums-largest-node-operators), [CryptoBriefing](https://cryptobriefing.com/optimum-debuts-optimump2p-testnet-with-support-from-ethereums-largest-node-operators/), and [Decrypt](https://decrypt.co/326655/major-validators-testnet-ethereum-bottleneck).

The Callaway–Sant'Anna estimator exists **to handle staggered adoption**: it estimates group-time average treatment effects ATT(g,t), indexed by adoption cohort `g`, then aggregates them. With **one cohort**, it degenerates to a plain 2×2 DiD. There is no timing variation to exploit, and therefore no staggered DiD — the entire methodological apparatus of the proposed design has nothing to consume.

The only apparent "staggering" is InfStones publishing its own post a week later (2025-07-01) and Everstake tweeting ~3 months later. Both restate participation in **the same private testnet**. They are not new adoption events.

## 2. The treatment never happened in the data — the fatal one

**Xatu is Ethereum mainnet data. mump2p has never run on Ethereum mainnet.**

Every source scopes the product to testnet:

- The June 2025 launch release: *"The team will continue onboarding node operators as OptimumP2P **prepares to roll out** on Ethereum's **Hoodi testnet** this summer."*
- Optimum's own results posts: *"Acceleration: mump2p Early Results on **Hoodi Testnet**"* (2025-09-23) and *"Behind the Metrics: mump2p's 6x Latency Win on Ethereum **Hoodi Testnet**"* (2025-10-16).
- Luganodes: *"currently participating in the OptimumP2P **private testnet**"*.
- InfStones: *"Optimum Selects InfStones as OptimumP2P's **Private Testnet Partner**"*.
- Obol (2025-10-07): *"currently **testing** Optimum with a **potential** integration in mind… once mainnet goes live."*
- Optimum COO Kent Lin, quoted 2026-04-24 — **ten months after the launch PR**: *"It **plans to support** the service on the Ethereum mainnet **within one to two months**."*

Optimum's blog carries 16 posts from April 2025 through May 2026. **None announces a mainnet launch.**

Therefore: the treated operators' **mainnet** validators — the only validators that appear in Xatu — were **never treated**. The treatment indicator is identically zero across the entire panel.

You cannot estimate the effect of a treatment that did not occur in your sample. Any non-zero coefficient would be noise, and there would be no way to distinguish it from a real effect. This is the one result you must not walk into a meeting with.

Note this also kills the weaker fallback estimand ("the effect of the *announcement*"): with all seven announced on one date, there is still no staggering, and one would be measuring a press release's effect on validators that changed nothing.

## 3. Two of the seven never claimed adoption at all

- **Ebunker** and **Blockdaemon**: no public statement of their own could be found. Blockdaemon's press page (26 items, Jul 2025 – Jul 2026) contains zero Optimum mentions.
- **Kiln**: appears only as a technology-endorsement quote inside Optimum's own PR. No claim of adoption anywhere on kiln.fi.
- **P2P.org**: only a canned quote in Optimum's PR — *"OptimumP2P represents exactly the kind of foundational advancement **we look for**"* — which is aspirational, not an adoption claim.

Treatment assignment would rest entirely on a **vendor's press release naming them**.

## 4. Even on mainnet, the mechanism is weak

mump2p is an RLNC-coded pub/sub **overlay that runs alongside gossipsub**, not in place of it — blocks continue to propagate over gossipsub in parallel, so the overlay is not load-bearing. It targets **blocks, blobs and transactions**; attestation *gossip* is not the target.

Optimum's own benchmark methodology confirms the measurement is a **passive shadow**: *"every block propagates through both mump2p and Gossipsub simultaneously,"* with gateways recording block arrival times. That measures **when a block reaches an Optimum gateway** — not attestation inclusion distance, not missed proposals, not any validator-side outcome.

The headline "6x" also compares mump2p on **Hoodi** against ethPandaOps' gossipsub baseline **on a different network**.

## 5. Two further traps found while building the operator mapping

Investigating the Lido `NodeOperatorsRegistry` (`0x5503…28d5`) surfaced two problems that would have wrecked the DiD independently:

- **Kiln currently has ZERO live Lido curated validators** (`usedSigningKeys` == `stoppedValidators` == 10,579). A Lido-registry-based mapping would have handed us an **empty treatment cell** for one of the seven.
- **Luganodes is not a Lido node operator at all**, in any of the three modules.
- Lido's allocator equalises deposits, so every active curated operator sits at ~7,575 live validators — **near-zero cross-operator variance** in Lido-side fleet size.

---

## What was built instead

The premise underneath the original design — *"faster block propagation makes validators earn more"* — is the thing that every propagation-acceleration product assumes and **nobody has measured on mainnet**. Optimum's own March 2026 post ("Optimizing a $100B Market: Effects of Latency Reduction on ETH Staking Revenue") is reaching for exactly this number.

So this repo measures **that**: the causal, mainnet dose-response from **block propagation latency → attestation outcomes → ETH**, identified off the spec-mandated 4-second attestation deadline (a sharp RD) and RANDAO's random assignment of attesters to slots.

See [`README.md`](README.md). It is the *first stage* that the mump2p claim depends on — and unlike the DiD, it is a study that can actually be run.

## Postscript: we went to the network where mump2p DOES run

Since mainnet never carried the treatment, we redid the physics on **Hoodi** —
the testnet where mump2p actually deployed ("rolling out on Ethereum's Hoodi
testnet this summer" per the 2025-06-24 PR; Optimum published Hoodi results on
2025-09-23 and 2025-10-16) — with **Sepolia as a never-treated control** (same
clients, same forks, no mump2p). Daily propagation physics, 2025-06-01 →
2026-01-31, 487 network-days (`event_study.py` / `analyze_event_study.py`).

**1. No positive mump2p signature exists.** A break-scan on Hoodi's transit
finds a −56% compression on 2025-09-29 — which would look like a triumphant
treatment effect, except the *control* network breaks three days later, harder
(−72%) and cleaner (r²=0.84 vs 0.34). The October compression is a common shock
(the Fusaka client-release/fork window, plus heavy Xatu sentry churn on both
networks). Every clean break in the data belongs to the network *without*
mump2p.

**2. During the deployment window, the treated network got WORSE.** Hoodi's
median transit tripled (Jun/Jul ~600-700ms → Aug/Sep ~2,000ms) while Sepolia
stayed flat — a ~1,600ms adverse relative swing — and sentry composition was
stable through exactly those months, so it is not an observation artifact. We
do not claim mump2p *caused* the degradation (testnets host many experiments);
we note only that the deployment window coincides with the network's worst
propagation of the study period.

**3. Current-state Hoodi (2026-06, attestation tables live) still has the
cliff.** Correct-head votes: 95.9% → 77.4% → 40.5% across the 4s deadline.
On the mump2p network, 2.1% of blocks arrive past the deadline — more than
double mainnet's 0.93% — and the attestation floor is ~20x worse than mainnet.

**4. The "6x" benchmark venue is unrepresentative in the flattering
direction.** Hoodi's gossipsub transit (~1,900ms median in mid-2025) is more
than twice mainnet's (880ms). Beating a slow baseline by 6x implies far less
against mainnet's fast one.

Shadow-mode, confirmed observationally: Optimum's own methodology states blocks
propagate "through both mump2p and Gossipsub simultaneously," and eight months
of daily physics on their home network shows exactly what that predicts —
nothing. This is *no evidence of effect*, not proof of none; but for a product
whose pitch is propagation speed, leaving no visible trace on its only deployed
network is the single most load-bearing fact in this repository.

### If mump2p does ship to mainnet

The design becomes viable *only if* rollout is genuinely staggered across operators and deployment dates are observable. Should that happen, the RD harness here already computes the outcome panel; a Callaway–Sant'Anna layer on top would be a modest addition. The blocker was never the code — it was that **the treatment did not exist**.
