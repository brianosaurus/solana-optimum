"""
Validator grouping: cluster Solana validators into OPERATORS so the study can be
run per-group (the analog of the Ethereum study's named-operator panel).

Why group at all
----------------
A single operator runs many validator nodes sharing infrastructure — the same
relay/Jito setup, the same data centers, the same timing strategy. Those are
exactly the things that move propagation and MEV-tip capture. Comparing operators
is the natural way to ask "who is good at the delay game", and grouping is what
lets us aggregate our per-leader panel (where `leader` is a node identity) up to
the entity that actually makes the decisions.

The grouping key (on-chain, no third-party service)
---------------------------------------------------
PRIMARY: the vote account's AUTHORIZED WITHDRAWER. An operator's nodes typically
share one withdraw-authority key (the treasury that rewards flow to), so it is a
strong "same operator" signal readable straight from the vote account
(bytes [36:68] of the account data). Caveats, stated plainly:
  * Some large operators use a distinct cold withdrawer per validator (best
    practice) — those get UNDER-grouped (split into singletons). So this is a
    lower bound on concentration, never an over-count.
  * A custodian/stake-pool could in principle share a withdrawer across unrelated
    operators — rare, but why we also carry the validator-info NAME.

NAMING: the Config program (Config1111…) stores validator info (name, website,
keybase). We label each group by the modal name of its members. Groups whose
members all share a name/website but not a withdrawer are flagged so you can
merge them by hand if you want a coarser operator definition.

Output is an identity→group map (for joining to the slot panel) plus a ranked
table of groups by validator count and total stake.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field

CONFIG_PROGRAM_ID = "Config1111111111111111111111111111111111111"
# vote-state layout: version(4) + node_pubkey(32) + authorized_withdrawer(32)…
WITHDRAWER_OFFSET = 36
WITHDRAWER_LEN = 32
LAMPORTS_PER_SOL = 1_000_000_000


def _rpc(url: str, method: str, params: list):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise RuntimeError(f"{method}: {out['error']}")
    return out.get("result")


@dataclass
class Validator:
    identity: str          # node pubkey — matches `leader` in the slot panel
    vote: str
    stake_sol: float
    credits: int
    delinquent: bool
    withdrawer: str | None = None
    name: str | None = None
    website: str | None = None

    @property
    def domain(self) -> str | None:
        """Normalised website domain — an operator signal orthogonal to the
        withdrawer, so two nodes of one operator that use different treasury keys
        still cluster if they advertise the same site."""
        if not self.website:
            return None
        d = self.website.lower().split("//")[-1].split("/")[0]
        d = d[4:] if d.startswith("www.") else d
        # ignore generic hosts that would wrongly merge unrelated operators
        if d in {"", "solana.com", "github.com", "twitter.com", "x.com", "t.me"}:
            return None
        return d or None


@dataclass
class Group:
    key: str               # grouping key (withdrawer hex, or "solo:<identity>")
    identities: list[str] = field(default_factory=list)
    stake_sol: float = 0.0
    name: str | None = None


def get_validators(rpc: str) -> list[Validator]:
    res = _rpc(rpc, "getVoteAccounts", [{"keepUnstakedDelinquents": True}])
    vals: list[Validator] = []
    for status, delinq in (("current", False), ("delinquent", True)):
        for v in res.get(status, []):
            vals.append(Validator(
                identity=v["nodePubkey"],
                vote=v["votePubkey"],
                stake_sol=v["activatedStake"] / LAMPORTS_PER_SOL,
                credits=(v["epochCredits"][-1][1] if v.get("epochCredits") else 0),
                delinquent=delinq,
            ))
    return vals


def attach_withdrawers(rpc: str, vals: list[Validator], batch: int = 100) -> None:
    """Fill each validator's withdrawer via getMultipleAccounts (bytes 36:68)."""
    by_vote = {v.vote: v for v in vals}
    votes = list(by_vote)
    for i in range(0, len(votes), batch):
        chunk = votes[i:i + batch]
        res = _rpc(rpc, "getMultipleAccounts", [chunk, {
            "encoding": "base64",
            "dataSlice": {"offset": WITHDRAWER_OFFSET, "length": WITHDRAWER_LEN},
        }])
        for vote, acc in zip(chunk, res.get("value", [])):
            if not acc:
                continue
            raw = base64.b64decode(acc["data"][0])
            if len(raw) == WITHDRAWER_LEN:
                by_vote[vote].withdrawer = raw.hex()


def attach_names(rpc: str, vals: list[Validator]) -> None:
    """Label validators from the Config program's validator-info accounts."""
    by_identity = {v.identity: v for v in vals}
    try:
        accs = _rpc(rpc, "getProgramAccounts", [CONFIG_PROGRAM_ID, {"encoding": "jsonParsed"}])
    except Exception:
        return
    for a in accs or []:
        try:
            parsed = a["account"]["data"]["parsed"]["info"]
            keys = parsed.get("keys", [])
            ident = next((k["pubkey"] for k in keys if k.get("signer")), None)
            cfg = parsed.get("configData") or {}
            if ident in by_identity:
                if cfg.get("name"):
                    by_identity[ident].name = cfg["name"]
                if cfg.get("website"):
                    by_identity[ident].website = cfg["website"]
        except (KeyError, TypeError):
            continue


def build_groups(vals: list[Validator]) -> list[Group]:
    """Cluster validators into operators by UNION over shared signals.

    Two validators are the same operator if they share a withdrawer OR a website
    domain. Union-find over both signals merges an operator that uses distinct
    treasury keys per node (caught by the shared site) with one that shares a key
    but publishes no site — neither signal alone suffices, together they do.
    """
    parent: dict[str, str] = {v.identity: v.identity for v in vals}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    # bucket identities by each shared signal, then union within each bucket
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for v in vals:
        if v.withdrawer:
            buckets[("w", v.withdrawer)].append(v.identity)
        if v.domain:
            buckets[("d", v.domain)].append(v.identity)
    for members in buckets.values():
        for other in members[1:]:
            union(members[0], other)

    by_ident = {v.identity: v for v in vals}
    comps: dict[str, Group] = {}
    names: dict[str, Counter] = defaultdict(Counter)
    for v in vals:
        root = find(v.identity)
        g = comps.setdefault(root, Group(key=root))
        g.identities.append(v.identity)
        g.stake_sol += v.stake_sol
        label = v.name or (v.domain if v.domain else None)
        if label:
            names[root][label] += 1
    for root, g in comps.items():
        g.name = names[root].most_common(1)[0][0] if names[root] else None
    return sorted(comps.values(), key=lambda g: (-g.stake_sol, -len(g.identities)))


def identity_to_group(groups: list[Group]) -> dict[str, str]:
    """identity -> group label, for joining to the slot panel's `leader`."""
    m = {}
    for g in groups:
        label = g.name or g.key[:12]
        for ident in g.identities:
            m[ident] = label
    return m
