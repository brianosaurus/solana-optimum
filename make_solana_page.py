#!/usr/bin/env python3
"""
Render the public Solana revenue page from the latest study outputs.

Reads data/revenue_graph.json (operators + slopes, written by run_solana.py),
computes network-wide daily revenue from each collector's OWN coverage span
(so a just-started stream isn't diluted over the whole panel), and fills the
data-driven template into:

  * a standalone HTML doc  -> STANDALONE_OUT (default ~/watcher/static/solana.html)
  * the artifact fragment  -> FRAGMENT_OUT  (default data/revenue_fragment.html)

Env: SENTRY_DATA_DIR, SOLANA_RPC_URL, STANDALONE_OUT, FRAGMENT_OUT, INFLATION_RATE.
Reusable from the daily cron so brian.biz/optimum/solana tracks the live study.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import urllib.request
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def _open(f):
    return gzip.open(f, "rt") if f.endswith(".gz") else open(f)


def rate_sol_per_day(files, key: str) -> float:
    """SOL/day for a stream, over ITS OWN recv-time span (not the panel span)."""
    lo = hi = None
    lam = 0
    for f in files:
        with _open(f) as fh:
            for line in fh:
                r = json.loads(line)
                t = r.get("recv_ms")
                if t is None:
                    continue
                lo = t if lo is None else min(lo, t)
                hi = t if hi is None else max(hi, t)
                lam += r.get(key, 0)
    h = (hi - lo) / 3.6e6 if lo else 0
    return (lam / 1e9 / h * 24) if h else 0.0


def total_active_stake_sol(rpc: str) -> float:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getVoteAccounts",
                       "params": [{"keepUnstakedDelinquents": False}]}).encode()
    req = urllib.request.Request(rpc, data=body, headers={"content-type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
    return sum(v["activatedStake"] for v in res.get("current", [])) / 1e9


def sol_price_usd(fallback: float) -> float:
    """Live SOL/USD. Env SOL_PRICE_USD wins (so a run can pin it); else CoinGecko;
    else the fallback carried in the graph. Only affects the USD display — every
    SOL-denominated figure is price-independent."""
    env = os.getenv("SOL_PRICE_USD")
    if env:
        return float(env)
    try:
        req = urllib.request.Request(
            "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
            headers={"user-agent": "solana-optimum"})
        return float(json.loads(urllib.request.urlopen(req, timeout=15).read())["solana"]["usd"])
    except Exception:
        return float(fallback)


def _usd(v: float) -> str:
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${round(v/1e3):,.0f}k"
    return f"${v:,.0f}"


KAPPA_DEFAULT = 0.25  # must match the template's page-level KAPPA


def prerender(html: str, d: dict) -> str:
    """Bake the headline numbers into the markup.

    The page is generated, so there is no excuse for a fetcher, a link preview, a
    reader mode or a PDF seeing a wall of em-dashes under the words "live
    measurement". JS still re-renders these on load (identically) and drives the
    interactive parts; this is the server-side default, not a duplicate source of
    truth — both read the same DATA block.
    """
    n, price = d["network"], d["price"]
    st_n = (d.get("stats") or {}).get("n_slots")
    st_l = (d.get("stats") or {}).get("n_leaders")
    hrs = d["spanHours"]
    span = f"{hrs/24:.1f} days" if hrs >= 24 else f"~{round(hrs)} h"
    mov = n["fees"] + n["tips"]
    tot = n["inflation"] + mov
    ops = d["operators"]
    sub = {
        '<b id="span">—</b>': f'<b id="span">{span}</b>',
        '<b id="spd">6×</b>': f'<b id="spd">{d["speedup"]:g}×</b>',
        '<b id="price">@ $150</b>': f'<b id="price">@ ${price:g}</b>',
        '<b id="kaHdr">κ=25%</b>': f'<b id="kaHdr">κ={KAPPA_DEFAULT:.0%}</b>',
        '<p id="mSample">Sample: — of continuous capture.</p>':
            f'<p id="mSample">Sample: <b>{span}</b> of continuous capture'
            + (f' ({st_n:,} leader slots · {st_l} leaders).' if st_n else '.') + '</p>',
        '<span id="hMov">—</span>': f'<span id="hMov">{mov:,.0f}</span>',
        '<p class="v" id="hRatio">—</p>':
            f'<p class="v" id="hRatio">{n["fees"]/n["tips"]:.1f}×</p>' if n["tips"] else "",
        '<span id="hTop">—</span>': "",
        '<span class="tot" id="totFull">—</span>':
            f'<span class="tot" id="totFull">{tot:,.0f} SOL / day</span>',
        '<span class="tot" id="totMov">—</span>':
            f'<span class="tot" id="totMov">{mov:,.0f} SOL / day</span>',
    }
    # the pre-registered rule, server-side too: a channel whose slope is not
    # distinct from zero contributes nothing. Must match the template.
    st0 = d.get("stats") or {}
    def live(c):
        p = (st0.get(c) or {}).get("p")
        return 0.0 if (p is not None and p >= 0.05) else 1.0
    t_on, f_on = live("tips"), live("fees")
    if ops:
        top = max(ops, key=lambda o: o["up_tips_sol_yr"]*t_on + o["up_fees_sol_yr"]*f_on)
        sub['<span id="hTop">—</span>'] = (
            f'<span id="hTop">{_usd((top["up_tips_sol_yr"]*t_on+top["up_fees_sol_yr"]*f_on)*KAPPA_DEFAULT*price)}</span>')
    for k, v in sub.items():
        if v:
            html = html.replace(k, v)
    # the inference cells — the section a reviewer reads first
    st = d.get("stats") or {}
    cells = []
    for key, label in (("tips", "Jito tips · sealing slope"), ("fees", "Leader fees · sealing slope")):
        s = st.get(key)
        if not s or s.get("slope") is None:
            cells.append(f'<div class="cell"><p class="k">{label}</p><div class="val">not yet estimated</div></div>')
            continue
        sig = s.get("p") is not None and s["p"] < 0.05
        col = "var(--fees)" if sig else "var(--muted)"
        verdict = "✓ significant" if sig else "— not distinct from 0"
        p = "n/a" if s.get("p") is None else f"{s['p']:.3f}"
        cells.append(
            f'<div class="cell"><p class="k">{label}</p><div class="val">'
            f'{s["slope"]:+.5f} SOL/100ms<br>t {s["t"]:+.2f} · wild-p {p} '
            f'<span class="verdict" style="color:{col}">{verdict}</span></div></div>')
    cells.append('<div class="cell"><p class="k">Inference</p><div class="val">within-leader-window FE<br>'
                 f'cluster-robust + {st.get("bootstrap_reps", 0)}× wild bootstrap</div></div>')
    html = html.replace('<div class="infer" id="infer"></div>',
                        '<div class="infer" id="infer">' + "".join(cells) + "</div>")

    # two-way clustering result, rendered from the run rather than hardcoded —
    # a stale t-stat in an objection is worse than no objection.
    tw = (st.get("two_way") or {})
    twf, twt = tw.get("fees"), tw.get("tips")
    if twf:
        holds = twf["p"] < 0.05
        bits = (f'The fee slope {"holds" if holds else "<b>does not hold</b>"} at '
                f'<b>t {twf["t"]:+.2f}, p {twf["p"]:.4f}</b>, with standard errors inflating '
                f'{twf["se_inflation"]:.2f}×')
        if twt:
            bits += f'; tips inflates {twt["se_inflation"]:.2f}×'
        bits += (". So the headline "
                 + ("survives" if holds else "does NOT survive")
                 + " the correction my own catalogue demanded. "
                 f'<b>The caveat I will not bury:</b> this panel yields only <b>{twf["n_hours"]} '
                 f'hour-clusters</b>, which is few for asymptotics in that dimension, so it is '
                 "reassurance rather than proof — a multi-day panel is the real test.")
        html = html.replace(
            '<span id="twoWay">Result pending the next run.</span>',
            f'<span id="twoWay">{bits}</span>')
    return html


def main() -> None:
    dd = os.getenv("SENTRY_DATA_DIR", "data")
    rpc = os.getenv("SOLANA_RPC_URL", "")
    infl_rate = float(os.getenv("INFLATION_RATE", "0.045"))
    g = json.load(open(os.path.join(dd, "revenue_graph.json")))

    tips = rate_sol_per_day(sorted(glob.glob(f"{dd}/jitotips_titan_*.jsonl*")), "tip_lamports")
    fees = rate_sol_per_day(sorted(glob.glob(f"{dd}/fees_titan_*.jsonl*")), "fee_lamports")
    inflation = (total_active_stake_sol(rpc) * infl_rate / 365) if rpc else 0.0

    ops = [{"group": o["group"], "n_val": o["n_val"],
            "up_tips_sol_yr": round(o["up_tips_sol_yr"], 2),
            "up_fees_sol_yr": round(o["up_fees_sol_yr"], 2)} for o in g["operators"]]
    data = {
        "network": {"inflation": round(inflation), "fees": round(fees), "tips": round(tips)},
        "operators": ops,
        "price": round(sol_price_usd(g["sol_price_usd"]), 2), "speedup": g["speedup"],
        "spanHours": round(g["span_h"], 1),
        "tipSlope": g["tip_slope_100ms"], "feeSlope": g["fee_slope_100ms"],
        "stats": g.get("stats", {}),
        # inputs for the adoption-saturation sweep
        "medianTransitMs": round(median([o["transit_ms"] for o in g["operators"]]), 1)
                           if g["operators"] else 350.0,
        "slotsPerYear": round(g["stats"]["n_slots"] / g["span_h"] * 24 * 365)
                        if g.get("stats", {}).get("n_slots") else 0,
    }

    tpl = (HERE / "deploy" / "revenue_template.html").read_text()
    frag = re.sub(r"/\*__DATA__\*/.*?/\*__END__\*/",
                  lambda _m: "/*__DATA__*/" + json.dumps(data) + "/*__END__*/",
                  tpl, flags=re.S)
    frag = prerender(frag, data)

    head, _, body = frag.partition("</style>")
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
           + head + "</style>\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")

    standalone = os.getenv("STANDALONE_OUT", str(Path.home() / "watcher" / "static" / "solana.html"))
    fragment = os.getenv("FRAGMENT_OUT", os.path.join(dd, "revenue_fragment.html"))
    Path(standalone).write_text(doc)
    Path(fragment).write_text(frag)
    top = max((o["up_tips_sol_yr"] + o["up_fees_sol_yr"]) * data["price"] for o in ops)
    print(f"network SOL/day: {data['network']} | top operator 6x uplift ${top:,.0f}/yr")
    print(f"wrote {standalone}\nwrote {fragment}")


if __name__ == "__main__":
    main()
