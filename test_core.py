"""Offline tests for pricing, chain analytics and Yahoo and Schwab providers: python test_core.py"""
import math
import sys
import types

import numpy as np
import pandas as pd

import core
from core import ProviderError, SchwabProvider

bs = market = core


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    assert ok, name


S, T, r, q, sig = 5800.0, 300 / 365, 0.04, 0.013, 0.17




# put-call parity: C - P = S e^{-qT} - K e^{-rT}
K = 5200.0
c, p = bs.price(S, K, T, r, q, sig, "call"), bs.price(S, K, T, r, q, sig, "put")
check("put-call parity", abs((c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-8)

# implied vol round trip
for kind in ("call", "put"):
    px = bs.price(S, K, T, r, q, 0.21, kind)
    check(f"iv round trip ({kind})", abs(bs.implied_vol(px, S, K, T, r, q, kind) - 0.21) < 1e-6)

# strike-for-delta round trip
for kind, d in (("call", 0.75), ("put", 0.75), ("call", 0.30)):
    k = bs.strike_for_delta(S, T, r, q, sig, d, kind)
    got = abs(bs.greeks(S, k, T, r, q, sig, kind).delta)
    check(f"strike_for_delta {kind} {d}", abs(got - d) < 1e-9)

# greeks vs finite differences
g = bs.greeks(S, K, T, r, q, sig, "call")
h = 0.01
fd_delta = (bs.price(S + h, K, T, r, q, sig) - bs.price(S - h, K, T, r, q, sig)) / (2 * h)
fd_gamma = (bs.price(S + 1, K, T, r, q, sig) - 2 * bs.price(S, K, T, r, q, sig) + bs.price(S - 1, K, T, r, q, sig)) / 1
fd_vega = (bs.price(S, K, T, r, q, sig + 0.0001) - bs.price(S, K, T, r, q, sig - 0.0001)) / 0.0002 / 100
fd_theta = (bs.price(S, K, T - 1 / 365, r, q, sig) - bs.price(S, K, T, r, q, sig))
fd_rho = (bs.price(S, K, T, r + 0.0001, q, sig) - bs.price(S, K, T, r - 0.0001, q, sig)) / 0.0002 / 100
check("delta", abs(g.delta - fd_delta) < 1e-6)
check("gamma", abs(g.gamma - fd_gamma) < 1e-6)
check("vega", abs(g.vega - fd_vega) < 1e-5)
check("theta (1 day)", abs(g.theta - fd_theta) < 5e-3)
check("rho", abs(g.rho - fd_rho) < 1e-5)

# synthetic chain: carry inference + 75-delta pick
strikes = np.arange(4000, 7001, 50.0)
rows = []
for k in strikes:
    for kind in ("call", "put"):
        px = bs.price(S, k, T, r, q, 0.17 + 0.00004 * (S - k), kind)
        rows.append(dict(kind=kind, strike=k, bid=px - 0.5, ask=px + 0.5, lastPrice=px,
                         mid=px, quote="mid", volume=10, openInterest=100))
chain = pd.DataFrame(rows)
q_hat = market.implied_carry(chain, S, T, r)
check(f"implied carry ~ {q:.3%} (got {q_hat:.3%})", abs(q_hat - q) < 1e-4)

df = market.enrich(chain, S, T, r, q_hat)
pick = market.pick_contract(df, "call", 0.75)
check(f"75-delta call strike {pick.strike:.0f}, delta {pick.delta:.3f}", abs(pick.delta - 0.75) < 0.02)



# ------------------------------------------------------------------ fake HTTP layer
class FakeResp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self, get_script, token_status=200):
        self.posts, self.gets, self._script, self._token_status = 0, [], list(get_script), token_status

    def post(self, url, data=None, auth=None, timeout=None):
        self.posts += 1
        assert data["grant_type"] == "refresh_token" and auth == ("key", "secret")
        if self._token_status != 200:
            return FakeResp(self._token_status, text="invalid_grant")
        return FakeResp(200, {"access_token": f"tok{self.posts}", "expires_in": 1800})

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, params, headers["Authorization"]))
        return self._script.pop(0)


def contract(strike, bid, ask, vol="NaN", delta=-999.0, oi=10, root="SPX"):
    return dict(symbol=f"{root} {strike}", strikePrice=strike, bid=bid, ask=ask, last=0.0,
                totalVolume=1, openInterest=oi, volatility=vol, delta=delta)


CHAIN = {
    "callExpDateMap": {"2027-07-30:297": {
        "5350.0": [contract(5350.0, 700.0, 702.0, vol=17.5, delta=0.75, oi=50),
                   contract(5350.0, 701.0, 703.0, oi=5, root="SPXW")],      # duplicate root, less OI
        "9000.0": [contract(9000.0, 0.0, 0.0)]}},
    "putExpDateMap": {"2027-07-30:297": {"5350.0": [contract(5350.0, 90.0, 92.0, delta=-0.25)]}},
}

# --------------------------------------------------------------------------- Schwab
df = SchwabProvider.parse_chain(CHAIN)
call = df[(df.kind == "call") & (df.strike == 5350)].iloc[0]
check("parse: duplicate roots collapse to the higher open interest", len(df[(df.kind == "call") & (df.strike == 5350)]) == 1 and call.openInterest == 50)
check("parse: mid from two-sided quote", call.mid == 701.0 and call.quote == "mid")
check("parse: vendor IV scaled to a fraction", abs(call.vendor_iv - 0.175) < 1e-12)
check("parse: -999 / 'NaN' sentinels become NaN", math.isnan(df[(df.kind == "put")].iloc[0].vendor_iv) and math.isnan(df[df.strike == 9000].iloc[0].vendor_delta))
check("parse: no quote falls back to last", df[df.strike == 9000].iloc[0].quote == "last")

try:
    SchwabProvider.parse_chain({"callExpDateMap": {}, "putExpDateMap": {}})
    check("parse: empty chain raises", False)
except ProviderError:
    check("parse: empty chain raises", True)

http = FakeHttp([
    FakeResp(200, {"$SPX": {"quote": {"lastPrice": 5812.5}}}),
    FakeResp(200, {"expirationList": [{"expirationDate": "2001-01-19"}, {"expirationDate": "2999-01-15"}, {"expirationDate": "2999-01-15"}]}),
])
sp = SchwabProvider("key", "secret", "refresh", http=http)
check("spot parsed", sp.spot() == 5812.5)
check("expiries: past dropped, duplicates collapsed", sp.expiries() == ["2999-01-15"])
check("access token reused (one token POST for two calls)", http.posts == 1)
check("bearer header sent", http.gets[0][2] == "Bearer tok1")

http = FakeHttp([FakeResp(401, text="expired"), FakeResp(200, {"$SPX": {"quote": {"lastPrice": 1.0}}})])
sp = SchwabProvider("key", "secret", "refresh", http=http)
sp.spot()
check("401 triggers one forced token refresh and a retry", http.posts == 2 and http.gets[1][2] == "Bearer tok2")

sp = SchwabProvider("key", "secret", "refresh", http=FakeHttp([], token_status=400))
try:
    sp.spot()
    check("expired refresh token explains how to fix it", False)
except ProviderError as exc:
    check("expired refresh token explains how to fix it", "schwab-login" in str(exc) and "HTTP 400" in str(exc))

# ------------------------------------------------- end to end through the analytics
S, T, r, q = 5800.0, 297 / 365, 0.04, 0.013
rows = []
for k in range(4000, 7001, 50):
    for kind in ("call", "put"):
        px = bs.price(S, k, T, r, q, 0.17, kind)
        rows.append(dict(kind=kind, strike=float(k), bid=px - 0.4, ask=px + 0.4, openInterest=100))
full = market.finalize_chain(pd.DataFrame(rows))
pick = market.pick_contract(market.enrich(full, S, T, r, q), "call", 0.75)
check(f"vendor-style chain -> 75-delta pick (strike {pick.strike:.0f}, delta {pick.delta:.3f})", abs(pick.delta - 0.75) < 0.02)


# ------------------------------------------------------------------------ Yahoo
def _fake_yfinance(history_ok=True):
    calls = pd.DataFrame({"contractSymbol": ["C1", "C2"], "strike": [5350.0, 9000.0], "bid": [700.0, 0.0],
                          "ask": [702.0, 0.0], "lastPrice": [701.0, 0.0], "volume": [5, None],
                          "openInterest": [50, 0], "impliedVolatility": [0.17, 0.0]})
    puts = pd.DataFrame({"contractSymbol": ["P1"], "strike": [5350.0], "bid": [90.0], "ask": [92.0],
                         "lastPrice": [91.0], "volume": [3], "openInterest": [20], "impliedVolatility": [0.2]})

    class Ticker:
        def __init__(self, sym):
            self.fast_info = types.SimpleNamespace(last_price=17.4 if sym == "^VIX" else 5812.5)
            self.options = ["2001-01-19", "2999-01-15"]

        def option_chain(self, expiry):
            return types.SimpleNamespace(calls=calls, puts=puts)

        def history(self, period):
            if not history_ok:
                raise RuntimeError("offline")
            return pd.DataFrame({"Close": [4.1]})

    mod = types.ModuleType("yfinance")
    mod.Ticker = Ticker
    return mod


sys.modules["yfinance"] = _fake_yfinance()
yp = core.YahooProvider()
check("yahoo: spot", yp.spot() == 5812.5)
check("yahoo: past expiries dropped", yp.expiries() == ["2999-01-15"])
ydf = yp.chain("2999-01-15")
yc = ydf[(ydf.kind == "call") & (ydf.strike == 5350)].iloc[0]
check("yahoo: chain mapped to the shared schema", yc.mid == 701.0 and yc.quote == "mid" and math.isnan(yc.vendor_delta))
check("yahoo: one-sided strike falls back to last", ydf[ydf.strike == 9000].iloc[0].quote == "last")
check("rate: ^IRX percent -> decimal", abs(core.get_rate() - 0.041) < 1e-12)
sys.modules["yfinance"] = _fake_yfinance(history_ok=False)
check("rate: falls back to the default on failure", core.get_rate() == 0.04 and core.get_rate(0.05) == 0.05)
del sys.modules["yfinance"]


# ------------------------------------------------------------------ scenario engine
S0, r0, q0, dte0 = 5800.0, 0.04, 0.013, 300
ks = np.array([5085.0, 5385.0, 5755.0, 5990.0, 6360.0])
ivs = np.array([0.19, 0.18, 0.17, 0.165, 0.16])
mv = np.round(np.arange(-0.20, 0.2001, 0.025), 6)

# vectorised price == scalar price, calls and puts, incl. expiry intrinsic
for kind in ("call", "put"):
    vec = core.price_vec(S0, ks[:, None], 0.8, r0, q0, ivs[:, None], kind)
    sca = np.array([core.price(S0, k, 0.8, r0, q0, v, kind) for k, v in zip(ks, ivs)])[:, None]
    check(f"price_vec matches scalar price ({kind})", np.allclose(vec, sca, atol=1e-9))
check("price_vec at expiry is intrinsic", np.allclose(core.price_vec(5900.0, 5800.0, 0.0, r0, q0, 0.2), 100.0))

# value at "today, no move, unchanged IV" reproduces the model price
v0 = core.scenario_values(S0, ks, ivs, "call", dte0, 0, r0, q0, [0.0])
check("scenario value with no time, move or IV change equals today's price",
      np.allclose(v0[:, 0], [core.price(S0, k, dte0 / 365, r0, q0, v) for k, v in zip(ks, ivs)], atol=1e-9))

vals = core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, mv)
check("call value rises with the SPX level for every strike", bool(np.all(np.diff(vals, axis=1) > 0)))
vals_put = core.scenario_values(S0, ks, ivs, "put", dte0, 90, r0, q0, mv)
check("put value falls with the SPX level for every strike", bool(np.all(np.diff(vals_put, axis=1) < 0)))
check("holding to expiry gives intrinsic value",
      np.allclose(core.scenario_values(S0, ks, ivs, "call", dte0, dte0, r0, q0, mv),
                  np.maximum(S0 * (1 + mv)[None, :] - ks[:, None], 0.0)))

# IV assumptions move values the right way (long options have positive vega)
base = core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, mv)
check("a parallel IV drop lowers call values", bool(np.all(core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, mv, -3.0) < base)))
beta = core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, mv, 0.0, -0.5)
up_cols, dn_cols = mv > 0, mv < 0
check("negative IV beta: rallies worth less, sell-offs worth more than constant IV",
      bool(np.all(beta[:, up_cols] < base[:, up_cols]) and np.all(beta[:, dn_cols] > base[:, dn_cols])))
check("zero move is unaffected by the IV beta", np.allclose(beta[:, mv == 0], base[:, mv == 0]))

# probabilities
sd = 0.17 * math.sqrt(90 / 365)
wide = np.round(np.arange(-0.60, 1.2001, 0.005), 6)
for mean in (-0.05, 0.0, 0.03):
    pr = core.move_probabilities(wide, mean, sd)
    check(f"probabilities sum to 1 and mean move ~ {mean:+.0%} (got {float(pr @ wide):+.4f})",
          abs(pr.sum() - 1) < 1e-12 and abs(float(pr @ wide) - mean) < 2e-3)
pr = core.move_probabilities(mv, 0.0, sd)
check("probabilities are non-negative and sum to 1 on the working grid", bool(np.all(pr >= 0)) and abs(pr.sum() - 1) < 1e-12)
check("a more bullish view puts more mass on rises",
      core.move_probabilities(mv, 0.05, sd)[mv > 0].sum() > core.move_probabilities(mv, -0.05, sd)[mv > 0].sum())
dn, fl, rs = core.regime_masks(mv, 0.02)
check("regimes partition the grid", bool(np.all(dn.astype(int) + fl.astype(int) + rs.astype(int) == 1)) and fl.sum() == 1)

# breakevens: option is worth exactly its entry price there
entry = np.array([core.price(S0, k, dte0 / 365, r0, q0, v) for k, v in zip(ks, ivs)])
be = core.scenario_breakevens(entry, S0, ks, ivs, "call", dte0, 90, r0, q0)
at_be = np.array([core.scenario_values(S0, [k], [v], "call", dte0, 90, r0, q0, [m])[0, 0] for k, v, m in zip(ks, ivs, be)])
check("breakeven move reprices to the entry price", np.allclose(at_be, entry, atol=1e-6))
check("breakevens are larger the further out of the money", bool(np.all(np.diff(be) > 0)))

# stats table
deltas = np.array([core.greeks(S0, k, dte0 / 365, r0, q0, v).delta for k, v in zip(ks, ivs)])
table, regimes = core.scenario_stats(base, entry, pr, mv, ks, deltas, be, 0.02)
check("stats: one row per strike with the expected columns",
      len(table) == len(ks) and {"exp_ret", "p_profit", "exp_pnl", "r_decline", "r_flat", "r_rise", "breakeven"} <= set(table.columns))
check("stats: regime probabilities sum to 1", abs(sum(regimes.values()) - 1) < 1e-12)
tight = core.move_probabilities(mv, 0.0, 1e-4)
tt, _ = core.scenario_stats(base, entry, tight, mv, ks, deltas, be, 0.02)
check("stats: with all mass on a flat market, expected return is the flat-market return",
      np.allclose(tt.exp_ret, base[:, mv == 0][:, 0] / entry - 1, atol=1e-9))
check("stats: deeper-ITM calls lose less in a flat market", bool(np.all(np.diff(tt.exp_ret.to_numpy()) < 0)))
check("stats: probability of profit is within [0, 1] and falls for further-OTM strikes",
      bool(np.all((table.p_profit >= 0) & (table.p_profit <= 1))) and bool(np.all(np.diff(table.p_profit.to_numpy()) <= 1e-12)))
check("stats: best case beats worst case; conditional returns rise from decline to rise",
      bool(np.all(table.best > table.worst)) and bool(np.all(table.r_decline < table.r_flat)) and bool(np.all(table.r_flat < table.r_rise)))


# fine grid: dense, honest integration independent of the display grid
fm, fp = core.fine_grid(0.03, sd)
check("fine grid: probabilities sum to 1 and mean move matches", abs(fp.sum() - 1) < 1e-12 and abs(float(fp @ fm) - 0.03) < 5e-4)
fvals = core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, fm)
ftab, freg = core.scenario_stats(fvals, entry, fp, fm, ks, deltas, be, 0.02, display_values=base)
check("fine grid: probability of profit now differs across strikes (no grid-bucket ties)",
      len(set(np.round(ftab.p_profit, 4))) == len(ks))
cdf_check = 1 - core.norm.cdf((math.log1p(be[1]) - (math.log1p(0.03) - 0.5 * sd**2)) / sd)
check("fine grid: probability of profit matches the closed form P(move > breakeven)", abs(ftab.p_profit[1] - cdf_check) < 3e-3)
check("fine grid: worst/best come from the displayed grid",
      np.allclose(ftab.worst, (base / entry[:, None] - 1).min(axis=1)) and np.allclose(ftab.best, (base / entry[:, None] - 1).max(axis=1)))
wide_view = core.fine_grid(0.0, 0.35)
wv = core.scenario_values(S0, ks, ivs, "call", dte0, 90, r0, q0, wide_view[0])
wt, _ = core.scenario_stats(wv, entry, wide_view[1], wide_view[0], ks, deltas, be, 0.02)
gt, _ = core.scenario_stats(base, entry, core.move_probabilities(mv, 0.0, 0.35), mv, ks, deltas, be, 0.02)
check("wide view: clamping to a +/-20% grid understates the upside the fine grid captures",
      bool(np.all(wt.exp_ret > gt.exp_ret)))


# blank / padded credentials
for blank_args, missing in ((("key", "secret", ""), "refresh_token"), (("", "secret", "tok"), "app_key"),
                            (("key", "  ", "tok"), "app_secret"), (("key", "secret", None), "refresh_token")):
    try:
        SchwabProvider(*blank_args, http=FakeHttp([])).spot()
        check(f"blank credential is reported by name ({missing})", False)
    except ProviderError as exc:
        check(f"blank credential is reported by name ({missing})", missing in str(exc) and "empty" in str(exc))
padded_http = FakeHttp([FakeResp(200, {"$SPX": {"quote": {"lastPrice": 1.0}}})])
SchwabProvider(' "key" ', "secret\n", " 'refresh' ", http=padded_http).spot()
check("quotes and whitespace around credentials are stripped", padded_http.posts == 1)


code_like = "C0.b2F1dGgyLmJkYy5zY2h3YWIuY29t.-6PHEur2o_Zev2oCbAyn0iAEfJqB7jD_vRxK93xAdHg%40"
try:
    SchwabProvider("key", "secret", code_like, http=FakeHttp([])).spot()
    check("authorization code pasted as refresh token is recognised", False)
except ProviderError as exc:
    check("authorization code pasted as refresh token is recognised", "authorization code" in str(exc) and "schwab-login" in str(exc))
ok_http = FakeHttp([FakeResp(200, {"$SPX": {"quote": {"lastPrice": 1.0}}})])
SchwabProvider("key", "secret", "realRefreshTokenValue", http=ok_http).spot()
check("a normal refresh token is not mistaken for a code", ok_http.posts == 1)


# ------------------------------------------------- first strike above the target delta
def _chain(deltas, quotes=None, kind="call"):
    quotes = quotes or ["mid"] * len(deltas)
    return pd.DataFrame({"kind": kind, "strike": [5000.0 + 50 * i for i in range(len(deltas))],
                         "delta": deltas, "quote": quotes})


cd = _chain([0.60, 0.70, 0.74, 0.752, 0.78, 0.85])
check("pick: first delta strictly above 0.75 (0.752, not the nearer-from-below 0.74)",
      core.pick_contract(cd, "call", 0.75).delta == 0.752)
check("pick: a strike at exactly the target is skipped",
      core.pick_contract(_chain([0.70, 0.75, 0.76, 0.80]), "call", 0.75).delta == 0.76)
check("pick: works when listed in any order",
      core.pick_contract(cd.sample(frac=1, random_state=3), "call", 0.75).delta == 0.752)
check("pick: puts are compared by absolute delta",
      core.pick_contract(_chain([-0.55, -0.70, -0.76, -0.82], kind="put"), "put", 0.75).delta == -0.76)
check("pick: two-sided quotes are preferred over a closer one-sided strike",
      core.pick_contract(_chain([0.751, 0.76, 0.80], ["last", "mid", "mid"]), "call", 0.75).delta == 0.76)
check("pick: falls back to the one-sided strike when nothing else exists",
      core.pick_contract(_chain([0.751, 0.80], ["last", "last"]), "call", 0.75).delta == 0.751)
check("pick: if nothing exceeds the target, return the highest delta",
      core.pick_contract(_chain([0.40, 0.55, 0.62]), "call", 0.75).delta == 0.62)
check("pick: no usable rows gives None",
      core.pick_contract(_chain([math.nan, math.nan]), "call", 0.75) is None
      and core.pick_contract(_chain([0.8]), "put", 0.75) is None)
check("pick: on the enriched synthetic chain the delta is above 0.75 and within one strike of it",
      0.75 < pick.delta < 0.77)

# ------------------------------------------------- VIX lookups
sys.modules["yfinance"] = _fake_yfinance()
check("yahoo: vix reads ^VIX, not the SPX quote", core.YahooProvider().vix() == 17.4 and core.YahooProvider().spot() == 5812.5)
del sys.modules["yfinance"]

def vixp(script, **kw):
    http = FakeHttp(script)
    return SchwabProvider("key", "secret", "refresh", http=http, **kw), http

prov, http = vixp([FakeResp(200, {"$VIX": {"quote": {"lastPrice": 17.42, "closePrice": 16.0}}})])
check("schwab: vix asks for $VIX first and uses the last price", prov.vix() == 17.42 and http.gets[0][1]["symbols"] == "$VIX")
prov, http = vixp([FakeResp(200, {"$VIX": {"quote": {"mark": 18.1}}})])
check("schwab: vix falls back to the mark when there is no last price", prov.vix() == 18.1)
prov, http = vixp([FakeResp(200, {"errors": {"invalidSymbols": ["$VIX"]}}), FakeResp(200, {"VIX": {"quote": {"lastPrice": 16.2}}}),
                   FakeResp(200, {"VIX": {"quote": {"lastPrice": 16.3}}})])
check("schwab: vix falls back from $VIX to VIX", prov.vix() == 16.2 and [g[1]["symbols"] for g in http.gets] == ["$VIX", "VIX"])
check("schwab: the symbol that worked is tried first next time", prov.vix() == 16.3 and http.gets[2][1]["symbols"] == "VIX")
prov, http = vixp([FakeResp(200, {"AAPL": {"quote": {"lastPrice": 200.0}}, "MSFT": {"quote": {"lastPrice": 400.0}}})] * 2)
try:
    prov.vix()
    check("schwab: vix does not guess among several unrelated quotes", False)
except ProviderError as exc:
    check("schwab: vix does not guess among several unrelated quotes", "$VIX" in str(exc) and "VIX" in str(exc))
prov, http = vixp([FakeResp(200, {"/VX": {"quote": {"lastPrice": 15.5}}})], vix_symbol=" /VX ")
check("schwab: vix symbol can be overridden (and then is the only one tried)",
      prov.vix() == 15.5 and http.gets[0][1]["symbols"] == "/VX" and len(http.gets) == 1)
check("schwab: a blank override uses $VIX then VIX",
      SchwabProvider("k", "s", "r", vix_symbol="  ").vix_symbols == ("$VIX", "VIX")
      and SchwabProvider("k", "s", "r", vix_symbol=None).vix_symbol == "$VIX")
for body, why in (({}, "symbol missing from the response"), ({"$VIX": {"quote": {"closePrice": 16.0}}}, "only a stale close"),
                  ({"errors": {"invalidSymbols": ["$VIX"]}}, "Schwab reports an invalid symbol")):
    prov, _ = vixp([FakeResp(200, body)] * 2)
    try:
        prov.vix()
        check(f"schwab: vix raises instead of guessing ({why})", False)
    except ProviderError as exc:
        check(f"schwab: vix raises instead of guessing ({why})", "VIX" in str(exc))

print("\nAll checks passed.")