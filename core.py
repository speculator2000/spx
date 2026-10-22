"""SPX option desk core: Black-Scholes pricing, chain analytics and market-data providers.

No Streamlit in here, so it can be tested and reused on its own.

Sections
--------
1. Pricing       : Black-Scholes-Merton price, Greeks, implied vol, delta-to-strike solver
2. Chain data    : normalising a chain, dividend inference, IV/Greeks per strike, contract pick
3. Providers     : Yahoo Finance (default, delayed) and Schwab (live, REST API)
4. Scenarios     : reprice strikes at a future date across SPX moves; probabilities and ranking stats

Command line (run from this folder)
-----------------------------------
    python core.py schwab-login                       mint a Schwab refresh token (expires in 7 days)
    python core.py check-schwab [dte] [delta] [rate%] smoke-test your credentials and data

Pricing units
-------------
T      : years (calendar days / 365)
r, q   : continuous rates as decimals (0.04 = 4%)
sigma  : annualised vol as a decimal (0.18 = 18%)
vega   : price change per 1 vol point; theta: per calendar day; rho: per 1 rate point
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

CALL, PUT = "call", "put"



# ============================================================================
# 1. PRICING
# ============================================================================

@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, q, sigma, kind=CALL) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == CALL else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if kind == CALL:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def greeks(S, K, T, r, q, sigma, kind=CALL) -> Greeks:
    if T <= 0 or sigma <= 0:
        intrinsic = price(S, K, T, r, q, sigma, kind)
        itm = intrinsic > 0
        delta = (1.0 if kind == CALL else -1.0) if itm else 0.0
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t / 100.0
    decay = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t)

    if kind == CALL:
        delta = df_q * norm.cdf(d1)
        theta = decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2) / 100.0
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = decay + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2) / 100.0

    return Greeks(price(S, K, T, r, q, sigma, kind), delta, gamma, vega, theta / 365.0, rho)


def implied_vol(px, S, K, T, r, q, kind=CALL) -> float:
    """Implied vol from an option price, or NaN if the price breaks no-arbitrage bounds."""
    if px is None or T <= 0 or not math.isfinite(px) or px <= 0:
        return math.nan
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if kind == CALL:
        lower, upper = max(S * df_q - K * df_r, 0.0), S * df_q
    else:
        lower, upper = max(K * df_r - S * df_q, 0.0), K * df_r
    if px <= lower + 1e-8 or px >= upper:
        return math.nan
    lo, hi = 1e-4, 5.0
    f = lambda s: price(S, K, T, r, q, s, kind) - px
    if f(lo) > 0 or f(hi) < 0:
        return math.nan
    return brentq(f, lo, hi, xtol=1e-8, maxiter=200)


def strike_for_delta(S, T, r, q, sigma, target_delta, kind=CALL) -> float:
    """Exact strike giving `target_delta` (absolute value, e.g. 0.75) at a given vol."""
    df_q_inv = math.exp(q * T)
    p = target_delta * df_q_inv
    if not 0.0 < p < 1.0:
        raise ValueError("target delta is unreachable for these inputs")
    d1 = norm.ppf(p) if kind == CALL else -norm.ppf(p)
    return S * math.exp((r - q + 0.5 * sigma**2) * T - d1 * sigma * math.sqrt(T))

# ============================================================================
# 2. CHAIN DATA AND ANALYTICS
# ============================================================================

MULTIPLIER = 100
GREEK_COLS = ["iv", "delta", "gamma", "vega", "theta", "rho"]


# --------------------------------------------------------------------------- data
VENDOR_COLS = ["vendor_iv", "vendor_delta"]
CHAIN_COLS = ["contractSymbol", "kind", "strike", "bid", "ask", "lastPrice",
              "volume", "openInterest"] + VENDOR_COLS


def get_rate(default: float = 0.04) -> float:
    """13-week T-bill yield (^IRX, quoted in percent) as a decimal; `default` on any failure."""
    try:
        import yfinance as yf

        px = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1])
        if 0 < px < 20:
            return px / 100.0
    except Exception:
        pass
    return default


def dte(expiry: str, today: date | None = None) -> int:
    today = today or date.today()
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days


def pick_expiry(expiries: list[str], target_dte: int, today: date | None = None) -> str:
    return min(expiries, key=lambda e: abs(dte(e, today) - target_dte))


def finalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any vendor's chain to one schema and add mid / quote-type columns."""
    df = df.copy()
    for c in CHAIN_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("contractSymbol", "kind") else np.nan
    for c in ("bid", "ask", "lastPrice"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["strike", "volume", "openInterest"] + VENDOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    two_sided = (df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)
    df["mid"] = np.where(two_sided, (df.bid + df.ask) / 2.0, df.lastPrice)
    df["quote"] = np.where(two_sided, "mid", "last")
    # SPX and SPXW can list the same strike on one date: keep the busier contract.
    df = df.sort_values("openInterest", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["kind", "strike"], keep="first")
    return df[CHAIN_COLS + ["mid", "quote"]].sort_values(["kind", "strike"]).reset_index(drop=True)


# ----------------------------------------------------------------------- analytics
def implied_carry(chain: pd.DataFrame, S: float, T: float, r: float, n: int = 6) -> float | None:
    """Dividend yield implied by put-call parity near the money.

    F = K + e^{rT} (C - P);  q = r - ln(F / S) / T.  Returns None if unusable.
    """
    calls = chain[(chain.kind == "call") & (chain.quote == "mid")].set_index("strike").mid
    puts = chain[(chain.kind == "put") & (chain.quote == "mid")].set_index("strike").mid
    strikes = calls.index.intersection(puts.index)
    if len(strikes) == 0 or T <= 0:
        return None
    near = sorted(strikes, key=lambda k: abs(k - S))[:n]
    fwd = float(np.median([k + math.exp(r * T) * (calls[k] - puts[k]) for k in near]))
    if fwd <= 0:
        return None
    q = r - math.log(fwd / S) / T
    return q if -0.03 <= q <= 0.08 else None


def enrich(chain: pd.DataFrame, S: float, T: float, r: float, q: float) -> pd.DataFrame:
    rows = []
    for K, kind, px in zip(chain.strike, chain.kind, chain.mid):
        iv = implied_vol(px, S, K, T, r, q, kind)
        if math.isnan(iv):
            rows.append((math.nan,) * len(GREEK_COLS))
        else:
            g = greeks(S, K, T, r, q, iv, kind)
            rows.append((iv, g.delta, g.gamma, g.vega, g.theta, g.rho))
    out = pd.DataFrame(rows, columns=GREEK_COLS, index=chain.index)
    return chain.join(out)


def pick_contract(df: pd.DataFrame, kind: str, target_delta: float) -> pd.Series | None:
    """First strike whose |delta| is strictly greater than `target_delta`.

    "First" means the one closest to the target from above: the smallest |delta| that still
    exceeds it. A 0.75 target therefore returns, say, a 0.752 delta rather than a 0.747 one,
    and a strike at exactly 0.750 is skipped. Strikes with two-sided quotes are preferred. If
    no strike exceeds the target, the strike with the highest |delta| is returned instead.
    """
    sub = df[(df.kind == kind) & df.delta.notna()]
    liquid = sub[sub.quote == "mid"]
    sub = liquid if not liquid.empty else sub
    if sub.empty:
        return None
    absd = sub.delta.abs()
    above = sub[absd > target_delta]
    if not above.empty:
        return above.loc[above.delta.abs().idxmin()]
    return sub.loc[absd.idxmax()]

# ============================================================================
# 3. MARKET DATA PROVIDERS
# ============================================================================

class ProviderError(RuntimeError):
    """Raised with a message that is safe and useful to show in the UI."""


def _na(x) -> float:
    """Vendor sentinel values (None, 'NaN', -999) become NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if (not math.isfinite(v) or v <= -998) else v


# ---------------------------------------------------------------------------- Yahoo
class YahooProvider:
    """Free, delayed SPX data via yfinance. No credentials needed."""

    name = "Yahoo Finance (delayed)"
    SYMBOL = "^SPX"

    @staticmethod
    def _last(symbol: str) -> float:
        import yfinance as yf

        tk = yf.Ticker(symbol)
        try:
            px = float(tk.fast_info.last_price)
            if math.isfinite(px) and px > 0:
                return px
        except Exception:
            pass
        return float(tk.history(period="5d")["Close"].dropna().iloc[-1])

    def spot(self) -> float:
        return self._last(self.SYMBOL)

    def vix(self) -> float:
        """Latest VIX level. Yahoo's index quote can lag the live value."""
        return self._last("^VIX")

    def expiries(self) -> list[str]:
        import yfinance as yf

        today = date.today().isoformat()
        return sorted(e for e in yf.Ticker(self.SYMBOL).options if e >= today)

    def chain(self, expiry: str) -> pd.DataFrame:
        import yfinance as yf

        oc = yf.Ticker(self.SYMBOL).option_chain(expiry)
        df = pd.concat([oc.calls.assign(kind="call"), oc.puts.assign(kind="put")], ignore_index=True)
        return finalize_chain(df)


# ------------------------------------------------------------------------- Schwab
class SchwabProvider:
    """Schwab Market Data API over plain REST (no third-party SDK to break).

    Access tokens last ~30 minutes and are refreshed automatically. The refresh
    token expires every 7 days; mint a new one with `python core.py schwab-login`.
    """

    name = "Schwab (live)"
    API = "https://api.schwabapi.com"
    VIX_SYMBOLS = ("$VIX", "VIX")  # the VIX index; tried in this order. Override with vix_symbol in the [schwab] secrets.

    def __init__(self, app_key: str, app_secret: str, refresh_token: str,
                 symbol: str = "$SPX", http: requests.Session | None = None, vix_symbol: str | None = None):
        clean = lambda v: str(v or "").strip().strip("\"'")
        self._key, self._secret, self._refresh = clean(app_key), clean(app_secret), clean(refresh_token)
        self.symbol = symbol
        self.vix_symbols = (clean(vix_symbol),) if clean(vix_symbol) else self.VIX_SYMBOLS
        self.vix_symbol = self.vix_symbols[0]
        self._http = http or requests.Session()
        self._access: str | None = None
        self._expires_at = 0.0

    def _token(self, force: bool = False) -> str:
        if not force and self._access and time.time() < self._expires_at - 60:
            return self._access
        blank = [n for n, v in (('app_key', self._key), ('app_secret', self._secret),
                                ('refresh_token', self._refresh)) if not v]
        if blank:
            raise ProviderError(
                f"these [schwab] secrets are empty: {', '.join(blank)}. Fill them in "
                "(run `python core.py schwab-login` for the refresh_token), save, and restart the app.")
        if self._refresh.startswith("C0.") and self._refresh.lower().endswith(("%40", "@")):
            raise ProviderError(
                "the refresh_token value is an authorization code (it starts with 'C0.' and ends with '@' or "
                "'%40'), copied from the browser address bar. It is not a refresh token. Run "
                "`python core.py schwab-login`, paste the redirected address when asked, and use the "
                "refresh_token it prints.")
        resp = self._http.post(
            f"{self.API}/v1/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": self._refresh},
            auth=(self._key, self._secret), timeout=15)
        if resp.status_code != 200:
            detail = (resp.text or "").strip().replace("\n", " ")[:200]
            raise ProviderError(
                f"Schwab refused the token request (HTTP {resp.status_code}: {detail or 'no detail'}). "
                "Likely causes: the refresh token is older than 7 days, was copied incompletely, or "
                "does not belong to this app key and secret. Run `python core.py schwab-login`, then "
                "update refresh_token (and check app_key and app_secret) in your secrets.")
        body = resp.json()
        self._access = body["access_token"]
        self._expires_at = time.time() + float(body.get("expires_in", 1800))
        self._refresh = body.get("refresh_token", self._refresh)
        return self._access

    def _get(self, path: str, params: dict) -> dict:
        for attempt in (1, 2):
            resp = self._http.get(
                f"{self.API}/marketdata/v1/{path}", params=params, timeout=20,
                headers={"Authorization": f"Bearer {self._token(force=attempt == 2)}"})
            if resp.status_code != 401:
                break
        if resp.status_code != 200:
            raise ProviderError(f"Schwab /{path} returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _last_price(self, symbol: str, keys=("lastPrice", "mark", "closePrice")) -> float:
        data = self._get("quotes", {"symbols": symbol, "fields": "quote"})
        entry = data.get(symbol) if isinstance(data, dict) else None
        if entry is None and isinstance(data, dict) and len(data) == 1:
            entry = next(iter(data.values()))  # one symbol asked, one answered, e.g. a futures root resolved to a contract
        try:
            quote = entry["quote"]
        except (KeyError, TypeError):
            raise ProviderError(f"Schwab returned no quote for {symbol}") from None
        for key in keys:
            v = _na(quote.get(key))
            if v == v and v > 0:
                return v
        raise ProviderError(f"Schwab returned no usable price for {symbol}")

    def spot(self) -> float:
        return self._last_price(self.symbol)

    def vix(self) -> float:
        """Latest VIX index reading: tries $VIX, then VIX (or only the vix_symbol override).

        Uses the last trade or the mark only, never a stale close. The symbol that answered is
        remembered in vix_symbol so later calls try it first.
        """
        errors = []
        for sym in self.vix_symbols:
            try:
                value = self._last_price(sym, keys=("lastPrice", "mark"))
            except ProviderError as exc:
                errors.append(str(exc))
                continue
            self.vix_symbol = sym
            self.vix_symbols = (sym,) + tuple(x for x in self.vix_symbols if x != sym)
            return value
        raise ProviderError("; ".join(errors))

    def expiries(self) -> list[str]:
        data = self._get("expirationchain", {"symbol": self.symbol})
        found = {e["expirationDate"] for e in data.get("expirationList", [])}
        today = date.today().isoformat()
        return sorted(e for e in found if e >= today)

    def chain(self, expiry: str) -> pd.DataFrame:
        data = self._get("chains", {"symbol": self.symbol, "contractType": "ALL",
                                    "fromDate": expiry, "toDate": expiry,
                                    "includeUnderlyingQuote": "false"})
        return self.parse_chain(data)

    @staticmethod
    def parse_chain(data: dict) -> pd.DataFrame:
        rows = []
        for kind, key in (("call", "callExpDateMap"), ("put", "putExpDateMap")):
            for strikes in (data.get(key) or {}).values():
                for contracts in strikes.values():
                    for c in contracts:
                        iv = _na(c.get("volatility"))
                        rows.append(dict(
                            contractSymbol=c.get("symbol", ""), kind=kind,
                            strike=float(c["strikePrice"]),
                            bid=c.get("bid"), ask=c.get("ask"), lastPrice=c.get("last"),
                            volume=c.get("totalVolume"), openInterest=c.get("openInterest"),
                            vendor_iv=iv / 100.0, vendor_delta=_na(c.get("delta"))))
        if not rows:
            raise ProviderError("Schwab returned no contracts for this expiry")
        return finalize_chain(pd.DataFrame(rows))


# ======================================================================================
# 4. SCENARIO ENGINE
# ======================================================================================
def price_vec(S, K, T, r, q, sigma, kind=CALL) -> np.ndarray:
    """Vectorised Black-Scholes-Merton price; inputs broadcast. T <= 0 gives intrinsic value."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(S, K, T, sigma))
    live = (T > 0) & (sigma > 0)
    Tl, sl = np.where(live, T, 1.0), np.where(live, sigma, 1.0)
    sqrt_t = np.sqrt(Tl)
    d1 = (np.log(S / K) + (r - q + 0.5 * sl**2) * Tl) / (sl * sqrt_t)
    d2 = d1 - sl * sqrt_t
    df_r, df_q = np.exp(-r * Tl), np.exp(-q * Tl)
    if kind == CALL:
        model = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        model = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    return np.where(live, model, intrinsic)


def scenario_values(S, strikes, ivs, kind, dte_days, hold_days, r, q, moves,
                    iv_shift_pts: float = 0.0, iv_beta: float = 0.0) -> np.ndarray:
    """Option value at the exit date. Shape (n_strikes, n_moves).

    Each strike keeps its own current IV (sticky-strike), then
        new IV = IV + iv_shift_pts + iv_beta * (SPX move in %)         [vol points]
    so a negative `iv_beta` makes vol fall in rallies and rise in sell-offs.
    """
    strikes = np.asarray(strikes, float)[:, None]
    ivs = np.asarray(ivs, float)[:, None]
    moves = np.asarray(moves, float)[None, :]
    t_left = max(dte_days - hold_days, 0) / 365.0
    iv_new = np.clip(ivs + (iv_shift_pts + iv_beta * moves * 100.0) / 100.0, 0.01, 3.0)
    return price_vec(S * (1.0 + moves), strikes, t_left, r, q, iv_new, kind)


def move_probabilities(moves, mean_move: float, sd_log: float) -> np.ndarray:
    """Probability mass on each SPX move in a grid (ascending), summing to 1.

    The move is lognormal: ln(1+move) ~ Normal with standard deviation `sd_log` and mean chosen so
    that E[move] = `mean_move`. Each grid point gets the mass nearer to it than to its neighbours;
    the two end points also absorb the tails.
    """
    x = np.log1p(np.asarray(moves, float))
    mu = math.log1p(mean_move) - 0.5 * sd_log**2
    edges = (x[1:] + x[:-1]) / 2.0
    cdf = norm.cdf((edges - mu) / sd_log)
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


def fine_grid(mean_move: float, sd_log: float, n: int = 1201, span: float = 6.0):
    """Dense grid of SPX moves covering +/- `span` standard deviations, with its probabilities.

    Used for the table statistics so they do not depend on the (coarser) grid shown in the heatmap
    and are not cut off at its edges. Probabilities of profit are accurate to about 0.15 percentage
    points at the default resolution.
    """
    mu = math.log1p(mean_move) - 0.5 * sd_log**2
    moves = np.expm1(np.linspace(mu - span * sd_log, mu + span * sd_log, n))
    return moves, move_probabilities(moves, mean_move, sd_log)


def regime_masks(moves, flat_band: float):
    """Boolean masks (decline, flat, rise) for moves, with flat meaning |move| <= flat_band."""
    m = np.asarray(moves, float)
    return m < -flat_band - 1e-12, np.abs(m) <= flat_band + 1e-12, m > flat_band + 1e-12


def scenario_breakevens(entry, S, strikes, ivs, kind, dte_days, hold_days, r, q,
                        iv_shift_pts: float = 0.0, iv_beta: float = 0.0) -> np.ndarray:
    """SPX move (decimal) at which each option is worth exactly its entry price at the exit date."""
    out = []
    for px, K, iv in zip(entry, strikes, ivs):
        def f(m, K=K, iv=iv, px=px):
            return float(scenario_values(S, [K], [iv], kind, dte_days, hold_days, r, q, [m],
                                         iv_shift_pts, iv_beta)[0, 0]) - px
        lo, hi = -0.6, 1.5
        out.append(brentq(f, lo, hi) if f(lo) * f(hi) < 0 else math.nan)
    return np.array(out)


def scenario_stats(values, entry, probs, moves, strikes, deltas, breakevens, flat_band: float,
                   display_values=None):
    """Per-strike comparison table plus regime probabilities.

    `values`/`probs`/`moves` should be a dense grid (see fine_grid). `display_values` (optional) is the
    value matrix on the coarser grid shown to the user, used only for the worst/best columns.
    Returns (DataFrame, {"decline": p, "flat": p, "rise": p}). Returns are on premium paid.
    """
    values = np.asarray(values, float)
    entry_col = np.asarray(entry, float)[:, None]
    probs = np.asarray(probs, float)
    ret = values / entry_col - 1.0
    pnl = (values - entry_col) * MULTIPLIER
    shown_ret = ret if display_values is None else np.asarray(display_values, float) / entry_col - 1.0
    exp_ret = ret @ probs
    sd = np.sqrt(np.maximum(((ret - exp_ret[:, None]) ** 2) @ probs, 0.0))
    down, flat, up = regime_masks(moves, flat_band)

    def cond(mask):
        w = probs * mask
        return (ret @ w) / w.sum() if w.sum() > 0 else np.full(len(ret), np.nan)

    table = pd.DataFrame({
        "strike": np.asarray(strikes, float),
        "delta": np.asarray(deltas, float),
        "cost": np.asarray(entry, float) * MULTIPLIER,
        "breakeven": np.asarray(breakevens, float),
        "exp_ret": exp_ret,
        "risk": sd,
        "ret_per_risk": np.where(sd > 0, exp_ret / np.where(sd > 0, sd, 1.0), np.nan),
        "p_profit": (values > entry_col) @ probs,
        "exp_pnl": pnl @ probs,
        "worst": shown_ret.min(axis=1),
        "best": shown_ret.max(axis=1),
        "r_decline": cond(down),
        "r_flat": cond(flat),
        "r_rise": cond(up),
    })
    regimes = {"decline": float(probs[down].sum()), "flat": float(probs[flat].sum()),
               "rise": float(probs[up].sum())}
    return table, regimes


# ======================================================================================
# COMMAND LINE
# ======================================================================================
def _load_secrets() -> dict:
    import tomllib
    from pathlib import Path

    path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        raise SystemExit(f"Create {path} from secrets.toml.example first.")
    return tomllib.loads(path.read_text())


def check_schwab(target_dte: int = 300, target_delta: float = 0.75, rate_pct: float = 4.0) -> None:
    """Smoke-test the Schwab credentials and data end to end, printing what it finds."""
    cfg = _load_secrets().get("schwab")
    if not cfg:
        raise SystemExit("No [schwab] section in .streamlit/secrets.toml")
    prov = SchwabProvider(cfg["app_key"], cfg["app_secret"], cfg["refresh_token"], vix_symbol=cfg.get("vix_symbol"))

    S = prov.spot()
    print(f"1. spot            {S:,.2f}")
    try:
        vix_value = prov.vix()  # tries $VIX, then VIX; vix_symbol then names the one that answered
        print(f"   VIX ({prov.vix_symbol})      {vix_value:.2f}")
    except ProviderError as exc:
        print(f"   VIX             unavailable: {exc}")

    exps = prov.expiries()
    expiry = pick_expiry(exps, target_dte)
    print(f"2. expiries        {len(exps)} listed; nearest {target_dte} DTE is {expiry} ({dte(expiry)} days)")

    chain = prov.chain(expiry)
    two_sided = int((chain.quote == "mid").sum())
    print(f"3. chain           {len(chain)} contracts, {two_sided} two-sided quotes, "
          f"{int(chain.vendor_delta.notna().sum())} with Schwab greeks, "
          f"{int(chain.openInterest.notna().sum())} with open interest")
    if two_sided == 0:
        raise SystemExit("   No two-sided quotes. The market may be closed, or this account lacks SPX data access.")

    r, T = rate_pct / 100.0, max(dte(expiry), 1) / 365.0
    q = implied_carry(chain, S, T, r)
    q_used = 0.013 if q is None else q
    print(f"4. carry           r {r:.2%} (you set), q {q_used:.2%} ({'parity' if q is not None else 'default 1.3%'})")
    pick = pick_contract(enrich(chain, S, T, r, q_used), "call", target_delta)
    if pick is None:
        raise SystemExit("   No priced call found.")
    print(f"5. first call above {target_delta:.0%} delta   strike {pick.strike:,.0f}  mid {pick.mid:,.2f}  "
          f"model delta {pick.delta:+.3f} IV {pick.iv:.1%}  |  Schwab delta {pick.vendor_delta:+.3f} IV {pick.vendor_iv:.1%}")


def schwab_login() -> None:
    """Mint a Schwab refresh token (expires every 7 days) and print the secrets block."""
    import getpass
    import sys
    from urllib.parse import parse_qs, urlencode, urlparse

    app_key = input("App key: ").strip()
    app_secret = getpass.getpass("App secret (hidden): ").strip()
    callback = input("Callback URL registered with Schwab (e.g. https://127.0.0.1): ").strip()
    print("\nOpen this URL, sign in, and approve access:\n")
    print(f"{SchwabProvider.API}/v1/oauth/authorize?{urlencode({'client_id': app_key, 'redirect_uri': callback})}\n")
    pasted = input("Paste the full URL your browser was redirected to: ").strip()
    code = parse_qs(urlparse(pasted).query).get("code", [None])[0]
    if not code:
        sys.exit("No ?code= found in that URL.")
    resp = requests.post(f"{SchwabProvider.API}/v1/oauth/token", auth=(app_key, app_secret), timeout=15,
                         data={"grant_type": "authorization_code", "code": code, "redirect_uri": callback})
    if resp.status_code != 200:
        sys.exit(f"Token exchange failed (HTTP {resp.status_code}): {resp.text[:300]}")
    print("\nAdd this to your Streamlit secrets (valid 7 days):\n\n[schwab]")
    print(f'app_key = "{app_key}"\napp_secret = "{app_secret}"\nrefresh_token = "{resp.json()["refresh_token"]}"')


if __name__ == "__main__":
    import sys

    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "check-schwab":
        check_schwab(int(args[0]) if args else 300, float(args[1]) if len(args) > 1 else 0.75,
                     float(args[2]) if len(args) > 2 else 4.0)
    elif cmd == "schwab-login":
        schwab_login()
    else:
        print(__doc__)