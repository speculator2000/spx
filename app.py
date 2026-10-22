"""SPX option desk: scan, price and track SPX options (default: ~300 DTE, first strike above 75 delta)."""
from __future__ import annotations

import math
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

# ------------------------------------------------------------------ design tokens
INK, IVORY = "#2B3B50", "#F6F4EE"
EMERALD, GOLD, BURGUNDY = "#33604F", "#B0925F", "#8A4A4A"
MULT = core.MULTIPLIER
STRIKES_EACH_SIDE = 5  # rows shown above and below the selected strike in the Scanner table
DEFAULT_QTY = 4.0      # contracts entered when a position is added from the Scanner

st.set_page_config(page_title="SPX option desk", page_icon="📈", layout="wide")


def themed_layout() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; color: {INK}; }}
        .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
        [data-testid="stSidebar"] {{ background: {INK}; }}
        [data-testid="stSidebar"] * {{ color: {IVORY}; }}
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {INK}; }}
        .masthead {{ border-bottom: 2px solid {INK}; padding-bottom: .6rem; margin-bottom: 1.2rem; }}
        .masthead h1 {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.2rem; margin: 0; color: {INK}; }}
        .masthead p {{ margin: .2rem 0 0; color: {INK}; opacity: .75; font-size: .95rem; }}
        .section-h {{ font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.3rem;
                      border-left: 4px solid {GOLD}; padding-left: .6rem; margin: 1.6rem 0 .7rem; }}
        .kpi {{ background: #fff; border: 1px solid #DDD8C8; border-top: 3px solid {EMERALD};
                padding: .7rem .9rem; border-radius: 3px; }}
        .kpi .l {{ font-size: .78rem; opacity: .7; }}
        .kpi .v {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.45rem; font-weight: 500; }}
        .kpi .s {{ font-size: .75rem; opacity: .6; }}
        .kpi.gold {{ border-top-color: {GOLD}; }}
        .kpi.burgundy {{ border-top-color: {BURGUNDY}; }}
        div[data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def masthead(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="masthead"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-h">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
        font=dict(family="Inter", color=INK),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(gridcolor="#E7E3D6"), yaxis=dict(gridcolor="#E7E3D6"),
    )
    return fig


# ------------------------------------------------------------------------ data sources
SRC_YAHOO, SRC_SCHWAB = "Yahoo Finance (delayed)", "Schwab (live)"
SOURCES = [SRC_YAHOO, SRC_SCHWAB]  # the first entry is the default


def schwab_credentials() -> tuple[str, str, str, str] | None:
    try:
        c = st.secrets["schwab"]
        return c["app_key"], c["app_secret"], c["refresh_token"], str(c.get("vix_symbol", "") or "")
    except Exception:  # no secrets file, or an incomplete [schwab] section
        return None


@st.cache_resource(show_spinner=False)
def _schwab_provider(app_key: str, app_secret: str, refresh_token: str, vix_symbol: str = "") -> core.SchwabProvider:
    """One provider per credential set, so the access token survives between reruns."""
    return core.SchwabProvider(app_key, app_secret, refresh_token, vix_symbol=vix_symbol or None)


def get_provider(source: str):
    if source == SRC_SCHWAB:
        creds = schwab_credentials()
        if creds is None:
            raise core.ProviderError(
                "no Schwab credentials found. Add a [schwab] section with app_key, app_secret and "
                "refresh_token to .streamlit/secrets.toml (or the Community Cloud Secrets box). "
                "`python core.py schwab-login` mints the refresh token.")
        return _schwab_provider(*creds)
    return core.YahooProvider()


@st.cache_data(ttl=15, show_spinner=False)
def cached_quotes(source: str) -> dict:
    """SPX and VIX from the chosen source, stamped with the time they were fetched (New York time)."""
    provider = get_provider(source)
    spot = float(provider.spot())  # a failure here stops the app with the vendor's message
    try:
        vix, vix_err = float(provider.vix()), None
    except Exception as exc:  # the tables show a dash and one warning instead of failing the page
        vix, vix_err = None, str(exc)
    return dict(spot=spot, vix=vix, vix_err=vix_err, fetched=datetime.now(MARKET_TZ).replace(microsecond=0, tzinfo=None))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rate() -> float:
    return core.get_rate()


@st.cache_data(ttl=900, show_spinner=False)
def cached_expiries(source: str) -> list[str]:
    return get_provider(source).expiries()


@st.cache_data(ttl=30, show_spinner=False)
def cached_chain(source: str, expiry: str) -> pd.DataFrame:
    return get_provider(source).chain(expiry)


def analyse(source: str, expiry: str, S: float, r: float, carry_mode: str, q_fixed: float):
    """Chain with IV and Greeks for one expiry. Returns (df, T, q_used, q_source)."""
    chain = cached_chain(source, expiry)
    T = max(core.dte(expiry), 1) / 365.0
    q, src = q_fixed, "fixed"
    if carry_mode.startswith("Infer"):
        q_hat = core.implied_carry(chain, S, T, r)
        if q_hat is not None:
            q, src = q_hat, "put-call parity"
    return core.enrich(chain, S, T, r, q), T, q, src


# ---------------------------------------------------------------------------- state
MARKET_TZ = ZoneInfo("America/New_York")  # entry stamps use U.S. market time, whatever the server clock says
POS_COLS = ["entry_dt", "expiry", "kind", "strike", "qty", "entry_price", "entry_spot", "entry_vix"]
REQUIRED_POS_COLS = ["expiry", "kind", "strike", "qty", "entry_price"]  # older files without the entry stamp still load
POS_NUMERIC = ["strike", "qty", "entry_price", "entry_spot", "entry_vix"]


def _as_date(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


def _as_time(v):
    if v is None:
        return None
    if isinstance(v, dtime):
        return v.replace(microsecond=0)
    if not isinstance(v, str) and pd.isna(v):
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(v).strip(), fmt).time()
        except ValueError:
            pass
    return None


def _entry_datetimes(df: pd.DataFrame) -> pd.Series:
    """One New York wall-clock datetime per row.

    Reads entry_dt, or the older separate entry_date and entry_time columns (either may be missing).
    """
    if "entry_dt" in df.columns:
        dt = pd.to_datetime(df["entry_dt"], errors="coerce", utc=False)
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
        return dt
    if "entry_date" not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    day = pd.to_datetime(df["entry_date"].map(_as_date), errors="coerce")
    if "entry_time" in df.columns:
        secs = df["entry_time"].map(_as_time).map(
            lambda t: 0 if t is None or t is pd.NaT or (not isinstance(t, dtime) and pd.isna(t))
            else t.hour * 3600 + t.minute * 60 + t.second)
        day = day + pd.to_timedelta(secs.astype(float), unit="s")
    return day


def vix_change(entry, current):
    """Percentage change in the VIX since entry, or NaN when either reading is missing."""
    try:
        e, c = float(entry), float(current)
    except (TypeError, ValueError):
        return np.nan
    return (c / e - 1.0) * 100.0 if e > 0 and c == c and c > 0 else np.nan


def days_open(entry, now):
    """Whole calendar days from the entry date to now (0 on the day of entry), or NaN without an entry date."""
    if entry is None or pd.isna(entry):
        return np.nan
    return float((pd.Timestamp(now).normalize() - pd.Timestamp(entry).normalize()).days)


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Positions with every expected column, a real datetime for the entry stamp and float numbers."""
    out = df.copy()
    entry_dt = _entry_datetimes(out)
    for col in POS_COLS:
        if col not in out.columns:
            out[col] = None
    out = out[POS_COLS].copy()
    out["entry_dt"] = entry_dt.astype("datetime64[ns]").dt.floor("s")
    for col in ("expiry", "kind"):
        out[col] = out[col].map(lambda v: None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v))
    for col in POS_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


SNAP_COLS = ["ts", "spot", "expiry", "dte", "kind", "strike", "mid", "iv", "delta"]
if "positions" not in st.session_state:
    st.session_state.positions = normalize_positions(pd.DataFrame(columns=POS_COLS))
# Streamlit keeps session state across code reloads, so a table saved by an older version of this
# app can lack newer columns. Normalising on every run upgrades it in place and changes nothing else.
st.session_state.positions = normalize_positions(st.session_state.positions)
if "snaps" not in st.session_state:
    st.session_state.snaps = pd.DataFrame(columns=SNAP_COLS)

themed_layout()

# -------------------------------------------------------------------------- sidebar
with st.sidebar:
    if st.button("Refresh market data", type="primary", use_container_width=True,
                 help="Clears every cached quote and chain (Yahoo and Schwab), starts a fresh Schwab session, "
                      "and reloads spot, VIX, option prices and the current date/time."):
        st.cache_data.clear()
        st.cache_resource.clear()  # rebuilds the Schwab provider, so it also gets a fresh access token
        st.rerun()
    st.markdown("### Data source")
    source = st.selectbox("Source", SOURCES, label_visibility="collapsed")
    if source == SRC_YAHOO:
        st.caption("Free but delayed. Switch to Schwab for live quotes once your credentials are set up.")
    st.markdown("### Target contract")
    target_dte = st.number_input("Days to expiration", 30, 1100, 300, step=10)
    target_delta = st.slider("Target delta (absolute)", 0.05, 0.95, 0.75, 0.01,
                             help="The Scanner selects the first strike whose delta is above this value.")
    kind_label = st.radio("Option type", ["Call", "Put"], horizontal=True)
    kind = kind_label.lower()

    st.markdown("### Rates and carry")
    rate_mode = st.radio("Risk-free rate", ["13-week T-bill (live)", "Manual"])
    r_manual = st.number_input("Manual rate (%)", 0.0, 15.0, 4.0, 0.05,
                               help="A Treasury yield near the option's expiry (about 1 year for 300 days) "
                                    "is closer than the 13-week bill.") / 100
    carry_mode = st.radio("Dividend yield", ["Infer from put-call parity", "Fixed"])
    q_fixed = st.number_input("Fixed dividend yield (%)", -3.0, 8.0, 1.3, 0.05) / 100

# ----------------------------------------------------------------------- market data
try:
    quotes = cached_quotes(source)
    S = quotes["spot"]
    r = cached_rate() if rate_mode.startswith("13") else r_manual
    expiries = cached_expiries(source)
except Exception as exc:  # expired token, network, missing credentials, or a vendor error
    st.error(f"Could not load SPX market data from {source}: {exc}")
    st.stop()

if not expiries:
    st.error(f"{source} returned no SPX expiries. Try again in a minute.")
    st.stop()

vix_text = f"{quotes['vix']:.2f}" if quotes["vix"] is not None else "unavailable"
masthead("SPX option desk",
         f"Spot {S:,.2f} &nbsp;|&nbsp; VIX {vix_text} &nbsp;|&nbsp; risk-free {r:.2%} &nbsp;|&nbsp; "
         f"{source} &nbsp;|&nbsp; quotes fetched {quotes['fetched']:%H:%M:%S} ET")
with st.sidebar:
    st.caption("ⓒ Franklin Chidi (FC) - Alpha Logic LLC")

scan_tab, track_tab, method_tab = st.tabs(["Scanner", "Tracker", "Methodology"])

# ------------------------------------------------------------------------- scanner
with scan_tab:
    candidates = [e for e in expiries if 30 <= core.dte(e) <= 1100] or expiries
    default_exp = core.pick_expiry(candidates, int(target_dte))
    expiry = st.selectbox(
        "Expiry", candidates, index=candidates.index(default_exp),
        format_func=lambda e: f"{e}  ({core.dte(e)} days)")

    try:
        df, T, q, q_src = analyse(source, expiry, S, r, carry_mode, q_fixed)
    except Exception as exc:
        st.error(f"Could not load the {expiry} chain: {exc}")
        st.stop()

    # A table row the user clicks becomes the selected contract for the whole page. Streamlit reports the click
    # on the next run, so it is read here, before the pick is used. The choice is kept as a strike (not a row
    # number) and only for this source, expiry, type and target delta; changing any of them goes back to the
    # first strike above the target.
    ctx = (source, expiry, kind, round(float(target_delta), 4))
    sel = st.session_state.get("band_sel") or {}
    if sel.get("ctx") != ctx:
        sel = dict(ctx=ctx, strikes=[], ver=sel.get("ver", 0) + 1, rows=[], manual=None)
    try:
        rows = list(st.session_state[f"band_table_{sel['ver']}"]["selection"]["rows"])
    except Exception:  # no table state yet
        rows = []
    if rows != sel["rows"]:  # the user clicked a row in the table drawn on the previous run
        sel["rows"] = rows
        sel["manual"] = sel["strikes"][rows[0]] if rows and rows[0] < len(sel["strikes"]) else None

    pick, manual_pick = None, False
    if sel["manual"] is not None:
        chosen = df[(df.kind == kind) & (df.strike == sel["manual"]) & df.delta.notna() & df.iv.notna()]
        if not chosen.empty:
            pick, manual_pick = chosen.iloc[0], True
    if pick is None:
        pick = core.pick_contract(df, kind, target_delta)
    if pick is None:
        st.warning("No contract with a usable implied vol on this expiry. Pick another expiry.")
        st.stop()

    # The table: the selected strike with up to STRIKES_EACH_SIDE strikes above and below it.
    ladder = df[(df.kind == kind) & df.delta.notna()].sort_values("strike").reset_index(drop=True)
    centre = int((ladder.strike - pick.strike).abs().idxmin())
    band = ladder.iloc[max(0, centre - STRIKES_EACH_SIDE): centre + STRIKES_EACH_SIDE + 1].copy()
    optional = {"volume", "openInterest", "vendor_iv", "vendor_delta"}
    cols = ["strike", "bid", "ask", "mid", "quote", "iv", "delta", "gamma", "vega", "theta",
            "volume", "openInterest", "vendor_iv", "vendor_delta"]
    cols = [c for c in cols if c not in optional or band[c].notna().any()]
    band = band[cols].reset_index(drop=True)
    strikes = band.strike.tolist()
    if strikes != sel["strikes"]:  # a new centre or moved prices: draw the table fresh (the choice is kept)
        sel.update(strikes=strikes, ver=sel["ver"] + 1, rows=[])
    st.session_state["band_sel"] = sel

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(kpi("Expiry", expiry, f"{core.dte(expiry)} days, q {q:.2%} ({q_src})"), unsafe_allow_html=True)
    k2.markdown(kpi("Selected strike (your choice)" if manual_pick else f"First strike above {target_delta:.0%} delta",
                    f"{pick.strike:,.0f}", f"{pick.strike / S - 1:+.1%} vs spot", "gold"), unsafe_allow_html=True)
    k3.markdown(kpi("Mid price", f"{pick.mid:,.2f}", f"bid {pick.bid:,.2f} / ask {pick.ask:,.2f}"), unsafe_allow_html=True)
    k4.markdown(kpi("Delta", f"{pick.delta:+.3f}", f"gamma {pick.gamma:.5f}"), unsafe_allow_html=True)
    k5.markdown(kpi("Implied vol", f"{pick.iv:.1%}", f"cost ${pick.mid * MULT:,.0f} per contract", "burgundy"),
                unsafe_allow_html=True)
    if manual_pick:
        auto = core.pick_contract(df, kind, target_delta)
        st.caption(f"You selected the {pick.strike:,.0f} strike from the table"
                   + (f" instead of the first strike above {target_delta:.0%} delta ({auto.strike:,.0f})." if auto is not None
                      and auto.strike != pick.strike else ", which is also the first strike above the target.")
                   + " The cards, chart, snapshot and position all use it.")
        if st.button("Reset to the first strike above the target"):
            sel.update(manual=None, rows=[], ver=sel["ver"] + 1)  # a new table key clears the highlighted row
            st.session_state["band_sel"] = sel
            st.rerun()
    elif abs(pick.delta) <= target_delta:
        st.caption(f"No strike on this expiry has a delta above {target_delta:.0%}. Showing the highest available "
                   f"({abs(pick.delta):.3f}).")
    if pick.quote == "last":
        st.caption("This strike has no two-sided quote, so the price is the last trade and may be stale.")
    if pd.notna(pick.vendor_delta) or pd.notna(pick.vendor_iv):
        st.caption(f"Cross-check from {source}: delta {pick.vendor_delta:+.3f}, IV {pick.vendor_iv:.1%} "
                   f"(this app: delta {pick.delta:+.3f}, IV {pick.iv:.1%}).")

    section_header("Strikes around the target delta")
    fmt = {"strike": "{:,.0f}", "bid": "{:,.2f}", "ask": "{:,.2f}", "mid": "{:,.2f}", "iv": "{:.1%}",
           "delta": "{:+.3f}", "gamma": "{:.5f}", "vega": "{:.2f}", "theta": "{:.2f}", "volume": "{:,.0f}",
           "openInterest": "{:,.0f}", "vendor_iv": "{:.1%}", "vendor_delta": "{:+.3f}"}
    st.caption(f"The selected strike and the {STRIKES_EACH_SIDE} strikes above and below it. Click a row to make that strike "
               "the selected contract (the table then re-centres on it). The highlighted row is the one in use.")
    st.dataframe(
        band.style.format({k: v for k, v in fmt.items() if k in cols}, na_rep="-")
        .apply(lambda row: [f"background-color: {GOLD}33" if row.strike == pick.strike else "" for _ in row], axis=1),
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row",
        key=f"band_table_{sel['ver']}")

    section_header("Implied vol by strike")
    smile = df[(df.kind == kind) & df.iv.notna() & (df.quote == "mid")].sort_values("strike")
    fig = go.Figure(go.Scatter(x=smile.strike, y=smile.iv, mode="lines+markers",
                               line=dict(color=EMERALD), marker=dict(size=4), name=f"{kind_label} IV"))
    fig.add_trace(go.Scatter(x=[pick.strike], y=[pick.iv], mode="markers", name="Selected",
                             marker=dict(color=GOLD, size=12, line=dict(color=INK, width=1))))
    fig.add_vline(x=S, line_dash="dot", line_color=BURGUNDY, annotation_text="spot")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("Log snapshot of this contract", use_container_width=True):
        row = dict(ts=datetime.now().isoformat(timespec="seconds"), spot=S, expiry=expiry,
                   dte=core.dte(expiry), kind=kind, strike=pick.strike, mid=pick.mid,
                   iv=pick.iv, delta=pick.delta)
        st.session_state.snaps = pd.concat([st.session_state.snaps, pd.DataFrame([row])], ignore_index=True)
        st.toast("Snapshot logged. See the Tracker tab.")
    if c2.button("Add to positions at the mid price", use_container_width=True):
        stamp = datetime.now(MARKET_TZ).replace(microsecond=0, tzinfo=None)
        provider = get_provider(source)
        notes = []
        try:
            spot_now = round(float(provider.spot()), 2)  # fetched now, not from the cache
        except Exception as exc:
            spot_now = round(float(S), 2)
            notes.append(f"the entry spot is the last loaded SPX value ({exc})")
        try:
            vix_now = round(float(provider.vix()), 2)
        except Exception as exc:
            vix_now = None
            notes.append(f"no VIX value ({exc}); type it into the VIX column on the Tracker tab")
        if notes:
            st.warning("Position added, but " + "; ".join(notes) + ".")
        new_row = normalize_positions(pd.DataFrame([dict(
            entry_dt=stamp, expiry=expiry, kind=kind, strike=float(pick.strike), qty=DEFAULT_QTY,
            entry_price=round(float(pick.mid), 2), entry_spot=spot_now, entry_vix=vix_now)]))
        held = st.session_state.positions
        st.session_state.positions = new_row if held.empty else pd.concat([held, new_row], ignore_index=True)
        st.toast(f"Position added at {stamp:%H:%M:%S} ET, SPX {spot_now:,.2f}"
                 + (f", VIX {vix_now:.2f}." if vix_now is not None else ".") + " See the Tracker tab.")

# ------------------------------------------------------------------------- tracker
with track_tab:
    section_header("Positions")
    st.caption("Streamlit Community Cloud does not keep files between restarts, so download your positions "
               "and snapshots and upload them next time.")
    up = st.file_uploader("Load positions (CSV)", type="csv", key="pos_up")
    if up is not None and st.session_state.get("pos_up_name") != up.file_id:
        loaded = pd.read_csv(up)
        if set(REQUIRED_POS_COLS).issubset(loaded.columns):
            st.session_state.positions = normalize_positions(loaded)
            st.session_state.pos_up_name = up.file_id
            st.rerun()
        else:
            st.error(f"CSV needs these columns: {', '.join(REQUIRED_POS_COLS)}. "
                     "The entry_dt (or older entry_date and entry_time), entry_spot and entry_vix columns are optional.")

    edited = st.data_editor(
        st.session_state.positions, num_rows="dynamic", use_container_width=True, key="pos_editor",
        column_config={
            "entry_dt": st.column_config.DatetimeColumn("Entry date (ET)", format="YYYY-MM-DD HH:mm:ss", step=1),
            "entry_spot": st.column_config.NumberColumn("Entry Spot", format="%.2f", min_value=0),
            "entry_vix": st.column_config.NumberColumn("VIX at entry", format="%.2f", min_value=0),
            "expiry": st.column_config.SelectboxColumn("Expiry", options=expiries, required=True),
            "kind": st.column_config.SelectboxColumn("Type", options=["call", "put"], required=True),
            "strike": st.column_config.NumberColumn("Strike", format="%.0f", min_value=0),
            "qty": st.column_config.NumberColumn("Quantity (+long / -short)", step=1),
            "entry_price": st.column_config.NumberColumn("Entry price", format="%.2f", min_value=0),
        })
    st.session_state.positions = edited

    now_et = datetime.now(MARKET_TZ).replace(microsecond=0, tzinfo=None)
    if quotes["vix"] is None and not edited.dropna(subset=["expiry", "kind", "strike", "qty", "entry_price"]).empty:
        st.warning(f"Current VIX unavailable from {source}: {quotes['vix_err']}")
    marks = []
    for _, pos in edited.dropna(subset=["expiry", "kind", "strike", "qty", "entry_price"]).iterrows():
        try:
            pdf_, pT, pq, _src = analyse(source, str(pos.expiry), S, r, carry_mode, q_fixed)
            hit = pdf_[(pdf_.kind == pos.kind) & (pdf_.strike == pos.strike)]
            if hit.empty or pd.isna(hit.iloc[0].iv):
                raise ValueError("no priced contract at this strike")
            h = hit.iloc[0]
            marks.append(dict(
                entry_dt=pos.get("entry_dt"), current_dt=now_et, days_open=days_open(pos.get("entry_dt"), now_et), entry_spot=pos.get("entry_spot"), current_spot=S,
                spot_diff=S - pos.get("entry_spot"), entry_vix=pos.get("entry_vix"), current_vix=quotes["vix"],
                vix_diff=vix_change(pos.get("entry_vix"), quotes["vix"]), expiry=pos.expiry, type=pos.kind, strike=pos.strike, qty=pos.qty, dte=core.dte(str(pos.expiry)),
                entry_price=pos.entry_price, mark=h.mid, iv=h.iv, delta=h.delta,
                pnl=(h.mid - pos.entry_price) * pos.qty * MULT,
                pos_delta=h.delta * pos.qty * MULT, pos_theta=h.theta * pos.qty * MULT,
                pos_vega=h.vega * pos.qty * MULT))
        except Exception as exc:
            marks.append(dict(entry_dt=pos.get("entry_dt"), current_dt=now_et, days_open=days_open(pos.get("entry_dt"), now_et), entry_spot=pos.get("entry_spot"),
                              current_spot=S, spot_diff=np.nan, entry_vix=pos.get("entry_vix"),
                              current_vix=quotes["vix"], vix_diff=np.nan, expiry=pos.expiry, type=pos.kind, strike=pos.strike, qty=pos.qty,
                              dte=np.nan, entry_price=pos.entry_price, mark=np.nan, iv=np.nan, delta=np.nan,
                              pnl=np.nan, pos_delta=np.nan, pos_theta=np.nan, pos_vega=np.nan))
            st.warning(f"Could not mark {pos.expiry} {pos.kind} {pos.strike:,.0f}: {exc}")

    if marks:
        section_header("Marked to market")
        mk = pd.DataFrame(marks)
        t1, t2, t3, t4 = st.columns(4)
        t1.markdown(kpi("Total P&L", f"${mk.pnl.sum():,.0f}", "mid vs entry, 100x multiplier",
                        "gold" if mk.pnl.sum() >= 0 else "burgundy"), unsafe_allow_html=True)
        t2.markdown(kpi("Position delta", f"{mk.pos_delta.sum():,.1f}", "SPX points of exposure"), unsafe_allow_html=True)
        t3.markdown(kpi("Theta per day", f"${mk.pos_theta.sum():,.0f}"), unsafe_allow_html=True)
        t4.markdown(kpi("Vega per vol point", f"${mk.pos_vega.sum():,.0f}"), unsafe_allow_html=True)
        # Column order of the table. The position delta, theta and vega stay out; they feed the totals above.
        MARK_ORDER = ["expiry", "type", "strike", "dte", "entry_spot", "current_spot", "spot_diff",
                      "entry_vix", "current_vix", "vix_diff", "qty", "entry_price", "mark", "pnl",
                      "delta", "iv", "entry_dt", "current_dt", "days_open"]
        # column -> (header shown, number format)
        MARK_COLS = {
            "expiry": ("Expiry", None), "strike": ("Strike", "{:,.0f}"),
            "entry_spot": ("Spot (Entry)", "{:,.2f}"), "current_spot": ("Spot (Curr)", "{:,.2f}"),
            "spot_diff": ("Spot (diff)", "{:+,.2f}"),
            "entry_vix": ("VIX (Entry)", "{:.2f}"), "current_vix": ("VIX (Curr)", "{:.2f}"),
            "vix_diff": ("VIX (diff)", "{:+.2f}%"),
            "type": ("Type", None), "qty": ("Qty", "{:+.0f}"), "dte": ("DTE", "{:.0f}"),
            "entry_price": ("Entry Price", "{:,.2f}"), "mark": ("Market", "{:,.2f}"),
            "iv": ("IV", "{:.1%}"), "delta": ("Delta", "{:+.3f}"), "pnl": ("Profit-Loss", "${:,.0f}"),
            "entry_dt": ("Entry Date", "{:%Y-%m-%d %H:%M:%S}"), "current_dt": ("Current Date", "{:%Y-%m-%d %H:%M:%S}"),
            "days_open": ("Days Open", "{:.0f}"),
        }
        shown = mk[MARK_ORDER].rename(columns={c: v[0] for c, v in MARK_COLS.items()})
        st.dataframe(
            shown.style.format({MARK_COLS[c][0]: f for c, f in ((c, v[1]) for c, v in MARK_COLS.items()) if f}, na_rep="-"),
            use_container_width=True, hide_index=True)

    st.download_button("Download positions (CSV)", edited.to_csv(index=False), "spx_positions.csv", "text/csv")

    section_header("Snapshot history")
    up2 = st.file_uploader("Load snapshots (CSV)", type="csv", key="snap_up")
    if up2 is not None and st.session_state.get("snap_up_name") != up2.file_id:
        loaded = pd.read_csv(up2)
        if set(SNAP_COLS).issubset(loaded.columns):
            st.session_state.snaps = pd.concat([loaded[SNAP_COLS], st.session_state.snaps], ignore_index=True) \
                .drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
            st.session_state.snap_up_name = up2.file_id
            st.rerun()
        else:
            st.error(f"CSV needs these columns: {', '.join(SNAP_COLS)}")

    snaps = st.session_state.snaps
    if snaps.empty:
        st.info("No snapshots yet. Use \"Log snapshot of this contract\" on the Scanner tab each time you check in.")
    else:
        if len(snaps) >= 2:
            sx = pd.to_datetime(snaps.ts)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=sx, y=snaps.mid, name="Mid price", line=dict(color=EMERALD)))
            fig3.add_trace(go.Scatter(x=sx, y=snaps.iv, name="Implied vol", yaxis="y2", line=dict(color=GOLD)))
            fig3.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
            st.plotly_chart(style_fig(fig3), use_container_width=True)
        st.dataframe(snaps.style.format({"spot": "{:,.2f}", "strike": "{:,.0f}", "mid": "{:,.2f}",
                                         "iv": "{:.1%}", "delta": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)
        st.download_button("Download snapshots (CSV)", snaps.to_csv(index=False), "spx_snapshots.csv", "text/csv")


# --------------------------------------------------------------------------- methodology
def md(text: str) -> None:
    """Markdown with dollar signs escaped, so Streamlit does not read '$...$' as maths."""
    st.markdown(text.replace("$", "\\$"))


with method_tab:
    st.caption("How the numbers in this app are built, what the terms mean, and how to tell an uptrend from a "
               "decline. Educational reference only, not investment advice.")

    section_header("1. Option basics")
    md("""
| Term | Meaning here |
|---|---|
| **SPX** | The S&P 500 index. SPX options are **European** (exercised only at expiry) and **cash-settled**, so no shares change hands. |
| **Multiplier** | Each index point is worth $100. An option quoted at 650.00 costs $65,000 per contract. |
| **Call / put** | A call gains when SPX rises; a put gains when it falls. |
| **Strike** | The index level the option is struck at. |
| **Expiry and DTE** | The expiration date, and *days to expiration*: calendar days from today to that date. |
| **Bid, ask, mid** | What buyers bid, what sellers ask, and the midpoint of the two. The app values contracts at the **mid** when both sides are quoted, otherwise at the last trade (such rows are flagged). |
| **Premium / cost** | The price paid for the option. "Cost per contract" on the Scanner is the mid price × $100. |
| **Intrinsic and time value** | Intrinsic value is what the option is worth if exercised now (call: SPX − strike, never below zero). Everything above that is time value, which decays toward expiry. |
| **In / at / out of the money** | A call is in the money when SPX is above the strike. A 75-delta call is well in the money. |
| **SPX vs SPXW** | Monthly (SPX) and weekly/daily (SPXW) contracts can share an expiry date. The app keeps the one with the larger open interest. |
""")

    section_header("2. How options are priced")
    md("""
Prices and Greeks use the **Black-Scholes-Merton** model with a continuous dividend yield, which suits
European, cash-settled index options.

- **Inputs:** SPX level (S), strike (K), time to expiry in years (days ÷ 365), risk-free rate (r),
  dividend yield (q) and volatility (σ).
- **Risk-free rate:** by default the 13-week Treasury bill yield from Yahoo Finance. Pick *Manual* in the sidebar
  to use a yield nearer the option's term (about one year for 300 days).
- **Dividend yield:** by default *inferred from put-call parity* near the money, which captures what the market
  itself is pricing in. *Fixed* uses the value you type.
- **Implied volatility (IV):** the σ that makes the model price equal the contract's mid price, found numerically.
  It is the market's expectation of annualised volatility over the option's life, not a forecast of direction.
  Prices that break no-arbitrage bounds get no IV and are excluded.
- **Smile / skew:** IV differs by strike (usually higher for low strikes). The Scanner's chart shows this.

**Greeks** measure how an option's value responds to inputs:

| Greek | What it measures | Units in this app |
|---|---|---|
| **Delta** | Change in option price for a 1-point SPX move. Roughly, the chance of finishing in the money (a rule of thumb, since it is risk-neutral). A call is 0 to +1, a put 0 to −1. | per index point |
| **Gamma** | How fast delta itself changes as SPX moves. | delta per point |
| **Theta** | Value lost per day from time passing, other things equal. | per calendar day |
| **Vega** | Change in value for a 1 percentage-point rise in IV. | per vol point |
| **Rho** | Change in value for a 1 percentage-point rise in the interest rate. | per rate point |

Per-position figures multiply by quantity and the $100 multiplier.
""")

    section_header("3. Choosing the contract (Scanner)")
    md("""
- The Scanner picks the expiry closest to your **target DTE** (default 300 days).
- It then selects the **first strike whose absolute delta is strictly above your target** (default 0.75), rather than
  the strike closest to it, which could sit just under. The pick is therefore always above the target. Contracts with both a
  bid and an ask are preferred. If no strike exceeds the target, it falls back to the highest delta available and says so.
- The "Strikes around the target delta" table lists the selected strike and the 5 strikes above and below it.
- You can override the pick: **click a row** in the "Strikes around the target delta" table and that strike becomes the
  active contract for the cards, chart, snapshot and position. The table then re-centres on it. The choice lasts until you change the
  expiry, call/put, source or target delta, or press *Reset*.
- Higher delta means deeper in the money: a higher price, less time value, and behaviour closer to owning the index.
  Lower delta means cheaper, more leveraged and more dependent on a large move.
""")

    section_header("4. Tracker columns")
    md("""
| Column | Definition |
|---|---|
| **Expiry, Type, Strike** | The contract you hold. |
| **DTE** | Calendar days from today to expiry. |
| **Spot (Entry)** | SPX when you added the position (fetched at that moment). |
| **Spot (Curr)** | SPX now, from the selected data source. |
| **Spot (diff)** | Spot (Curr) − Spot (Entry), in index points. |
| **VIX (Entry)** | The VIX when you added the position. |
| **VIX (Curr)** | The VIX now. |
| **VIX (diff)** | Percentage change in the VIX since entry: (current ÷ entry − 1) × 100. |
| **Qty** | Contracts held. Positive is long, negative is short. Positions added from the Scanner start at 4 contracts; edit the number in the table. |
| **Entry Price** | The option price you paid (mid when added). |
| **Market** | The option's current mid price. |
| **Profit-Loss** | (Market − Entry Price) × Qty × $100. Unrealised, before commissions. |
| **Delta** | The contract's current delta. |
| **IV** | The contract's current implied volatility. |
| **Entry Date** | When the position was added, in New York time. |
| **Current Date** | The time of the latest page refresh, in New York time. |
| **Days Open** | Whole calendar days from the entry date to today (0 on the day of entry). |

The tiles above the table total the whole book: **Position delta** (SPX points of exposure, delta × qty × 100),
**Theta per day** and **Vega per vol point** in dollars.

**The VIX** is the market's 30-day expected volatility of the S&P 500, derived from SPX option prices and quoted in
annualised percentage points. Yahoo uses ^VIX; Schwab uses $VIX, then VIX. VIX *futures* (such as /VX) trade at a
premium or discount to it, so they are not interchangeable.
""")

    section_header("5. Upward trend versus decline in the SPX")
    md("""
**What counts as a trend.** The app does not forecast or classify the market for you. The indicators below are the
conventional ways to judge whether SPX is trending up or down, and so which strike and expiry may suit your view. Rank
them by how many agree, not by any single one.

| Indicator | Points to an uptrend | Points to a decline |
|---|---|---|
| **Price vs the 200-day average** | SPX above a rising 200-day moving average | SPX below a falling 200-day average |
| **50-day vs 200-day average** | 50-day above the 200-day ("golden cross" on the crossover) | 50-day below the 200-day ("death cross" on the crossover) |
| **Swing structure** | Higher highs and higher lows; pullbacks hold above the prior low | Lower highs and lower lows; rallies fail below the prior high |
| **Distance from the high** | Within about 5% of the record high | Down 10% or more (a "correction"); down 20% or more is the usual definition of a bear market |
| **Momentum** | RSI mostly above 50 and not stuck below it; MACD above its signal line | RSI mostly below 50; MACD below its signal line |
| **The VIX** | Low and drifting down (rule of thumb: below about 20) | Rising sharply, often above 25 to 30 in stress |
| **VIX term structure** | Futures priced above spot VIX (contango) | Spot VIX above futures (backwardation), a sign of acute stress |
| **Market breadth** | A large share of S&P 500 stocks above their own 200-day averages; advances lead declines | Few stocks above their averages; leadership narrows to a handful of names |
| **Option skew** | Moderate skew; put demand is not rising fast | Steepening skew, with puts bid up relative to calls |

SPX and the VIX normally move in opposite directions: implied volatility tends to fall in rallies and rise in
sell-offs, which changes what options cost. A steady rise in the index with a falling VIX is the classic uptrend signature; a falling index with a
rising VIX confirms a decline. When the two rise together, treat the move with caution.

Trend indicators describe the past. They improve odds at best, they lag at turning points, and they say nothing
certain about the future.
""")

    section_header("6. Data sources and caveats")
    md("""
- **Yahoo Finance** is free and delayed (the default). **Schwab** is live and needs the credentials in your secrets.
- *Refresh market data* clears every cached quote and chain, starts a fresh Schwab session and reloads the clock.
  Otherwise spot and VIX are re-used for 15 seconds and option chains for 30 seconds.
- Entry and current stamps use New York time, whatever the server's clock says.
- Streamlit Community Cloud does not keep files between restarts: download your positions as CSV and upload them next time.
- All figures are model estimates for education and analysis. Options carry a risk of total loss of premium, and
  nothing here is investment advice.
""")