"""
dashboard.py
============
Pro-trader control terminal for the multi-agent paper-trading bot.

A dense, dark "trading desk" layout: a live KPI strip, an equity curve, a
trade blotter, open positions, and a real-time agent feed that streams the
desk's reasoning as it works. Everything is PAPER — fills resolve against real
prices (yfinance) or land on Binance Spot Testnet / Alpaca paper. No real money
moves.

Realtime: the KPI strip, equity curve and blotter live inside st.fragment
blocks that auto-rerun every few seconds and re-read trade_log.json from disk,
so fills written by a background bot run (or by the desk here) appear without a
manual page refresh.

Run:  streamlit run dashboard.py
      (use the project venv: ./v/bin/python -m streamlit run dashboard.py)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from config import MarketPhase, current_market_phase, settings
from execution import (  # noqa: PLC2701
    TRADE_LOG,
    _load_log,
    close_all_open,
    open_positions_count,
    performance_summary,
    reconcile_open,
)

st.set_page_config(
    page_title="ALPHA DESK — Trading Terminal",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Terminal theme — dark OLED, monospace numerics, dense panels.
# --------------------------------------------------------------------------- #
GREEN, RED, AMBER, BLUE, MUTED = "#16c784", "#ea3943", "#f5a623", "#3b82f6", "#6e7d92"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg:#0a0e14; --panel:#0f141d; --panel2:#111822; --border:#1c2735;
        --text:#c9d6e3; --muted:#6e7d92; --green:#16c784; --red:#ea3943;
        --amber:#f5a623; --blue:#3b82f6;
    }
    .stApp { background: var(--bg); }
    html, body, [class*="css"] { font-family:'Fira Sans', sans-serif; color:var(--text); }
    code, .mono, [data-testid="stMetricValue"] { font-family:'Fira Code', monospace !important; }

    #MainMenu, footer, header { visibility:hidden; }
    .block-container { padding-top:1rem; padding-bottom:2rem; max-width:1500px; }

    /* Kill the "running" dim: auto-refresh fetches data each tick, and Streamlit
       fades stale elements + shows a running overlay during the rerun. Keep
       everything at full opacity and hide the overlay so live updates don't grey
       the screen. */
    [data-stale="true"], .stApp [data-stale="true"] { opacity:1 !important; }
    [data-testid="stStatusWidget"] { display:none !important; }
    .stApp [data-testid="stAppViewBlockContainer"] { opacity:1 !important; }

    .topbar {
        display:flex; align-items:center; justify-content:space-between;
        padding:.55rem .9rem; border:1px solid var(--border); border-radius:8px;
        background:linear-gradient(90deg,#0d1420,#0a0e14); margin-bottom:.8rem;
    }
    .brand { font-family:'Fira Code',monospace; font-weight:700; letter-spacing:.18em;
        font-size:1.05rem; color:#e9f0f7; }
    .brand .dot { color:var(--amber); }
    .chips { display:flex; gap:.4rem; align-items:center; }
    .chip { font-family:'Fira Code',monospace; font-size:.68rem; font-weight:600;
        padding:.2rem .55rem; border-radius:5px; border:1px solid var(--border);
        background:#0c121b; color:var(--muted); letter-spacing:.06em; }
    .chip.paper { color:var(--amber); border-color:rgba(245,166,35,.4); background:rgba(245,166,35,.08); }
    .chip.live  { color:var(--red);   border-color:rgba(234,57,67,.4);  background:rgba(234,57,67,.08); }

    /* Live pulse */
    .livedot { display:inline-block; width:8px; height:8px; border-radius:50%;
        background:var(--green); box-shadow:0 0 0 0 rgba(22,199,132,.6);
        animation:pulse 1.6s infinite; margin-right:.4rem; vertical-align:middle; }
    .livedot.off { background:var(--muted); animation:none; box-shadow:none; }
    @keyframes pulse {
        0%   { box-shadow:0 0 0 0 rgba(22,199,132,.55); }
        70%  { box-shadow:0 0 0 7px rgba(22,199,132,0); }
        100% { box-shadow:0 0 0 0 rgba(22,199,132,0); }
    }
    .chip.feed { color:var(--green); border-color:rgba(22,199,132,.35); background:rgba(22,199,132,.07); }

    .kpi-row { display:grid; grid-template-columns:repeat(6,1fr); gap:.6rem; margin-bottom:.9rem; }
    .kpi { border:1px solid var(--border); border-radius:8px; background:var(--panel); padding:.6rem .75rem; }
    .kpi .lbl { font-size:.62rem; letter-spacing:.13em; text-transform:uppercase; color:var(--muted); }
    .kpi .val { font-family:'Fira Code',monospace; font-size:1.32rem; font-weight:600; margin-top:.18rem; color:#eef4fa; }
    .kpi .sub { font-family:'Fira Code',monospace; font-size:.72rem; margin-top:.1rem; }
    .up { color:var(--green) !important; } .down { color:var(--red) !important; } .flat { color:var(--muted) !important; }

    .phead { font-family:'Fira Code',monospace; font-size:.72rem; font-weight:700;
        letter-spacing:.16em; text-transform:uppercase; color:var(--muted);
        border-bottom:1px solid var(--border); padding-bottom:.35rem; margin:.2rem 0 .7rem; }
    .phead .acc { color:var(--blue); }

    .stButton>button {
        font-family:'Fira Code',monospace !important; font-weight:700 !important;
        letter-spacing:.08em; border-radius:7px !important; border:1px solid var(--green) !important;
        background:rgba(22,199,132,.12) !important; color:var(--green) !important; transition:all .15s ease;
    }
    .stButton>button:hover { background:var(--green) !important; color:#04120c !important; box-shadow:0 0 14px rgba(22,199,132,.4); }

    [data-testid="stSegmentedControl"] button {
        border-radius:6px !important; font-family:'Fira Code',monospace !important;
        font-size:.78rem !important; border:1px solid var(--border) !important;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background:var(--blue) !important; color:#fff !important; border-color:var(--blue) !important;
    }
    [data-testid="stTextInput"] input {
        font-family:'Fira Code',monospace !important; background:#0c121b !important;
        border:1px solid var(--border) !important; color:#eef4fa !important; letter-spacing:.05em;
    }
    .stRadio [role="radiogroup"] { gap:.4rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:8px; }
    div[data-testid="stExpander"] { border:1px solid var(--border); border-radius:8px; background:var(--panel); }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def fmt_task(out) -> tuple[str, str]:
    agent = str(getattr(out, "agent", "agent"))
    raw = getattr(out, "raw", None) or getattr(out, "summary", None) or str(out)
    raw = str(raw).strip()
    return agent, (raw if len(raw) <= 1500 else raw[:1500] + " …")


NICE_NAME = {
    "Global Macro Scout": "🌍 MACRO SCOUT",
    "Equity Universe Scanner": "📈 EQUITY SCANNER",
    "Crypto Universe Scanner": "🪙 CRYPTO SCANNER",
    "Opportunity Ranker": "🎯 RANKER",
    "Lead Market Researcher (Data Scout)": "🔎 RESEARCHER",
    "Quantitative Signal Analyst": "📊 ANALYST",
    "Chief Risk Officer (Risk Manager)": "🛡️ RISK DESK",
}


def topbar_html(live: bool) -> str:
    fill = settings.FILL_SOURCE
    # Real money only when crypto routes to a NON-testnet Binance. Alpaca here is
    # always paper, and Binance testnet is fake — both read as PAPER.
    mode_live = fill in ("binance", "live") and not settings.BINANCE_TESTNET
    mode_chip = (
        '<span class="chip live">LIVE $</span>' if mode_live
        else '<span class="chip paper">PAPER</span>'
    )
    clock = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    engine_chip = (
        "ENGINE RULES" if settings.DECISION_ENGINE == "rules"
        else f"LLM {settings.LLM_PROVIDER.upper()}"
    )
    dot = "livedot" if live else "livedot off"
    feed_chip = (
        f'<span class="chip feed"><span class="{dot}"></span>LIVE</span>' if live
        else '<span class="chip"><span class="livedot off"></span>PAUSED</span>'
    )
    return f"""
    <div class="topbar">
      <div class="brand">ALPHA<span class="dot">·</span>DESK
        <span style="color:#6e7d92;font-weight:400;font-size:.7rem;letter-spacing:.1em">MULTI-AGENT TERMINAL</span></div>
      <div class="chips">
        {feed_chip}
        {mode_chip}
        <span class="chip">{engine_chip}</span>
        <span class="chip">FILL {fill.upper()}</span>
        <span class="chip">{clock}</span>
      </div>
    </div>
    """


def kpi_html(perf: dict, df: pd.DataFrame) -> str:
    ret, pnl = perf["return_pct"], perf["net_pnl"]
    ret_cls = "up" if ret > 0 else ("down" if ret < 0 else "flat")
    pnl_cls = "up" if pnl > 0 else ("down" if pnl < 0 else "flat")
    wr = perf["win_rate_pct"]
    wr_cls = "up" if wr >= 50 and perf["resolved_trades"] else ("down" if perf["resolved_trades"] else "flat")
    open_n = int((df["result"] == "open").sum()) if not df.empty and "result" in df else 0
    cells = [
        ("EQUITY", f"${perf['current_equity']:,.2f}", f"{ret:+.2f}%", ret_cls),
        ("NET P&L", f"${pnl:,.2f}", "realized", pnl_cls),
        ("WIN RATE", f"{wr:.0f}%", f"{perf['resolved_trades']} resolved", wr_cls),
        ("TRADES", f"{perf['total_trades']}", f"{open_n} open", "flat"),
        ("MAX DD", f"${perf['max_drawdown_dollars']:,.2f}", f"-{perf['max_drawdown_pct']:.1f}%",
         "down" if perf["max_drawdown_dollars"] > 0 else "flat"),
        ("RISK/TRADE", f"${settings.MAX_RISK_DOLLARS:,.0f}", f"{settings.MAX_RISK_PCT*100:.0f}% cap", "flat"),
    ]
    body = "".join(
        f'<div class="kpi"><div class="lbl">{lbl}</div>'
        f'<div class="val">{val}</div><div class="sub {cls}">{sub}</div></div>'
        for lbl, val, sub, cls in cells
    )
    return f'<div class="kpi-row">{body}</div>'


def ledger_html(meta: dict) -> str:
    """Compact API-spend ledger from the log's running token totals."""
    u = meta.get("usage")
    if not u:
        return ('<div style="font-family:Fira Code,monospace;font-size:.68rem;color:#6e7d92;'
                'margin:-.4rem 0 .8rem">API SPEND  —  no LLM calls logged yet '
                '(free local model = $0; switch LLM_PROVIDER=anthropic for real $).</div>')
    spend = u.get("cost_usd", 0.0)
    pt, ct, cached = u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("cached_tokens", 0)
    toks = pt + cached + ct
    cache_pct = (cached / (pt + cached) * 100) if (pt + cached) else 0.0
    reqs = u.get("requests", 0)
    model = (u.get("model") or "").split("/")[-1]
    paid = spend > 0
    spend_txt = f"${spend:,.4f}" if paid else "$0.00 (free local)"
    cells = [
        ("API SPEND", spend_txt),
        ("TOKENS", f"{toks:,}"),
        ("CACHED", f"{cache_pct:.0f}%"),
        ("LLM CALLS", f"{reqs:,}"),
        ("MODEL", model or "—"),
    ]
    inner = "   ·   ".join(
        f'<span style="color:#6e7d92">{k}</span> '
        f'<span style="color:{"#f5a623" if paid and k=="API SPEND" else "#c9d6e3"}">{v}</span>'
        for k, v in cells
    )
    return (f'<div style="font-family:Fira Code,monospace;font-size:.7rem;'
            f'border:1px solid #1c2735;border-radius:6px;padding:.4rem .7rem;'
            f'margin:-.3rem 0 .9rem;background:#0c121b">⛁ {inner}</div>')


def _toast_closed(closed: list[dict]) -> None:
    """Pop a toast for each broker position that just settled (TP/SL)."""
    for c in closed or []:
        pnl = c.get("pnl", 0.0)
        icon = "✅" if c.get("result") == "take_profit" else ("❌" if c.get("result") == "stop_loss" else "➖")
        st.toast(f"{icon} {c.get('asset')} closed — {c.get('result')} · P&L ${pnl:+,.2f}", icon=icon)


def last_run_html(meta: dict) -> str:
    """One-line summary of the most recent run: when, phase, what it did, cost."""
    r = meta.get("last_run")
    if not r:
        return ('<div style="font-family:Fira Code,monospace;font-size:.68rem;color:#6e7d92;'
                'margin:-.4rem 0 .9rem">🗒 LAST RUN  —  none yet. Press EXECUTE.</div>')
    try:
        when = pd.to_datetime(r.get("timestamp_utc"), utc=True).strftime("%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001
        when = r.get("timestamp_utc", "—")
    pnl = r.get("pnl", 0.0)
    pcls = "up" if pnl > 0 else ("down" if pnl < 0 else "flat")
    assets = ", ".join(a for a in r.get("assets", []) if a) or "—"
    parts = [
        ("WHEN", when, ""),
        ("PHASE", str(r.get("phase", "")).upper(), ""),
        ("MODE", r.get("mode", ""), ""),
        ("PROCESSED", f"{r.get('processed',0)} ({r.get('traded',0)} trade / {r.get('held',0)} hold)", ""),
        ("RUN P&L", f"${pnl:+,.2f}", pcls),
        ("RUN COST", f"${r.get('cost_usd',0.0):,.4f}", ""),
    ]
    inner = "   ·   ".join(
        f'<span style="color:#6e7d92">{k}</span> '
        f'<span class="{cls}" style="color:#c9d6e3">{v}</span>' for k, v, cls in parts
    )
    return (f'<div style="font-family:Fira Code,monospace;font-size:.7rem;'
            f'border:1px solid #1c2735;border-radius:6px;padding:.4rem .7rem;'
            f'margin:-.3rem 0 .9rem;background:#0c121b">🗒 {inner}'
            f'<span style="color:#6e7d92">   ·   {assets}</span></div>')


def autorun_html(meta: dict) -> str:
    """Live status of the autonomous scheduler: cycle, last/next, last action."""
    a = meta.get("autorun")
    if not a:
        return ('<div style="font-family:Fira Code,monospace;font-size:.68rem;color:#6e7d92;'
                'margin:-.4rem 0 .9rem">🤖 AUTORUN  —  off. Start <code>scheduler.py</code> '
                'for hands-free trading.</div>')
    interval = a.get("interval_min", 30)
    try:
        last = pd.to_datetime(a.get("last_run_utc"), utc=True)
        age_min = (datetime.now(timezone.utc) - last.to_pydatetime()).total_seconds() / 60
        last_s = last.strftime("%H:%M")
    except Exception:  # noqa: BLE001
        age_min, last_s = 9999, "—"
    try:
        nxt_s = pd.to_datetime(a.get("next_run_utc"), utc=True).strftime("%H:%M")
    except Exception:  # noqa: BLE001
        nxt_s = "—"
    stale = age_min > interval * 2.5
    dot = '<span class="livedot off"></span>' if stale else '<span class="livedot"></span>'
    head = "AUTORUN IDLE (scheduler stopped?)" if stale else "AUTORUN LIVE"
    color = "#6e7d92" if stale else "#16c784"
    body = (f'cycle #{a.get("cycle",0)} · every {interval}m · last {last_s} UTC · '
            f'next ~{nxt_s} · {a.get("traded",0)} trade/{a.get("held",0)} hold · '
            f'{a.get("note","")} · eq ${a.get("equity",0):,.2f}')
    return (f'<div style="font-family:Fira Code,monospace;font-size:.7rem;'
            f'border:1px solid #1c2735;border-radius:6px;padding:.4rem .7rem;'
            f'margin:-.3rem 0 .9rem;background:#0c121b">{dot}'
            f'<span style="color:{color};font-weight:600">🤖 {head}</span> '
            f'<span style="color:#6e7d92">  {body}</span></div>')


def _show_trade_result(asset: str, ticket, record) -> None:
    action = ticket.action.value
    if action == "HOLD" or record["quantity"] == 0:
        st.info(f"🤚 **HOLD {asset}** — desk stood down, no clear edge.")
        return
    if record["result"] == "open":
        st.warning(
            f"📤 **{action} {record['quantity']} {asset}** @ ${record['entry_price']} "
            f"— bracket LIVE on broker ({record.get('fill_source','')}). Outcome pending."
        )
        return
    verdict = {"take_profit": "✅ TP HIT", "stop_loss": "❌ STOP HIT", "markout": "➖ MARKED OUT"}.get(
        record["result"], record["result"])
    pnl = record["pnl"]
    box = st.success if pnl > 0 else (st.error if pnl < 0 else st.info)
    box(f"**{action} {record['quantity']} {asset}** @ ${record['entry_price']} → {verdict} · "
        f"P&L **${pnl:,.2f}** (real prices)")


def render_equity_curve(df: pd.DataFrame, start: float) -> None:
    if df.empty:
        st.caption("No trades yet — the curve plots once the desk executes.")
        return
    curve = df[["equity_after"]].copy()
    curve.index = range(1, len(curve) + 1)
    curve = pd.concat([pd.DataFrame({"equity_after": [start]}, index=[0]), curve])
    curve = curve.reset_index().rename(columns={"index": "trade", "equity_after": "equity"})
    line = GREEN if curve["equity"].iloc[-1] >= start else RED
    try:
        import altair as alt
        base = alt.Chart(curve)
        area = base.mark_area(
            line={"color": line, "strokeWidth": 2},
            color=alt.Gradient(gradient="linear", stops=[
                alt.GradientStop(color=line, offset=1),
                alt.GradientStop(color="#0a0e14", offset=0)], x1=1, x2=1, y1=1, y2=0),
            opacity=0.18,
        ).encode(
            x=alt.X("trade:Q", title=None, axis=alt.Axis(grid=False, labelColor=MUTED, tickColor=MUTED, domainColor="#1c2735")),
            y=alt.Y("equity:Q", title=None, scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridColor="#141c27", labelColor=MUTED, tickColor=MUTED, domainColor="#1c2735", format="$,.0f")),
            tooltip=[alt.Tooltip("trade:Q", title="Trade #"), alt.Tooltip("equity:Q", title="Equity", format="$,.2f")],
        )
        rule = alt.Chart(pd.DataFrame({"y": [start]})).mark_rule(
            color=MUTED, strokeDash=[4, 4], opacity=0.6).encode(y="y:Q")
        chart = (area + rule).properties(height=230).configure_view(strokeWidth=0, fill="#0a0e14")
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        st.line_chart(curve.set_index("trade")["equity"], height=230)


def _fmt_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a compact 'Time' column (UTC) parsed from timestamp_utc."""
    df = df.copy()
    if "timestamp_utc" in df:
        ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df["Time"] = ts.dt.strftime("%m-%d %H:%M:%S").fillna("—")
    return df


def _live_prices(symbols: list[str]) -> dict:
    """
    Latest price per symbol via yfinance, cached ~10s in session_state. Without
    the cache the auto-refresh fragment fetches every 2-3s tick → the whole app
    sits under Streamlit's 'running' dim almost constantly. Caching makes most
    ticks instant.
    """
    import time as _t
    want = set(symbols)
    cache = st.session_state.get("_px_cache", {"ts": 0.0, "data": {}})
    if _t.time() - cache["ts"] < 10 and want <= set(cache["data"]):
        return cache["data"]
    out: dict[str, float] = {}
    try:
        import yfinance as yf
        for s in want:
            try:
                fi = yf.Ticker(s).fast_info
                p = fi.get("lastPrice") or fi.get("last_price")
                if p:
                    out[s] = float(p)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    st.session_state["_px_cache"] = {"ts": _t.time(), "data": out}
    return out


def render_open_positions(df: pd.DataFrame) -> None:
    if df.empty or "result" not in df or not (df["result"] == "open").any():
        return
    st.markdown('<div class="phead"><span class="acc">▎</span>OPEN POSITIONS · live</div>', unsafe_allow_html=True)
    op = _fmt_time(df[df["result"] == "open"]).copy()
    prices = _live_prices(op["asset"].tolist())

    def _live(r):
        return prices.get(r["asset"])

    def _unreal(r):
        live = prices.get(r["asset"])
        if live is None:
            return None
        edge = (live - r["entry_price"]) if r["action"] == "BUY" else (r["entry_price"] - live)
        return round(edge * r["quantity"], 2)

    def _to(r, level):  # % from live price to a stop/target level
        live = prices.get(r["asset"])
        return round((level - live) / live * 100, 1) if live else None

    op["Live"] = op.apply(_live, axis=1)
    op["Unreal $"] = op.apply(_unreal, axis=1)
    op["→Stop%"] = op.apply(lambda r: _to(r, r["stop_loss"]), axis=1)
    op["→Tgt%"] = op.apply(lambda r: _to(r, r["take_profit"]), axis=1)
    op = op.rename(columns={"asset": "Ticker", "action": "Side", "entry_price": "Entry",
                            "stop_loss": "Stop", "take_profit": "Target", "quantity": "Qty",
                            "fill_source": "Venue"})
    cols = [c for c in ["Time", "Ticker", "Side", "Qty", "Entry", "Live", "Unreal $",
                        "Stop", "→Stop%", "Target", "→Tgt%", "Venue"] if c in op]
    view = op[cols].iloc[::-1]

    def _pcolor(v):
        if pd.isna(v):
            return ""
        return f"color:{GREEN}" if v > 0 else (f"color:{RED}" if v < 0 else f"color:{MUTED}")

    styler = view.style.map(_pcolor, subset=["Unreal $"]) if "Unreal $" in view else view.style
    fmts = {c: "${:,.2f}" for c in ("Entry", "Live", "Stop", "Target") if c in view}
    if "Unreal $" in view:
        fmts["Unreal $"] = "${:+,.2f}"
    for c in ("→Stop%", "→Tgt%"):
        if c in view:
            fmts[c] = "{:+.1f}%"
    st.dataframe(styler.format(fmts, na_rep="—"), width="stretch", hide_index=True)

    if st.button("✖ Close ALL positions (market)", key="close_all_btn"):
        closed = close_all_open()
        _toast_closed(closed)
        st.toast(f"Closed {len(closed)} position(s).")
        st.rerun()


def render_blotter(df: pd.DataFrame) -> None:
    st.markdown('<div class="phead"><span class="acc">▎</span>TRADE BLOTTER</div>', unsafe_allow_html=True)
    if df.empty:
        st.caption("Blotter empty.")
        return
    nice = _fmt_time(df).rename(columns={
        "asset": "Ticker", "action": "Side", "entry_price": "Entry", "exit_price": "Exit",
        "result": "Result", "pnl": "P&L", "equity_after": "Equity", "llm_cost_usd": "LLM $"})
    cols = [c for c in ["Time", "Ticker", "Side", "Entry", "Exit", "Result", "P&L", "Equity", "LLM $"] if c in nice]
    blotter = nice[cols].iloc[::-1].head(25)

    def _pnl_color(v):
        if pd.isna(v):
            return ""
        return f"color:{GREEN}" if v > 0 else (f"color:{RED}" if v < 0 else f"color:{MUTED}")

    styler = blotter.style.map(_pnl_color, subset=["P&L"]) if "P&L" in blotter else blotter.style
    fmts = {c: "${:,.2f}" for c in ("Entry", "Exit", "Equity") if c in blotter}
    if "P&L" in blotter:
        fmts["P&L"] = "${:+,.2f}"
    if "LLM $" in blotter:
        fmts["LLM $"] = "${:,.4f}"
    st.dataframe(styler.format(fmts, na_rep="—"), width="stretch", hide_index=True, height=360)


# --------------------------------------------------------------------------- #
# Realtime controls — drive the fragments' auto-rerun interval.
# --------------------------------------------------------------------------- #
cc = st.columns([5, 1.1, 1.4])
with cc[1]:
    live = st.toggle("🔴 Live", value=True, help="Auto-refresh KPIs, curve & blotter from disk.")
with cc[2]:
    interval = st.selectbox("Refresh", [2, 3, 5, 10], index=1,
                            format_func=lambda s: f"{s}s", label_visibility="collapsed",
                            help="Auto-refresh interval")
every = float(interval) if live else None


# --------------------------------------------------------------------------- #
# Fragment 1: top status bar + KPI strip (full width, auto-rerun).
# --------------------------------------------------------------------------- #
@st.fragment(run_every=every)
def header_kpi() -> None:
    log = _load_log()
    df = pd.DataFrame(log["trades"]) if log["trades"] else pd.DataFrame()
    st.markdown(topbar_html(live), unsafe_allow_html=True)
    st.markdown(kpi_html(performance_summary(), df), unsafe_allow_html=True)
    st.markdown(ledger_html(log["meta"]), unsafe_allow_html=True)
    st.markdown(autorun_html(log["meta"]), unsafe_allow_html=True)
    st.markdown(last_run_html(log["meta"]), unsafe_allow_html=True)


header_kpi()

# --------------------------------------------------------------------------- #
# Desk: order desk + agent feed (left, interactive) | live market (right).
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1.55], gap="medium")

with left:
    st.markdown('<div class="phead"><span class="acc">▎</span>ORDER DESK</div>', unsafe_allow_html=True)
    mode = st.radio("Selection", ["🔍 Auto-discover movers", "✍️ Pick a ticker"],
                    horizontal=True, label_visibility="collapsed")
    discover = mode.startswith("🔍")
    if discover:
        st.caption(f"4 scouts scan live stock + crypto movers & macro, then the desk trades the top {settings.MAX_CANDIDATES}.")
        asset = ""
    else:
        asset = st.text_input("Ticker", value="AAPL", label_visibility="collapsed",
                              placeholder="AAPL / BTC-USD").strip().upper()
    # Phase auto-detected from the live US-market clock — the user shouldn't
    # have to choose it. A small override is available for manual testing.
    PHASE_LABELS = {"pre_market": "🌅 PRE-MARKET", "open": "🔔 OPEN", "mid_day": "☀️ MID-DAY", "close": "🌆 CLOSE"}
    detected = current_market_phase().value
    phase = detected
    st.markdown(
        f'<div style="margin-top:.5rem;font-family:Fira Code,monospace;font-size:.72rem;'
        f'border:1px solid #1c2735;border-radius:6px;padding:.35rem .6rem;background:#0c121b">'
        f'<span style="color:#6e7d92">SESSION (auto)</span> '
        f'<span style="color:#3b82f6;font-weight:600">{PHASE_LABELS.get(detected, detected)}</span> '
        f'<span style="color:#6e7d92">· from US market clock</span></div>',
        unsafe_allow_html=True,
    )
    if st.checkbox("override phase", value=False):
        _picked = st.segmented_control("Session", options=[p.value for p in MarketPhase],
                                       format_func=lambda v: PHASE_LABELS.get(v, v),
                                       default=detected, label_visibility="collapsed")
        phase = _picked or detected
    btn_label = "▶  SCOUT & TRADE" if discover else "▶  EXECUTE CYCLE"
    run = st.button(btn_label, type="primary", width="stretch")

    st.markdown('<div class="phead" style="margin-top:1rem"><span class="acc">▎</span>AGENT FEED</div>', unsafe_allow_html=True)
    feed = st.container(height=420, border=False)
    if not run:
        feed.caption("Idle. Press EXECUTE to stream the desk's reasoning here in real time.")


def feed_task(out) -> None:
    role, text = fmt_task(out)
    with feed:
        st.markdown(f"**{NICE_NAME.get(role, role)}**")
        st.code(text, language="json")


with right:
    @st.fragment(run_every=every)
    def market_panel() -> None:
        # Auto-notify on close: poll the broker ONLY while positions are open and
        # at most every 15s (decoupled from the faster view refresh) — settled
        # TP/SL pops a toast without hammering the broker API.
        if settings.FILL_SOURCE in ("binance", "alpaca", "live") and open_positions_count():
            import time as _t
            if _t.time() - st.session_state.get("_last_reconcile", 0.0) >= 15:
                st.session_state["_last_reconcile"] = _t.time()
                _toast_closed(reconcile_open())
        log = _load_log()
        df = pd.DataFrame(log["trades"]) if log["trades"] else pd.DataFrame()
        start = log["meta"]["starting_capital"]
        st.markdown('<div class="phead"><span class="acc">▎</span>EQUITY CURVE</div>', unsafe_allow_html=True)
        render_equity_curve(df, start)
        render_open_positions(df)
        render_blotter(df)

    market_panel()

# --------------------------------------------------------------------------- #
# Run a cycle — stream agents into the left feed, results below the curve.
# Fragments pick up the new fills on their next tick (no manual refresh).
# --------------------------------------------------------------------------- #
if run and (discover or asset):
    # LLM + fill source come from .env. Research stays free/local (candles) so no
    # Perplexity key is needed; FILL_SOURCE decides sim (yfinance) vs real broker
    # paper orders (alpaca / binance / live).
    settings.RESEARCH_SOURCE = "candles"

    if (settings.DECISION_ENGINE == "llm" and settings.LLM_PROVIDER == "anthropic"
            and not settings.ANTHROPIC_API_KEY):
        st.error("DECISION_ENGINE=llm + provider anthropic but ANTHROPIC_API_KEY is empty. "
                 "Add your key to .env or set DECISION_ENGINE=rules.")
        st.stop()

    from execution import execute_ticket, log_usage, reconcile_open, set_last_run
    from main import pop_last_usage, run_cycle, run_discovery

    # Settle any broker brackets that closed since last run before placing new ones.
    if settings.FILL_SOURCE in ("binance", "alpaca", "live"):
        _toast_closed(reconcile_open())

    results = right.container()
    cost0 = _load_log()["meta"].get("usage", {}).get("cost_usd", 0.0)
    run_records: list[dict] = []
    try:
        if discover:
            with feed:
                with st.status("🛰️ Scouts scanning the tape…", expanded=True):
                    shortlist = run_discovery(phase, task_callback=feed_task)
            log_usage(pop_last_usage())  # log scout-crew tokens to the ledger
            if shortlist is None or not shortlist.ideas:
                results.error("Scouts found no tradable ideas right now. Try another session.")
            else:
                results.markdown(
                    f'<div class="phead" style="margin-top:1rem"><span class="acc">▎</span>SHORTLIST · '
                    f'macro {shortlist.macro_bias.value} ({shortlist.macro_score:+.2f})</div>',
                    unsafe_allow_html=True)
                results.dataframe(pd.DataFrame([
                    {"Ticker": i.asset, "Class": i.asset_class.value, "Score": i.raw_score,
                     "Move %": i.change_pct, "Why": i.reason} for i in shortlist.ideas]),
                    width="stretch", hide_index=True)
                for n, idea in enumerate(shortlist.ideas[: settings.MAX_CANDIDATES], 1):
                    ideas_n = min(len(shortlist.ideas), settings.MAX_CANDIDATES)
                    with feed:
                        with st.status(f"📊 Desk trading {idea.asset} ({n}/{ideas_n})…", expanded=True):
                            ticket = run_cycle(idea.asset, phase, task_callback=feed_task)
                    cycle_usage = pop_last_usage()
                    if ticket is None:
                        log_usage(cycle_usage)
                        results.warning(f"No valid decision for {idea.asset} — skipped.")
                        continue
                    record = execute_ticket(ticket, market_phase=phase, usage=cycle_usage)
                    run_records.append(record)
                    with results:
                        _show_trade_result(idea.asset, ticket, record)
        else:
            with feed:
                with st.status(f"📊 Desk trading {asset}…", expanded=True):
                    ticket = run_cycle(asset, phase, task_callback=feed_task)
            cycle_usage = pop_last_usage()
            if ticket is None:
                log_usage(cycle_usage)
                results.error("The desk could not produce a valid decision. Try again.")
            else:
                record = execute_ticket(ticket, market_phase=phase, usage=cycle_usage)
                run_records.append(record)
                with results:
                    _show_trade_result(asset, ticket, record)
    except Exception as exc:  # noqa: BLE001
        results.error(f"Run failed: {exc}")
    finally:
        # Persist a one-line summary of this run for the LAST RUN panel.
        cost1 = _load_log()["meta"].get("usage", {}).get("cost_usd", 0.0)
        traded = [r for r in run_records if r.get("result") not in ("no_trade", None)]
        held = [r for r in run_records if r.get("result") == "no_trade"]
        set_last_run({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "mode": "discover" if discover else "single",
            "processed": len(run_records),
            "traded": len(traded),
            "held": len(held),
            "pnl": round(sum(r.get("pnl", 0.0) for r in run_records), 2),
            "cost_usd": round(cost1 - cost0, 6),
            "assets": [r.get("asset") for r in run_records],
        })
elif run and not discover and not asset:
    st.warning("Type a ticker first (e.g. AAPL or BTC-USD).")

# --------------------------------------------------------------------------- #
# Reset, tucked at the bottom.
# --------------------------------------------------------------------------- #
with st.expander("⚙ Desk controls"):
    st.caption(f"Risk cap ${settings.MAX_RISK_DOLLARS:,.0f}/trade · daily stop {settings.DAILY_MAX_LOSS_PCT*100:.0f}% · starting capital ${settings.STARTING_CAPITAL:,.0f} · fills via {settings.FILL_SOURCE.upper()}")
    if settings.FILL_SOURCE in ("binance", "alpaca", "live"):
        if st.button("🔄 Reconcile open positions (poll broker, settle TP/SL)"):
            closed = reconcile_open()
            _toast_closed(closed)
            st.success(f"Settled {len(closed)} closed position(s)." if closed else "No positions resolved yet.")
            st.rerun()
        # Manual close: pick an open broker position and flatten it at market.
        opens = [t for t in _load_log()["trades"]
                 if t.get("result") == "open" and t.get("alpaca_order_id")]
        if opens:
            labels = {
                f"{t['asset']} {t['action']} x{t['quantity']} @ ${t['entry_price']} "
                f"[{(t.get('alpaca_order_id') or '')[:8]}]": t["alpaca_order_id"]
                for t in opens
            }
            pick = st.selectbox("Open position to close", list(labels.keys()),
                                label_visibility="collapsed")
            if st.button("✖ Close selected position now (market)"):
                from execution import close_open
                res = close_open(labels[pick])
                if res:
                    st.success(f"Closed {res['asset']} — {res['result']} · P&L ${res['pnl']:+,.2f}")
                    st.rerun()
                else:
                    st.error("Close failed — check the venue/logs.")
    if st.button("Flatten book — clear all trades & reset balance"):
        if TRADE_LOG.exists():
            TRADE_LOG.unlink()
        st.rerun()
