"""
dashboard.py
============
Simple paper-trading control panel.

Plain and minimal on purpose: one screen, one button. No real money is ever
involved — the bot decides with AI agents, then the trade is tested against REAL
market prices (via yfinance) to see if it would have won or lost.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import MarketPhase, settings
from execution import TRADE_LOG, performance_summary, _load_log  # noqa: PLC2701

st.set_page_config(page_title="Trading Bot — Paper Mode", page_icon="📈", layout="centered")

# A little polish for the phase pills.
st.markdown(
    """
    <style>
    [data-testid="stSegmentedControl"] { gap: .35rem; }
    [data-testid="stSegmentedControl"] button {
        border-radius: 999px !important;
        padding: .35rem 1rem !important;
        font-weight: 600;
        border: 1px solid rgba(250,250,250,.12) !important;
        transition: all .15s ease;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"],
    [data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
        background: #ef4444 !important;
        color: #fff !important;
        border-color: #ef4444 !important;
    }
    [data-testid="stSegmentedControl"] button:hover {
        border-color: #ef4444 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def trades_df() -> pd.DataFrame:
    log = _load_log()
    if not log["trades"]:
        return pd.DataFrame()
    return pd.DataFrame(log["trades"])


def fmt_task(out) -> tuple[str, str]:
    """(agent role, its decision text) from a CrewAI task output."""
    agent = str(getattr(out, "agent", "agent"))
    raw = getattr(out, "raw", None) or getattr(out, "summary", None) or str(out)
    raw = str(raw).strip()
    return agent, (raw if len(raw) <= 1500 else raw[:1500] + " …")


# Friendly names for every agent — discovery scouts + decision desk.
NICE_NAME = {
    # Discovery crew
    "Global Macro Scout": "🌍 Macro Scout — read the world",
    "Equity Universe Scanner": "📈 Stock Scanner — found movers",
    "Crypto Universe Scanner": "🪙 Crypto Scanner — found movers",
    "Opportunity Ranker": "🎯 Ranker — picked the shortlist",
    # Decision desk
    "Lead Market Researcher (Data Scout)": "🔎 Researcher — read the market",
    "Quantitative Signal Analyst": "📊 Analyst — picked a direction",
    "Chief Risk Officer (Risk Manager)": "🛡️ Risk Manager — sized the trade",
}


def _show_trade_result(asset: str, ticket, record) -> None:
    """Render one executed ticket as a friendly success/error/info box."""
    action = ticket.action.value
    if action == "HOLD" or record["quantity"] == 0:
        st.info(f"🤚 Agents decided to **HOLD {asset}** — no trade taken (no clear edge).")
        return
    verdict = {
        "take_profit": "✅ hit the profit target",
        "stop_loss": "❌ hit the stop loss",
        "markout": "➖ closed flat at end of window",
    }.get(record["result"], record["result"])
    pnl = record["pnl"]
    box = st.success if pnl > 0 else (st.error if pnl < 0 else st.info)
    box(
        f"**{action} {record['quantity']} {asset}** at ${record['entry_price']} "
        f"→ {verdict}. Profit/loss: **${pnl:,.2f}** (tested on real prices)."
    )


# --------------------------------------------------------------------------- #
# Header + balance
# --------------------------------------------------------------------------- #
st.title("📈 Trading Bot")
st.caption("Paper mode — fake money, real prices. Nothing is actually bought or sold.")

perf = performance_summary()
c1, c2, c3 = st.columns(3)
c1.metric("💰 Balance", f"${perf['current_equity']:,.2f}", f"{perf['return_pct']:+.2f}%")
c2.metric("📈 Total profit/loss", f"${perf['net_pnl']:,.2f}")
c3.metric("🔁 Trades made", perf["total_trades"])

st.divider()

# --------------------------------------------------------------------------- #
# The action: either auto-discover movers, or pick one ticker — then trade.
# --------------------------------------------------------------------------- #
st.subheader("Run a trade")

mode = st.radio(
    "How should we pick what to trade?",
    ["🔍 Auto-discover movers", "✍️ Pick a ticker"],
    horizontal=True,
)
discover = mode.startswith("🔍")

if discover:
    st.caption(
        f"4 scout agents scan live stock + crypto movers and the macro backdrop, "
        f"then the desk trades the top {settings.MAX_CANDIDATES}."
    )
    asset = ""
else:
    asset = st.text_input("Stock / crypto ticker", value="AAPL").strip().upper()

# Market-phase picker — pill segmented control instead of a cramped dropdown.
PHASE_LABELS = {
    "pre_market": "🌅 Pre-market",
    "open": "🔔 Open",
    "mid_day": "☀️ Mid-day",
    "close": "🌆 Close",
}
_phase_values = [p.value for p in MarketPhase]
st.markdown("**🕑 Time of day**")
_picked = st.segmented_control(
    "Time of day",
    options=_phase_values,
    format_func=lambda v: PHASE_LABELS.get(v, v),
    default="open",
    label_visibility="collapsed",
)
phase = _picked or "open"

btn_label = "🛰️  Scout the market & trade" if discover else "▶️  Let the AI agents trade"
run = st.button(btn_label, type="primary", width="stretch")

# Live area for this run.
feed = st.container()


def feed_task(out) -> None:
    """Stream one agent's output into the live feed."""
    role, text = fmt_task(out)
    with feed:
        st.markdown(f"**{NICE_NAME.get(role, role)}**")
        st.code(text, language="json")


if run and (discover or asset):
    # Force the free local setup (no paid keys, real prices for the result).
    settings.LLM_PROVIDER = "ollama"
    settings.RESEARCH_SOURCE = "candles"
    settings.FILL_SOURCE = "yfinance"

    from execution import execute_ticket
    from main import run_cycle, run_discovery

    try:
        if discover:
            # 1) Discovery crew → ranked shortlist.
            with st.status("🛰️ Scouts scanning the market…", expanded=True):
                shortlist = run_discovery(phase, task_callback=feed_task)

            if shortlist is None or not shortlist.ideas:
                st.error("Scouts found no tradable ideas right now. Try another phase.")
            else:
                st.markdown(
                    f"**🎯 Shortlist** — macro tilt: `{shortlist.macro_bias.value}` "
                    f"({shortlist.macro_score:+.2f}). "
                    f"Themes: {', '.join(shortlist.themes) or '—'}"
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Ticker": i.asset,
                                "Class": i.asset_class.value,
                                "Score": i.raw_score,
                                "Move %": i.change_pct,
                                "Why": i.reason,
                            }
                            for i in shortlist.ideas
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

                # 2) Trade each shortlisted name through the desk.
                ideas = shortlist.ideas[: settings.MAX_CANDIDATES]
                for n, idea in enumerate(ideas, 1):
                    with st.status(f"📊 Desk trading {idea.asset} ({n}/{len(ideas)})…", expanded=True):
                        ticket = run_cycle(idea.asset, phase, task_callback=feed_task)
                    if ticket is None:
                        st.warning(f"No valid decision for {idea.asset} — skipped.")
                        continue
                    record = execute_ticket(ticket, market_phase=phase)
                    _show_trade_result(idea.asset, ticket, record)
        else:
            # Single hand-picked ticker.
            with st.status(f"📊 Agents trading {asset}…", expanded=True):
                ticket = run_cycle(asset, phase, task_callback=feed_task)
            if ticket is None:
                st.error("The agents could not produce a valid decision. Try again.")
            else:
                record = execute_ticket(ticket, market_phase=phase)
                _show_trade_result(asset, ticket, record)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
elif run and not discover and not asset:
    st.warning("Type a ticker first (e.g. AAPL).")

st.divider()

# --------------------------------------------------------------------------- #
# Balance over time + recent trades
# --------------------------------------------------------------------------- #
df = trades_df()
if df.empty:
    st.info("No trades yet. Pick a ticker above and press the button.")
else:
    st.subheader("Balance over time")
    curve = df[["equity_after"]].copy()
    curve.index = range(1, len(curve) + 1)
    start = _load_log()["meta"]["starting_capital"]
    curve = pd.concat([pd.DataFrame({"equity_after": [start]}, index=[0]), curve])
    st.line_chart(curve, height=240)

    st.subheader("Recent trades")
    nice = df.copy()
    nice = nice.rename(
        columns={
            "asset": "Ticker",
            "action": "Action",
            "entry_price": "Entry",
            "result": "Result",
            "pnl": "P/L ($)",
            "equity_after": "Balance",
        }
    )
    cols = [c for c in ["Ticker", "Action", "Entry", "Result", "P/L ($)", "Balance"] if c in nice]
    st.dataframe(
        nice[cols].iloc[::-1].head(15),
        width="stretch",
        hide_index=True,
    )

# --------------------------------------------------------------------------- #
# Small reset, tucked at the bottom.
# --------------------------------------------------------------------------- #
with st.expander("Reset"):
    if st.button("Clear all trades & reset balance to $1,000"):
        if TRADE_LOG.exists():
            TRADE_LOG.unlink()
        st.rerun()
