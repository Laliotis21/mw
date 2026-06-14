# Signal Playbook — how to read candles for a trade

This is the technical toolkit. The Researcher reports these values; the Analyst
scores them into a decision. Compute on the most recent completed candles for
the chosen interval (see `assets.md`). Never trade on a single indicator alone —
require **confluence** (2+ signals agreeing).

## Core indicators

| Signal | Meaning | Bullish | Bearish |
|--------|---------|---------|---------|
| **EMA(9) vs EMA(21)** | trend direction | 9 above 21 | 9 below 21 |
| **EMA cross** | trend change | 9 crosses up 21 | 9 crosses down 21 |
| **RSI(14)** | momentum / extremes | rising, >50; <30 oversold bounce | falling, <50; >70 overbought fade |
| **VWAP** | intraday fair value | price holds above | price holds below |
| **Volume** | conviction | breakout on >1.5× avg vol | move on weak vol = suspect |
| **Support / Resistance** | key levels | bounce off support | reject at resistance |
| **Candle pattern** | reversal/continuation | bullish engulfing, hammer | bearish engulfing, shooting star |

## Entry logic (confluence-based)

- **Long (BUY):** uptrend (EMA9>EMA21) **and** price above VWAP **and** a clean
  break of resistance or a pullback bounce, **confirmed by volume**.
- **Short (SELL):** downtrend (EMA9<EMA21) **and** price below VWAP **and** a
  breakdown of support or a failed bounce, **confirmed by volume**.
- **HOLD:** signals disagree, RSI in no-man's-land (45–55) with flat EMAs, or
  volume is dead. When unsure, HOLD.

## Stop & target geometry

- **Stop:** just beyond the structure that invalidates the idea — below the
  recent swing low (long) / above the swing high (short), or ~1× ATR.
- **Target:** the next resistance (long) / support (short), or a fixed
  reward:risk ≥ 1.5. Never set a target that implies < 1.5 R:R.

## Confidence scoring (0..1)

- 0.7–1.0: 3+ signals aligned, strong volume, clean level.
- 0.4–0.7: 2 signals aligned, decent setup.
- < 0.4: weak / mixed → **HOLD** (do not force a trade).
