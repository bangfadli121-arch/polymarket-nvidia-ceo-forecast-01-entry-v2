"""Forecast minute-by-minute Yes prices for NVIDIA CEO markets using volatility-scaled normal quantiles and exclusive-outcome normalization."""

import json
import math
import os


Z_VALUES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)


def market_sigma(prices, x):
    """Compute clamped volatility from the last 360 price changes and checkpoint price."""
    history = prices[-361:]
    if len(prices) < 2 or len(history) < 2:
        mean_abs = 0.0
    else:
        mean_abs = sum(abs(history[i + 1] - history[i]) for i in range(len(history) - 1)) / (len(history) - 1)
    sigma = mean_abs + 0.002 * math.sqrt(x * (1.0 - x))
    return min(0.005, max(0.0005, sigma))


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(name for name in os.listdir(input_dir) if name.endswith(".in")):
        input_path = os.path.join(input_dir, filename)
        with open(input_path, "r", encoding="utf-8") as source:
            case = json.load(source)

        markets = case["markets"]
        last_prices = [market["prices"][-1] for market in markets]
        total_price = sum(last_prices)
        # A zero total has no meaningful normalisation, so preserve the scale.
        scale = 1.0 if total_price == 0 else 1.0 / total_price
        forecasts = []

        for market, x in zip(markets, last_prices):
            prices = market["prices"]
            sigma = market_sigma(prices, x)
            rows = []
            for h in range(1, case["horizon_minutes"] + 1):
                spread = sigma * math.sqrt(h)
                row = []
                for z in Z_VALUES:
                    value = min(1.0, max(0.0, (x + z * spread) * scale))
                    value = round(value, 6)
                    row.append(0.0 if value == 0 else value)
                rows.append(row)
            forecasts.append({"id": market["id"], "q": rows})

        output = {"forecasts": forecasts}
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".out")
        with open(output_path, "w", encoding="utf-8") as destination:
            json.dump(output, destination, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
