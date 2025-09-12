from __future__ import annotations

from pathlib import Path

from ..backtest.runner import BestResult


class TradingViewExporter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(self, results: list[BestResult]):
        lines: list[str] = []
        lines.append("# TradingView Export\n")
        lines.append(
            "This file contains Pine v5 snippets or alert templates for best combinations.\n"
        )

        for r in results:
            lines.append(f"## {r.collection} / {r.symbol} / {r.timeframe} / {r.strategy}")
            lines.append(f"- Params: {r.params}")
            # We cannot import the strategy class here easily; include a placeholder.
            # If strategy implements to_tradingview_pine, users can copy the snippet.
            lines.append("- Pine Snippet:")
            lines.append("```")
            lines.append(
                "// Add your Pine code here or implement to_tradingview_pine in the strategy class."
            )
            lines.append("```")
            lines.append("")

        path = self.out_dir / "tradingview.md"
        path.write_text("\n".join(lines))
