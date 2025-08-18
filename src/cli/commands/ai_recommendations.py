"""CLI commands for AI Investment Recommendations."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ...ai.investment_recommendations import AIInvestmentRecommendations
from ...database import get_db_session


@click.group()
def ai():
    """AI-powered investment recommendations."""


@ai.command()
@click.option(
    "--risk-tolerance",
    "-r",
    default="moderate",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    help="Risk tolerance level",
)
@click.option(
    "--max-assets", "-n", default=10, help="Maximum number of assets to recommend"
)
@click.option("--min-confidence", "-c", default=0.7, help="Minimum confidence score")
@click.option("--quarter", "-q", help="Specific quarter to analyze (e.g., Q3_2025)")
@click.option("--output", "-o", help="Output file path for recommendations")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json", "summary"]),
    help="Output format",
)
def recommend(
    risk_tolerance, max_assets, min_confidence, quarter, output, output_format
):
    """Generate AI-powered investment recommendations."""
    click.echo(f"🤖 Generating AI recommendations for {risk_tolerance} risk profile...")

    try:
        # Initialize recommender
        session = get_db_session()
        recommender = AIInvestmentRecommendations(session)

        # Generate recommendations
        portfolio_rec = recommender.generate_recommendations(
            risk_tolerance=risk_tolerance,
            min_confidence=min_confidence,
            max_assets=max_assets,
            quarter=quarter,
        )

        # Display results
        if output_format == "table":
            _display_table(portfolio_rec)
        elif output_format == "json":
            _display_json(portfolio_rec)
        else:
            _display_summary(portfolio_rec)

        # Save to file if requested
        if output:
            _save_recommendations(portfolio_rec, output, output_format)
            click.echo(f"💾 Recommendations saved to {output}")

        session.close()

    except Exception as e:
        click.echo(f"❌ Error generating recommendations: {e}", err=True)
        raise click.Abort()


@ai.command()
@click.option(
    "--portfolio", "-p", required=True, help="Portfolio configuration file path"
)
@click.option(
    "--risk-tolerance",
    "-r",
    default="moderate",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    help="Risk tolerance level",
)
@click.option(
    "--max-assets", "-n", default=10, help="Maximum number of assets to recommend"
)
@click.option("--min-confidence", "-c", default=0.6, help="Minimum confidence score")
@click.option("--quarter", "-q", help="Specific quarter to analyze (e.g., Q3_2025)")
@click.option("--no-html", is_flag=True, help="Skip HTML report generation")
def portfolio_recommend(
    portfolio, risk_tolerance, max_assets, min_confidence, quarter, no_html
):
    """Generate AI recommendations for a specific portfolio with HTML report."""
    click.echo(f"🤖 Generating AI recommendations for portfolio: {portfolio}")

    try:
        # Initialize recommender
        session = get_db_session()
        recommender = AIInvestmentRecommendations(session)

        # Generate portfolio-specific recommendations
        portfolio_rec, html_path = recommender.generate_portfolio_recommendations(
            portfolio_config_path=portfolio,
            risk_tolerance=risk_tolerance,
            min_confidence=min_confidence,
            max_assets=max_assets,
            quarter=quarter,
            generate_html=not no_html,
        )

        # Display results
        _display_table(portfolio_rec)

        if html_path:
            click.echo(f"\n📄 HTML report generated: {html_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.ClickException(str(e))
    finally:
        session.close()


@ai.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--strategy", "-s", help="Filter by specific strategy")
def compare(symbols, strategy):
    """Compare multiple assets side by side."""
    click.echo(f"📊 Comparing assets: {', '.join(symbols)}")

    try:
        session = get_db_session()
        recommender = AIInvestmentRecommendations(session)

        comparison_df = recommender.get_asset_comparison(list(symbols), strategy)

        if comparison_df.empty:
            click.echo("No data found for specified assets")
            return

        # Display comparison table
        click.echo("\n" + comparison_df.to_string(index=False))

        session.close()

    except Exception as e:
        click.echo(f"❌ Error comparing assets: {e}", err=True)
        raise click.Abort()


@ai.command()
@click.argument("symbol")
@click.argument("strategy")
def explain(symbol, strategy):
    """Get detailed explanation for a specific asset recommendation."""
    click.echo(f"🔍 Explaining recommendation for {symbol} with {strategy} strategy...")

    try:
        session = get_db_session()
        recommender = AIInvestmentRecommendations(session)

        explanation = recommender.explain_recommendation(symbol, strategy)

        if "error" in explanation:
            click.echo(f"❌ {explanation['error']}")
            return

        click.echo(f"\n📈 {symbol} Analysis:")
        click.echo(f"Summary: {explanation['summary']}")

        if explanation.get("strengths"):
            click.echo("\n✅ Strengths:")
            for strength in explanation["strengths"]:
                click.echo(f"  • {strength}")

        if explanation.get("concerns"):
            click.echo("\n⚠️  Concerns:")
            for concern in explanation["concerns"]:
                click.echo(f"  • {concern}")

        click.echo(f"\n💡 Recommendation: {explanation['recommendation']}")

        session.close()

    except Exception as e:
        click.echo(f"❌ Error explaining recommendation: {e}", err=True)
        raise click.Abort()


def _display_table(portfolio_rec):
    """Display recommendations in table format."""
    click.echo(f"\n🎯 Portfolio Recommendations ({portfolio_rec.risk_profile.title()})")
    click.echo(
        f"Overall Score: {portfolio_rec.total_score:.2f} | Confidence: {portfolio_rec.confidence:.1%}"
    )
    click.echo(f"Diversification: {portfolio_rec.diversification_score:.1%}")

    if not portfolio_rec.recommendations:
        click.echo("No recommendations meet the specified criteria.")
        return

    # Header
    click.echo("\n" + "=" * 100)
    click.echo(
        f"{'Symbol':<8} {'Strategy':<12} {'Score':<6} {'Alloc%':<7} {'Sortino':<8} {'MaxDD':<8} {'Confidence':<10} {'Risk':<6}"
    )
    click.echo("=" * 100)

    # Recommendations
    for rec in portfolio_rec.recommendations:
        click.echo(
            f"{rec.symbol:<8} {rec.strategy:<12} {rec.score:<6.2f} {rec.allocation_percentage:<7.1f} "
            f"{rec.sortino_ratio:<8.2f} {rec.max_drawdown:<8.1%} {rec.confidence:<10.1%} {rec.risk_level:<6}"
        )

    # Warnings
    if portfolio_rec.warnings:
        click.echo("\n⚠️  Warnings:")
        for warning in portfolio_rec.warnings:
            click.echo(f"  • {warning}")

    click.echo(f"\n💭 AI Analysis: {portfolio_rec.overall_reasoning}")


def _display_json(portfolio_rec):
    """Display recommendations in JSON format."""
    # Convert to dict for JSON serialization
    data = {
        "risk_profile": portfolio_rec.risk_profile,
        "total_score": portfolio_rec.total_score,
        "confidence": portfolio_rec.confidence,
        "diversification_score": portfolio_rec.diversification_score,
        "recommendations": [
            {
                "symbol": rec.symbol,
                "strategy": rec.strategy,
                "score": rec.score,
                "confidence": rec.confidence,
                "allocation_percentage": rec.allocation_percentage,
                "risk_level": rec.risk_level,
                "metrics": {
                    "sortino_ratio": rec.sortino_ratio,
                    "calmar_ratio": rec.calmar_ratio,
                    "max_drawdown": rec.max_drawdown,
                    "win_rate": rec.win_rate,
                    "profit_factor": rec.profit_factor,
                },
                "reasoning": rec.reasoning,
                "red_flags": rec.red_flags,
            }
            for rec in portfolio_rec.recommendations
        ],
        "overall_reasoning": portfolio_rec.overall_reasoning,
        "warnings": portfolio_rec.warnings,
    }

    click.echo(json.dumps(data, indent=2))


def _display_summary(portfolio_rec):
    """Display recommendations in summary format."""
    click.echo("\n🎯 AI Investment Recommendations Summary")
    click.echo(f"Risk Profile: {portfolio_rec.risk_profile.title()}")
    click.echo(f"Recommended Assets: {len(portfolio_rec.recommendations)}")
    click.echo(f"Overall Score: {portfolio_rec.total_score:.2f}/1.0")
    click.echo(f"Confidence: {portfolio_rec.confidence:.1%}")

    if portfolio_rec.recommendations:
        top_rec = portfolio_rec.recommendations[0]
        click.echo(f"\nTop Recommendation: {top_rec.symbol} ({top_rec.strategy})")
        click.echo(f"  Allocation: {top_rec.allocation_percentage:.1f}%")
        click.echo(f"  Sortino Ratio: {top_rec.sortino_ratio:.2f}")

    click.echo(f"\n💭 {portfolio_rec.overall_reasoning}")


def _save_recommendations(portfolio_rec, output_path, format_type):
    """Save recommendations to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "json":
        with open(output_file, "w") as f:
            json.dump(_portfolio_to_dict(portfolio_rec), f, indent=2)
    else:
        with open(output_file, "w") as f:
            f.write(
                f"AI Investment Recommendations - {portfolio_rec.risk_profile.title()}\n"
            )
            f.write("=" * 60 + "\n\n")

            for rec in portfolio_rec.recommendations:
                f.write(f"{rec.symbol} ({rec.strategy})\n")
                f.write(
                    f"  Score: {rec.score:.2f} | Confidence: {rec.confidence:.1%}\n"
                )
                f.write(f"  Allocation: {rec.allocation_percentage:.1f}%\n")
                f.write(
                    f"  Sortino: {rec.sortino_ratio:.2f} | Max DD: {rec.max_drawdown:.1%}\n"
                )
                if rec.red_flags:
                    f.write(f"  Red Flags: {', '.join(rec.red_flags)}\n")
                f.write("\n")

            f.write(f"Overall Analysis:\n{portfolio_rec.overall_reasoning}\n")


def _portfolio_to_dict(portfolio_rec) -> dict:
    """Convert PortfolioRecommendation to dictionary."""
    return {
        "risk_profile": portfolio_rec.risk_profile,
        "total_score": portfolio_rec.total_score,
        "confidence": portfolio_rec.confidence,
        "recommendations": [
            {
                "symbol": rec.symbol,
                "strategy": rec.strategy,
                "score": rec.score,
                "allocation_percentage": rec.allocation_percentage,
                "metrics": {
                    "sortino_ratio": rec.sortino_ratio,
                    "calmar_ratio": rec.calmar_ratio,
                    "max_drawdown": rec.max_drawdown,
                },
            }
            for rec in portfolio_rec.recommendations
        ],
        "overall_reasoning": portfolio_rec.overall_reasoning,
        "warnings": portfolio_rec.warnings,
    }
