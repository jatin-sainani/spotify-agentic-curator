"""Main CLI entrypoint for spotify-agentic-curator."""


import typer

# Import command modules
from app import commands
from core.config import settings
from core.logging import logger, setup_logging

# Create main app
app = typer.Typer(
    name="spotify-agentic-curator",
    help="AI-powered Spotify playlist curation tool",
    add_completion=False,
)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Spotify Agentic Curator - AI-powered playlist curation."""
    # Setup logging
    log_level = "DEBUG" if debug or verbose else settings.log_level
    setup_logging(level=log_level)

    logger.info("Spotify Agentic Curator starting up")


# Register commands
app.add_typer(commands.connect_app, name="connect")
app.add_typer(commands.tools_app, name="tools-smoke-test")
app.add_typer(commands.llm_app, name="llm")
app.add_typer(commands.curate_app, name="curate")
app.add_typer(commands.draft_app, name="draft")
app.add_typer(commands.catalog_app, name="catalog")
app.add_typer(commands.eval_app, name="eval")
app.add_typer(commands.mcp_app, name="mcp")


if __name__ == "__main__":
    app()
