# Show available recipes
default:
    @just --list

# Install dependencies into a uv-managed virtualenv
install:
    uv sync

# Scrape jobs using config/base.yaml -> data/jobs_<timestamp>.jsonl
scrape:
    uv run python src/scraper.py

# LLM-clean the newest jsonl -> data/jobs_<timestamp>_revised.jsonl
revise:
    uv run python src/revise.py

# Export a stripped human-eval JSONL from the newest _revised.jsonl
eval:
    uv run python src/export_eval.py

# Scrape then revise in one go
all: scrape revise

# Print the path of the most recent output file
latest:
    @ls -t data/*.jsonl 2>/dev/null | head -1 || echo "no output yet"

# Delete all scraped output
clean:
    rm -f data/*.jsonl
