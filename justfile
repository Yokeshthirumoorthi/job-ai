# Show available recipes
default:
    @just --list

# Install dependencies into a uv-managed virtualenv
install:
    uv sync

# Scrape jobs using config/base.yaml -> data/jobs_<timestamp>.jsonl
scrape:
    uv run python src/scraper.py

# Print the path of the most recent output file
latest:
    @ls -t data/*.jsonl 2>/dev/null | head -1 || echo "no output yet"

# Delete all scraped output
clean:
    rm -f data/*.jsonl
