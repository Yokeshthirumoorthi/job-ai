# Show available recipes
default:
    @just --list

# Install dependencies into a uv-managed virtualenv
install:
    uv sync

# Scrape jobs using config/base.yaml -> data/jobs_<timestamp>.jsonl
scrape:
    uv run python src/pipeline.py scrape

# LLM-clean the newest jsonl -> data/jobs_<timestamp>_revised.jsonl
revise:
    uv run python src/pipeline.py revise

# Export a stripped human-eval JSONL from the newest _revised.jsonl
eval:
    uv run python src/pipeline.py eval

# Validate resume.yaml and chunk it -> data/resume_bullets.json
chunk-resume:
    uv run python src/resume.py chunk

# Embed resume bullets on the GPU (cached) -> cache/resume_embeddings.npz
embed:
    uv run python src/resume.py embed

# Load and schema-validate the newest jobs eval jsonl
load-jobs:
    uv run python src/score.py load

# Score the first job of the newest eval jsonl against the resume
score:
    uv run python src/score.py score

# Rank every job against the resume -> out/ranked_jobs.csv
rank:
    uv run python src/score.py rank

# Scrape then revise in one go
all: scrape revise

# Print the path of the most recent output file
latest:
    @ls -t data/*.jsonl 2>/dev/null | head -1 || echo "no output yet"

# Delete all scraped output
clean:
    rm -f data/*.jsonl
