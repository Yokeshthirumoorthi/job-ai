"""Export a stripped, human-eval JSONL from a revised jobs file.

Reads data/jobs_<ts>_revised.jsonl and writes data/jobs_<ts>_eval.jsonl with
only {job_id, position, company, url, labeled_jd}, where labeled_jd groups the
JD sentences by the label the revise step's LLM assigned — so a human can scan
each label's sentences and judge whether the labeling is right.

Flow: data/jobs_<ts>_revised.jsonl -> data/jobs_<ts>_eval.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

from revise import ChunkLabel, RevisedJobPost
from scraper import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent


class EvalGroup(BaseModel):
    """All JD sentences the revise step gave one label."""

    label: ChunkLabel
    text: list[str]


class EvalJob(BaseModel):
    """A job stripped down to what a human evaluator needs."""

    job_id: str
    position: str
    company: str
    url: str
    labeled_jd: list[EvalGroup] = Field(default_factory=list)


def latest_revised(data_dir: Path) -> Path:
    """Return the newest data/jobs_*_revised.jsonl."""
    candidates = sorted(
        data_dir.glob("jobs_*_revised.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"no jobs_*_revised.jsonl found in {data_dir}")
    return candidates[-1]


def read_revised(path: Path) -> list[RevisedJobPost]:
    """Read a revised jsonl back into typed RevisedJobPost models."""
    with path.open(encoding="utf-8") as handle:
        return [
            RevisedJobPost.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def to_eval(job: RevisedJobPost) -> EvalJob:
    """Project a revised job down to the human-eval fields, grouped by label."""
    groups: list[EvalGroup] = []
    for label in ChunkLabel:
        text = [chunk.text for chunk in job.description_chunks if chunk.label == label]
        if text:
            groups.append(EvalGroup(label=label, text=text))
    return EvalJob(
        job_id=job.id or "",
        position=job.title or "",
        company=job.company or "",
        url=job.job_url or job.job_url_direct or "",
        labeled_jd=groups,
    )


def save_eval(jobs: list[EvalJob], source: Path) -> Path:
    """Write the stripped jobs as jobs_<ts>_eval.jsonl next to the source."""
    out = source.with_name(f"{source.stem.removesuffix('_revised')}_eval.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(job.model_dump_json() + "\n")
    return out


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_revised(DATA_DIR)
    jobs = [to_eval(job) for job in read_revised(source)]
    out = save_eval(jobs, source)
    sentences = sum(len(group.text) for job in jobs for group in job.labeled_jd)
    print(f"Exported {len(jobs)} jobs ({sentences} sentences) -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
