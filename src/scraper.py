"""Scrape jobs from multiple boards into a typed JSONL file.

Flow: config/base.yaml -> ScrapeConfig -> scrape_jobs() -> list[JobPost]
      -> data/jobs_<timestamp>.jsonl
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from jobspy import scrape_jobs
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "base.yaml"
DATA_DIR = ROOT / "data"


class Site(str, Enum):
    """Job board JobSpy can scrape."""

    linkedin = "linkedin"
    indeed = "indeed"
    glassdoor = "glassdoor"
    google = "google"
    zip_recruiter = "zip_recruiter"
    bayt = "bayt"
    naukri = "naukri"
    bdjobs = "bdjobs"


class JobType(str, Enum):
    """Employment type filter."""

    fulltime = "fulltime"
    parttime = "parttime"
    internship = "internship"
    contract = "contract"


class ScrapeConfig(BaseModel):
    """Typed view of config/base.yaml — the inputs to a scrape run."""

    model_config = ConfigDict(extra="forbid")

    site_name: list[Site]
    search_term: str
    location: str | None = None
    google_search_term: str | None = None
    results_wanted: int = Field(default=20, ge=1)
    distance: int = Field(default=50, ge=0)
    job_type: JobType | None = None
    is_remote: bool = False
    hours_old: int | None = Field(default=None, ge=1)
    country_indeed: str = "usa"
    linkedin_fetch_description: bool = False
    verbose: int = Field(default=1, ge=0, le=2)

    @classmethod
    def load(cls, path: Path) -> "ScrapeConfig":
        """Read and validate a YAML spec file."""
        return cls.model_validate(yaml.safe_load(path.read_text()))


class JobPost(BaseModel):
    """One scraped job posting — one line in the output JSONL."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    site: str | None = None
    job_url: str | None = None
    job_url_direct: str | None = None
    title: str | None = None
    company: str | None = None
    location: str | None = None
    date_posted: date | None = None
    job_type: str | None = None
    salary_source: str | None = None
    interval: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    currency: str | None = None
    is_remote: bool | None = None
    job_level: str | None = None
    job_function: str | None = None
    company_industry: str | None = None
    company_url: str | None = None
    company_logo: str | None = None
    emails: str | None = None
    skills: str | None = None
    description: str | None = None


def scrape(config: ScrapeConfig) -> list[JobPost]:
    """Run JobSpy and return validated JobPost models."""
    # model_dump(mode="json") turns enums into the plain strings JobSpy wants;
    # this is the only place a raw mapping touches the library boundary.
    frame: pd.DataFrame = scrape_jobs(
        **config.model_dump(mode="json", exclude_none=True)
    )
    rows = frame.astype(object).where(frame.notna(), None)
    return [JobPost.model_validate(record) for record in rows.to_dict("records")]


def save(jobs: list[JobPost], data_dir: Path) -> Path:
    """Write jobs to data/jobs_<timestamp>.jsonl and return the path."""
    data_dir.mkdir(exist_ok=True)
    out = data_dir / f"jobs_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(job.model_dump_json() + "\n")
    return out


def main() -> None:
    config = ScrapeConfig.load(CONFIG_PATH)
    jobs = scrape(config)
    out = save(jobs, DATA_DIR)
    print(f"Saved {len(jobs)} jobs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
