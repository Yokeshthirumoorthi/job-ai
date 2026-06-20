"""End-to-end job pipeline: scrape -> revise -> eval, in one module.

Three stages, each usable as an importable function (from a notebook) or as a
CLI step (`python src/pipeline.py <step>`):

  scrape   config/base.yaml -> ScrapeConfig -> scrape_jobs() -> list[JobPost]
           -> data/jobs_<ts>.jsonl
  revise   data/jobs_<ts>.jsonl -> LLM-labeled chunks
           -> data/jobs_<ts>_revised.jsonl
  eval     data/jobs_<ts>_revised.jsonl -> slim, label-grouped review file
           -> data/jobs_<ts>_eval.jsonl

The revise stage is point-don't-type: the LLM only ever returns ids / numbers /
labels, so JD text is verbatim-derived and never reworded.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    # Heavy SDKs (openai, pandas, jobspy) are imported lazily inside the
    # functions that use them, so `import pipeline` stays fast. This block is
    # only for type checkers — it never runs.
    from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "base.yaml"
DATA_DIR = ROOT / "data"

_BATCH_SIZE = 25       # jobs per LLM call; a jsonl of <= this many = 2 calls total
_MAX_TOKENS = 16000    # generous cap so a full batch's JSON reply is not truncated
_MIN_CHUNK = 15        # drop sanitized chunks shorter than this many chars


# === Stage 1: scrape ==========================================================


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
    import pandas as pd
    from jobspy import scrape_jobs

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


# === Stage 2: revise ==========================================================


class LLMSettings(BaseSettings):
    """OpenRouter credentials and model — loaded from .env (OPENROUTER_* keys)."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="OPENROUTER_", extra="ignore"
    )

    api_key: str
    model: str = "anthropic/claude-haiku-4.5"
    base_url: str = "https://openrouter.ai/api/v1"


def make_client(settings: "LLMSettings") -> "OpenAI":
    """Build an OpenAI-SDK client pointed at OpenRouter.

    OpenRouter speaks the OpenAI API, so the `openai` package is used purely as
    the HTTP client aimed at OpenRouter's base URL — no OpenAI service is
    called. Imported lazily here to keep `import pipeline` fast.
    """
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


class ChunkLabel(str, Enum):
    """What a JD sentence is — used later to weight resume↔JD similarity."""

    responsibility = "responsibility"  # a duty the role performs
    requirement = "requirement"        # a must-have qualification/skill
    preferred = "preferred"            # a nice-to-have / bonus
    context = "context"                # role/team/company background, or other


# --- LLM response models (the LLM only ever fills in ids / numbers / labels) ---


class Boundary(BaseModel):
    """One contiguous JD section, as identified by the boundary-detection step."""

    boundary_name: str
    block_start: int
    block_end: int
    is_needed: bool


class JobBoundaries(BaseModel):
    """Step-1 reply for one job — its section list, keyed by echoed job_id."""

    job_id: str
    boundaries: list[Boundary]


class LabeledIndex(BaseModel):
    """Step-3 reply item: a sentence index and its label."""

    index: int
    label: ChunkLabel

    @field_validator("label", mode="before")
    @classmethod
    def _default_unknown(cls, value: object) -> object:
        """Fall back to `context` if the model returns an unexpected label."""
        try:
            return ChunkLabel(str(value).strip().lower())
        except ValueError:
            return ChunkLabel.context


class JobLabels(BaseModel):
    """Step-3 reply for one job — sentence labels, keyed by echoed job_id."""

    job_id: str
    labels: list[LabeledIndex]


# --- Internal + output models -------------------------------------------------


class Segment(BaseModel):
    """A piece of JD text (line or sentence) tagged with its section name."""

    section: str
    text: str


class JDChunk(BaseModel):
    """A labeled, sanitized JD sentence — one unit to embed later."""

    text: str
    label: ChunkLabel
    section: str


class RevisedJobPost(JobPost):
    """A JobPost plus its section map and labeled, sanitized chunks."""

    boundaries: list[Boundary] = Field(default_factory=list)
    description_chunks: list[JDChunk] = Field(default_factory=list)


class JobState(BaseModel):
    """Mutable per-job scratch carried through the batched pipeline."""

    job_id: str
    job: JobPost
    blocks: list[str] = Field(default_factory=list)
    boundaries: list[Boundary] = Field(default_factory=list)
    sentences: list[Segment] = Field(default_factory=list)
    labels: list[ChunkLabel] = Field(default_factory=list)


BOUNDARY_PROMPT = """You segment job descriptions into their sections.

You may receive SEVERAL jobs in one request. Each job starts with a line \
"### JOB <job_id>" followed by that job's numbered lines (numbering restarts \
at 0 for every job). Process every job and echo its job_id exactly.

For each job, identify every contiguous section. For each section return:
- boundary_name: a short name (e.g. "Role summary", "Responsibilities", \
"Requirements", "Preferred qualifications", "Benefits", "About the company", \
"Equal opportunity statement", "How to apply").
- block_start, block_end: the first and last line number of the section, \
inclusive (using that job's own numbering).
- is_needed: true if the section is core job-description content (role or team \
summary, responsibilities, duties, requirements, qualifications, skills, \
experience); false if it is noise (benefits and perks, company-wide marketing \
or "about the company", diversity/EEO or legal boilerplate, application \
instructions, compensation disclaimers).

Within each job the sections must be contiguous and in order, together \
covering every line exactly once with no gaps and no overlaps.

Return ONLY a JSON object:
{"jobs":[{"job_id":"<id>","boundaries":[{"boundary_name":"Role summary",\
"block_start":0,"block_end":3,"is_needed":true}]}]}"""

LABEL_PROMPT = """You label sentences taken from the kept parts of job \
descriptions.

You may receive SEVERAL jobs in one request. Each job starts with a line \
"### JOB <job_id>" followed by that job's numbered sentences (numbering \
restarts at 0 for every job). Process every job and echo its job_id exactly.

For each sentence return its index and exactly one label:
- "responsibility": a duty or activity the person in the role will perform.
- "requirement": a must-have qualification, skill, technology or experience.
- "preferred": a nice-to-have or bonus qualification ("preferred", "a plus").
- "context": role, team or company background, or anything not above.

Do not rewrite, summarize or alter any sentence — only classify it by index.

Return ONLY a JSON object:
{"jobs":[{"job_id":"<id>","labels":[{"index":0,"label":"context"}]}]}"""


# --- Markdown handling (deterministic) ----------------------------------------

_RULE = re.compile(r"^[-=_*\s]{3,}$")              # setext underlines / hr rules
_ATX = re.compile(r"^#{1,6}\s+")                   # "### Heading"
_BULLET = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+")  # leading list markers
_EMPH = re.compile(r"\*\*|\*|__|`")                # bold / italic / code marks
_ESCAPE = re.compile(r"\\([-\\`*_{}\[\]()#+.!&>~])")  # markdown backslash escapes
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # sentence boundary


def _is_heading(line: str) -> bool:
    """True for a line that is only a section heading, carrying no JD content."""
    stripped = line.strip()
    if _ATX.match(stripped):                          # "### What We Build"
        return True
    text = _EMPH.sub("", stripped).strip()
    if not text or len(text.split()) > 6:             # long line = real content
        return False
    if stripped.startswith("**") and stripped.endswith("**"):
        return True                                   # "**What you will do**"
    return text.replace(" ", "").isalpha() and text.isupper()  # "DESIRED QUALIFICATIONS"


def strip_markup(text: str) -> str:
    """Strip markdown marks and escapes from a single sentence (step 4)."""
    text = _ESCAPE.sub(r"\1", text)   # \& -> &
    text = _BULLET.sub("", text)      # drop leading "* " / "1. "
    text = _EMPH.sub("", text)        # drop ** * ` __
    return _SPACE.sub(" ", text).strip()


# --- LLM plumbing -------------------------------------------------------------


def _chat(client: OpenAI, model: str, system: str, user: str) -> str:
    """Send one system+user turn and return the raw reply text."""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _job_items(reply: str) -> list[object]:
    """Pull the 'jobs' array out of a batched LLM reply (tolerates fences)."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in LLM reply: {reply!r}")
    parsed = json.loads(reply[start : end + 1])
    items = parsed.get("jobs") if isinstance(parsed, dict) else None
    return items if isinstance(items, list) else []


def _numbered(items: list[str]) -> str:
    """Render items as '<n>: <item>' lines for an LLM prompt."""
    return "\n".join(f"{i}: {item}" for i, item in enumerate(items))


def _job_block(job_id: str, items: list[str]) -> str:
    """Render one job's section of a batched prompt."""
    return f"### JOB {job_id}\n{_numbered(items)}"


# --- Pipeline steps (batched) -------------------------------------------------


def _blocks(description: str | None) -> list[str]:
    """Split a raw JD into numbered blocks (its non-empty lines)."""
    return [line for line in (description or "").split("\n") if line.strip()]


def _whole_jd(block_count: int) -> list[Boundary]:
    """Fallback section map — one needed section covering the entire JD."""
    return [Boundary(
        boundary_name="all", block_start=0,
        block_end=max(block_count - 1, 0), is_needed=True,
    )]


def detect_boundaries(batch: list[JobState], client: OpenAI, model: str) -> None:
    """Step 1 — one LLM call marks section ranges for every job in the batch."""
    usable = [state for state in batch if state.blocks]
    parsed: list[JobBoundaries] = []
    if usable:
        prompt = "\n\n".join(_job_block(s.job_id, s.blocks) for s in usable)
        try:
            for item in _job_items(_chat(client, model, BOUNDARY_PROMPT, prompt)):
                try:
                    parsed.append(JobBoundaries.model_validate(item))
                except ValidationError:
                    continue  # skip one malformed job, keep the rest of the batch
        except (ValueError, json.JSONDecodeError) as error:
            print(f"  ! boundary step failed ({error}); keeping whole JDs")
    for state in batch:
        match = next((j for j in parsed if j.job_id == state.job_id), None)
        if match is None:
            state.boundaries = _whole_jd(len(state.blocks))
            continue
        last = len(state.blocks) - 1
        state.boundaries = [
            Boundary(
                boundary_name=b.boundary_name,
                block_start=min(max(b.block_start, 0), last),
                block_end=min(max(b.block_end, min(max(b.block_start, 0), last)), last),
                is_needed=b.is_needed,
            )
            for b in match.boundaries
        ]


def needed_segments(blocks: list[str], boundaries: list[Boundary]) -> list[Segment]:
    """Collect verbatim lines from needed sections, tagged with their section.

    Heading and rule lines are dropped. Blocks that no boundary covered are
    kept too, so the LLM omitting a range never silently loses JD content.
    """
    segments: list[Segment] = []
    covered: set[int] = set()
    for b in boundaries:
        covered.update(range(b.block_start, b.block_end + 1))
        if not b.is_needed:
            continue
        for raw in blocks[b.block_start : b.block_end + 1]:
            if not (_RULE.match(raw.strip()) or _is_heading(raw)):
                segments.append(Segment(section=b.boundary_name, text=raw))
    for index, raw in enumerate(blocks):
        if index not in covered and not (_RULE.match(raw.strip()) or _is_heading(raw)):
            segments.append(Segment(section="uncovered", text=raw))
    return segments


def to_sentences(segments: list[Segment]) -> list[Segment]:
    """Step 2 — split each line into sentences (deterministic); dedupe."""
    sentences: list[Segment] = []
    seen: set[str] = set()
    for segment in segments:
        for piece in _SENTENCE.split(segment.text.strip()):
            piece = piece.strip()
            if piece and piece not in seen:
                seen.add(piece)
                sentences.append(Segment(section=segment.section, text=piece))
    return sentences


def label_sentences(batch: list[JobState], client: OpenAI, model: str) -> None:
    """Step 3 — one LLM call labels every sentence of every job in the batch."""
    usable = [state for state in batch if state.sentences]
    parsed: list[JobLabels] = []
    if usable:
        prompt = "\n\n".join(
            _job_block(s.job_id, [x.text for x in s.sentences]) for s in usable
        )
        try:
            for item in _job_items(_chat(client, model, LABEL_PROMPT, prompt)):
                try:
                    parsed.append(JobLabels.model_validate(item))
                except ValidationError:
                    continue  # skip one malformed job, keep the rest of the batch
        except (ValueError, json.JSONDecodeError) as error:
            print(f"  ! label step failed ({error}); labeling all as context")
    for state in batch:
        labels = [ChunkLabel.context] * len(state.sentences)
        match = next((j for j in parsed if j.job_id == state.job_id), None)
        if match is not None:
            for item in match.labels:
                if 0 <= item.index < len(labels):
                    labels[item.index] = item.label
        state.labels = labels


def _assemble(state: JobState) -> RevisedJobPost:
    """Step 4 — sanitize each labeled sentence into the final chunk list."""
    chunks: list[JDChunk] = []
    for sentence, label in zip(state.sentences, state.labels):
        text = strip_markup(sentence.text)
        if len(text) >= _MIN_CHUNK:
            chunks.append(JDChunk(text=text, label=label, section=sentence.section))
    return RevisedJobPost(
        **state.job.model_dump(),
        boundaries=state.boundaries,
        description_chunks=chunks,
    )


def revise(jobs: list[JobPost], client: OpenAI, model: str) -> list[RevisedJobPost]:
    """Run the batched pipeline — 2 LLM calls per batch of _BATCH_SIZE jobs."""
    states = [
        JobState(job_id=job.id or f"J{i}", job=job, blocks=_blocks(job.description))
        for i, job in enumerate(jobs)
    ]
    batches = [states[i : i + _BATCH_SIZE] for i in range(0, len(states), _BATCH_SIZE)]
    revised: list[RevisedJobPost] = []
    for number, batch in enumerate(batches, start=1):
        print(f"  batch {number}/{len(batches)} ({len(batch)} jobs) — 2 LLM calls")
        detect_boundaries(batch, client, model)                       # LLM call 1
        for state in batch:
            state.sentences = to_sentences(
                needed_segments(state.blocks, state.boundaries)
            )
        label_sentences(batch, client, model)                         # LLM call 2
        for state in batch:
            result = _assemble(state)
            kept = sum(1 for b in result.boundaries if b.is_needed)
            print(
                f"    {state.job.title or '(untitled)'} —"
                f" {kept}/{len(result.boundaries)} sections kept,"
                f" {len(result.description_chunks)} chunks"
            )
            revised.append(result)
    return revised


def latest_jsonl(data_dir: Path) -> Path:
    """Return the newest data/jobs_*.jsonl that is not itself a revised file."""
    candidates = sorted(
        (p for p in data_dir.glob("jobs_*.jsonl") if not p.stem.endswith("_revised")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no jobs_*.jsonl found in {data_dir}")
    return candidates[-1]


def read_jobs(path: Path) -> list[JobPost]:
    """Read a jsonl file back into typed JobPost models."""
    with path.open(encoding="utf-8") as handle:
        return [JobPost.model_validate_json(line) for line in handle if line.strip()]


def save_revised(jobs: list[RevisedJobPost], source: Path) -> Path:
    """Write revised jobs next to the source as jobs_<ts>_revised.jsonl."""
    out = source.with_name(f"{source.stem}_revised.jsonl")
    with out.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(job.model_dump_json() + "\n")
    return out


# === Stage 3: eval export =====================================================


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


# === CLI ======================================================================


def _run_scrape(_argv: list[str]) -> None:
    config = ScrapeConfig.load(CONFIG_PATH)
    jobs = scrape(config)
    out = save(jobs, DATA_DIR)
    print(f"Saved {len(jobs)} jobs -> {out.relative_to(ROOT)}")


def _run_revise(argv: list[str]) -> None:
    settings = LLMSettings()  # type: ignore[call-arg]  # fields come from .env
    source = Path(argv[0]) if argv else latest_jsonl(DATA_DIR)
    print(f"Revising {source.relative_to(ROOT)} with {settings.model}")
    client = make_client(settings)
    jobs = read_jobs(source)
    revised = revise(jobs, client, settings.model)
    out = save_revised(revised, source)
    print(f"Saved {len(revised)} revised jobs -> {out.relative_to(ROOT)}")


def _run_eval(argv: list[str]) -> None:
    source = Path(argv[0]) if argv else latest_revised(DATA_DIR)
    jobs = [to_eval(job) for job in read_revised(source)]
    out = save_eval(jobs, source)
    sentences = sum(len(group.text) for job in jobs for group in job.labeled_jd)
    print(f"Exported {len(jobs)} jobs ({sentences} sentences) -> {out.relative_to(ROOT)}")


_STEPS = {"scrape": _run_scrape, "revise": _run_revise, "eval": _run_eval}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _STEPS:
        print(f"usage: python src/pipeline.py {{{'|'.join(_STEPS)}}} [path]")
        raise SystemExit(1)
    _STEPS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
