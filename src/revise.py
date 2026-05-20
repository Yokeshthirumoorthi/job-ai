"""Multi-step JD revision: split scraped JDs into labeled, sanitized chunks.

Pipeline per batch of jobs (point-don't-type — the LLM never emits JD text):

  step 1  boundary detection  (LLM) -> {job_id, boundary_name, block_start,
                                        block_end, is_needed}
  step 2  sentence splitting  (regex, deterministic)
  step 3  sentence labeling   (LLM) -> {job_id, index, label}
  step 4  markdown sanitize   (regex, deterministic)

Both LLM calls are BATCHED: every job in the jsonl is sent together, each
tagged with a job id the model echoes back. A jsonl of <= _BATCH_SIZE jobs is
exactly 2 LLM calls total; larger files split into safe batches of 2 calls each.

The LLM only ever returns ids/numbers/labels, so no JD text is reworded —
chunks are verbatim-derived, deterministically de-marked-down.

Flow: data/jobs_<ts>.jsonl -> data/jobs_<ts>_revised.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scraper import DATA_DIR, JobPost

ROOT = Path(__file__).resolve().parent.parent

_BATCH_SIZE = 25       # jobs per LLM call; a jsonl of <= this many = 2 calls total
_MAX_TOKENS = 16000    # generous cap so a full batch's JSON reply is not truncated
_MIN_CHUNK = 15        # drop sanitized chunks shorter than this many chars


class LLMSettings(BaseSettings):
    """OpenRouter credentials and model — loaded from .env (OPENROUTER_* keys)."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="OPENROUTER_", extra="ignore"
    )

    api_key: str
    model: str = "anthropic/claude-haiku-4.5"
    base_url: str = "https://openrouter.ai/api/v1"


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


# --- IO -----------------------------------------------------------------------


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


def main() -> None:
    settings = LLMSettings()  # type: ignore[call-arg]  # fields come from .env
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_jsonl(DATA_DIR)
    print(f"Revising {source.relative_to(ROOT)} with {settings.model}")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    jobs = read_jobs(source)
    revised = revise(jobs, client, settings.model)
    out = save_revised(revised, source)
    print(f"Saved {len(revised)} revised jobs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
