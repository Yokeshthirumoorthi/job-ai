"""Multi-step JD revision: split a scraped JD into labeled, sanitized chunks.

Pipeline per job (point-don't-type — the LLM never emits JD text):

  step 1  boundary detection  (LLM) -> {boundary_name, block_start, block_end,
                                        is_needed}; we slice the ORIGINAL lines
  step 2  sentence splitting  (regex, deterministic)
  step 3  sentence labeling   (LLM) -> {index, label}; we attach to OUR text
  step 4  markdown sanitize   (regex, deterministic)

Both LLM calls return only numbers/labels, so no JD text is ever reworded —
chunks are a verbatim-derived, deterministically de-marked-down subset.

Flow: data/jobs_<ts>.jsonl -> data/jobs_<ts>_revised.jsonl
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scraper import DATA_DIR, JobPost

ROOT = Path(__file__).resolve().parent.parent


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


# --- LLM response models (the LLM only ever fills in numbers / labels) --------


class Boundary(BaseModel):
    """One contiguous JD section, as identified by the boundary-detection step."""

    boundary_name: str
    block_start: int
    block_end: int
    is_needed: bool


class BoundaryMap(BaseModel):
    """Step-1 reply: the full list of sections."""

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


class LabelMap(BaseModel):
    """Step-3 reply: labels for every sentence index."""

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


BOUNDARY_PROMPT = """You segment a job description into its sections.

You receive the JD as numbered lines (each prefixed with "<n>: "). Identify \
every contiguous section. For each section return:
- boundary_name: a short name (e.g. "Role summary", "Responsibilities", \
"Requirements", "Preferred qualifications", "Benefits", "About the company", \
"Equal opportunity statement", "How to apply").
- block_start, block_end: the first and last line number of the section, \
inclusive.
- is_needed: true if the section is core job-description content (role or team \
summary, responsibilities, duties, requirements, qualifications, skills, \
experience); false if it is noise (benefits and perks, company-wide marketing \
or "about the company", diversity/EEO or legal boilerplate, application \
instructions, compensation disclaimers).

The sections must be contiguous and in order, together covering every line \
exactly once with no gaps and no overlaps.

Return ONLY a JSON object:
{"boundaries":[{"boundary_name":"Role summary","block_start":0,"block_end":3,\
"is_needed":true}]}"""

LABEL_PROMPT = """You label sentences taken from the kept parts of a job \
description.

You receive numbered sentences (each prefixed with "<n>: "). For each sentence \
return its index and exactly one label:
- "responsibility": a duty or activity the person in the role will perform.
- "requirement": a must-have qualification, skill, technology or experience.
- "preferred": a nice-to-have or bonus qualification ("preferred", "a plus").
- "context": role, team or company background, or anything not above.

Do not rewrite, summarize or alter any sentence — only classify it by index.

Return ONLY a JSON object:
{"labels":[{"index":0,"label":"context"}]}"""


# --- Markdown handling (deterministic) ----------------------------------------

_RULE = re.compile(r"^[-=_*\s]{3,}$")              # setext underlines / hr rules
_ATX = re.compile(r"^#{1,6}\s+")                   # "### Heading"
_BULLET = re.compile(r"^\s*(?:[*+-]|\d+[.)])\s+")  # leading list markers
_EMPH = re.compile(r"\*\*|\*|__|`")                # bold / italic / code marks
_ESCAPE = re.compile(r"\\([-\\`*_{}\[\]()#+.!&>~])")  # markdown backslash escapes
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # sentence boundary
_MIN_CHUNK = 15                                    # drop fragments shorter than this


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
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json(reply: str) -> str:
    """Pull the outermost JSON object out of an LLM reply (tolerates fences)."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in LLM reply: {reply!r}")
    return reply[start : end + 1]


def _numbered(items: list[str]) -> str:
    """Render items as '<n>: <item>' lines for an LLM prompt."""
    return "\n".join(f"{i}: {item}" for i, item in enumerate(items))


# --- Pipeline steps -----------------------------------------------------------


def detect_boundaries(blocks: list[str], client: OpenAI, model: str) -> list[Boundary]:
    """Step 1 — LLM marks section ranges; indices are clamped to valid blocks."""
    try:
        reply = _chat(client, model, BOUNDARY_PROMPT, _numbered(blocks))
        parsed = BoundaryMap.model_validate_json(_extract_json(reply))
    except Exception as error:  # noqa: BLE001 - never lose the JD on failure
        print(f"  ! boundary step failed ({error}); keeping the whole JD")
        return [Boundary(
            boundary_name="all", block_start=0,
            block_end=len(blocks) - 1, is_needed=True,
        )]
    last = len(blocks) - 1
    clamped: list[Boundary] = []
    for b in parsed.boundaries:
        start = min(max(b.block_start, 0), last)
        end = min(max(b.block_end, start), last)
        clamped.append(Boundary(
            boundary_name=b.boundary_name, block_start=start,
            block_end=end, is_needed=b.is_needed,
        ))
    return clamped


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


def label_sentences(
    sentences: list[Segment], client: OpenAI, model: str
) -> list[ChunkLabel]:
    """Step 3 — LLM labels each sentence by index; unlabelled -> context."""
    labels = [ChunkLabel.context] * len(sentences)
    if not sentences:
        return labels
    try:
        reply = _chat(
            client, model, LABEL_PROMPT, _numbered([s.text for s in sentences])
        )
        parsed = LabelMap.model_validate_json(_extract_json(reply))
    except Exception as error:  # noqa: BLE001 - keep all sentences as context
        print(f"  ! label step failed ({error}); labeling all as context")
        return labels
    for item in parsed.labels:
        if 0 <= item.index < len(sentences):
            labels[item.index] = item.label
    return labels


def revise_job(job: JobPost, client: OpenAI, model: str) -> RevisedJobPost:
    """Run the four-step pipeline for a single job."""
    blocks = [line for line in (job.description or "").split("\n") if line.strip()]
    if not blocks:
        return RevisedJobPost(**job.model_dump())
    boundaries = detect_boundaries(blocks, client, model)          # step 1
    sentences = to_sentences(needed_segments(blocks, boundaries))  # step 2
    labels = label_sentences(sentences, client, model)             # step 3
    chunks: list[JDChunk] = []
    for sentence, label in zip(sentences, labels):
        text = strip_markup(sentence.text)                         # step 4
        if len(text) >= _MIN_CHUNK:
            chunks.append(JDChunk(text=text, label=label, section=sentence.section))
    return RevisedJobPost(
        **job.model_dump(), boundaries=boundaries, description_chunks=chunks
    )


def revise(jobs: list[JobPost], client: OpenAI, model: str) -> list[RevisedJobPost]:
    """Run the pipeline over every job, printing a one-line summary each."""
    revised: list[RevisedJobPost] = []
    for index, job in enumerate(jobs, start=1):
        result = revise_job(job, client, model)
        kept = sum(1 for b in result.boundaries if b.is_needed)
        print(
            f"  [{index}/{len(jobs)}] {job.title or '(untitled)'}"
            f" — {kept}/{len(result.boundaries)} sections kept,"
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
