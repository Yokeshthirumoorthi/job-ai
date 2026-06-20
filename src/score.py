"""Score and rank jobs against the resume.

Loads the eval jsonl (pipeline's `eval` output) into typed Job models, embeds
each JD on the GPU (cached per job), compares against the resume embeddings, and
ranks every job into out/ranked_jobs.csv.

For each JD label, every JD bullet is compared (cosine, via normalized
embeddings) to every resume bullet; its score is its best resume match. The
label score is the mean of those best matches; the composite is a weighted mean
over the labels the job actually has. score_job() is called once per job — the
embedding model (resume.get_model) is loaded at most once, and not at all on a
fully-cached run.

CLI:
  python src/score.py load    # validate the newest *_eval.jsonl
  python src/score.py score   # score the first job (debug)
  python src/score.py rank    # rank all jobs -> out/ranked_jobs.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from resume import MODEL_NAME, OUTPUT_PATH as RESUME_BULLETS, embed_texts, load_or_embed

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JD_CACHE_DIR = ROOT / "cache" / "jd_embeddings"
OUT_CSV = ROOT / "out" / "ranked_jobs.csv"


# === Load + validate jobs =====================================================


class JobLabel(str, Enum):
    """What a JD sentence is — mirrors pipeline.ChunkLabel's four values.

    Defined locally rather than imported: pulling it from pipeline would
    transitively reach the scrape/revise machinery a scorer has no need for.
    """

    responsibility = "responsibility"
    requirement = "requirement"
    preferred = "preferred"
    context = "context"


class LabeledGroup(BaseModel):
    """One label's worth of JD sentences."""

    model_config = ConfigDict(extra="forbid")

    label: JobLabel
    text: list[str]


class Job(BaseModel):
    """One validated job posting — one line of the input jsonl."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    position: str
    company: str
    url: str
    labeled_jd: list[LabeledGroup]


def _loc(loc: tuple[object, ...]) -> str:
    """Render a pydantic error location as e.g. 'labeled_jd[0].label'."""
    rendered = ""
    for part in loc:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}" if rendered else str(part)
    return rendered or "<line>"


def _reason(error: ValidationError) -> str:
    """Condense a ValidationError into a one-line, human-readable reason."""
    return "; ".join(f"{_loc(e['loc'])}: {e['msg']}" for e in error.errors())


def _label_counts(jobs: list[Job]) -> dict[JobLabel, int]:
    """Tally how many JD sentences carry each label across all jobs."""
    counts = {label: 0 for label in JobLabel}
    for job in jobs:
        for group in job.labeled_jd:
            counts[group.label] += len(group.text)
    return counts


def _print_summary(jobs: list[Job], skipped: int) -> None:
    """Print the loaded / skipped / labels-seen summary."""
    counts = _label_counts(jobs)
    print(f"Loaded: {len(jobs)} jobs")
    print(f"Skipped: {skipped}" + (" (see warnings)" if skipped else ""))
    tally = ", ".join(f"{label.value}={counts[label]}" for label in JobLabel)
    print(f"Labels seen: {tally}")


def load_jobs(path: Path) -> list[Job]:
    """Read and validate a jobs jsonl; skip bad lines, return the clean list."""
    jobs: list[Job] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                jobs.append(Job.model_validate_json(raw))
            except ValidationError as error:
                skipped += 1
                print(f"  ! line {line_no} skipped — {_reason(error)}", file=sys.stderr)
    _print_summary(jobs, skipped)
    return jobs


def latest_eval(data_dir: Path) -> Path:
    """Return the newest data/jobs_*_eval.jsonl file."""
    candidates = sorted(
        data_dir.glob("jobs_*_eval.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"no jobs_*_eval.jsonl found in {data_dir}")
    return candidates[-1]


# === Semantic scoring =========================================================

# Composite weights — how much each label contributes to the overall score.
WEIGHTS: dict[JobLabel, float] = {
    JobLabel.requirement: 0.50,
    JobLabel.responsibility: 0.30,
    JobLabel.preferred: 0.15,
    JobLabel.context: 0.05,
}


def _resolve_jd_vecs(
    jd_vecs_by_label: dict[Any, "np.ndarray"] | None,
    label: JobLabel,
    texts: list[str],
) -> "np.ndarray":
    """JD vectors for one label — use pre-computed ones if supplied, else embed.

    Accepts a pre-computed dict keyed by either JobLabel or its string value;
    falls back to embedding any label the dict happens not to cover.
    """
    if jd_vecs_by_label is not None:
        if label in jd_vecs_by_label:
            return jd_vecs_by_label[label]
        if label.value in jd_vecs_by_label:
            return jd_vecs_by_label[label.value]
    return embed_texts(texts)


def score_job(
    job: Job,
    resume_vecs: "np.ndarray",
    resume_bullets: list[dict[str, Any]],
    jd_vecs_by_label: dict[Any, "np.ndarray"] | None = None,
) -> dict[str, Any]:
    """Score one job against the resume.

    resume_vecs (N, D) and resume_bullets (length N) must be index-aligned —
    row i of resume_vecs is the embedding of resume_bullets[i].

    jd_vecs_by_label, when given, supplies pre-embedded JD vectors per label
    (the disk-cache fast path); when None, JD bullets are embedded on demand via
    the singleton model.
    """
    if len(resume_bullets) != resume_vecs.shape[0]:
        raise ValueError(
            f"resume_bullets ({len(resume_bullets)}) and resume_vecs "
            f"({resume_vecs.shape[0]}) must be the same length and aligned"
        )

    # 1. group JD bullets by label (defensively merge any duplicate groups)
    jd_by_label: dict[JobLabel, list[str]] = {}
    for group in job.labeled_jd:
        jd_by_label.setdefault(group.label, []).extend(group.text)

    per_label: dict[str, float] = {}
    coverage: dict[str, list[dict[str, Any]]] = {}

    # 2. score each label that actually has bullets
    for label, jd_bullets in jd_by_label.items():
        if not jd_bullets:
            continue
        jd_vecs = _resolve_jd_vecs(jd_vecs_by_label, label, jd_bullets)
        sims = resume_vecs @ jd_vecs.T          # (n_resume, n_jd) cosine sims
        best_per_jd = sims.max(axis=0)          # best resume match per JD bullet
        best_idx = sims.argmax(axis=0)          # which resume bullet that was
        per_label[label.value] = float(best_per_jd.mean())
        coverage[label.value] = [
            {
                "jd_bullet": jd_bullets[i],
                "score": float(best_per_jd[i]),
                "best_resume_bullet_id": resume_bullets[int(best_idx[i])]["id"],
                "best_resume_bullet": resume_bullets[int(best_idx[i])]["text"],
            }
            for i in range(len(jd_bullets))
        ]

    # 3. composite — weighted MEAN over the labels actually present, normalized
    #    by their weights. Without the division a job whose JD has no `preferred`
    #    bullets is summed over only 0.85 of total weight and so is structurally
    #    capped below an otherwise-equal job that has all four labels.
    present = [label for label in jd_by_label if label.value in per_label]
    weight_sum = sum(WEIGHTS[label] for label in present)
    composite = (
        sum(WEIGHTS[label] * per_label[label.value] for label in present)
        / weight_sum
        if weight_sum
        else 0.0
    )

    return {
        "composite": float(composite),
        "per_label": per_label,
        "coverage": coverage,
    }


def aligned_resume_bullets(ids: list[str]) -> list[dict[str, Any]]:
    """Load resume_bullets.json and reorder it to match the embedding ids."""
    by_id = {b["id"]: b for b in json.loads(RESUME_BULLETS.read_text("utf-8"))}
    return [by_id[bullet_id] for bullet_id in ids]


# === Ranking (per-job JD embedding cache) =====================================

_LABELS = ["requirement", "responsibility", "preferred", "context"]
CSV_COLUMNS = ["rank", "job_id", "position", "company", "composite", *_LABELS, "url"]


def _content_hash(obj: Any) -> str:
    """sha256 of a JSON-serializable object (sorted keys → order-independent)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def job_hash(job: Job) -> str:
    """sha256 of a job's labeled-JD content — the JD-cache invalidation key."""
    return _content_hash(job.model_dump(mode="json")["labeled_jd"])


def embed_job(job: Job) -> dict[str, "np.ndarray"]:
    """Embed each label's JD bullets on the GPU; return a label-keyed dict."""
    by_label: dict[str, list[str]] = {}
    for group in job.labeled_jd:
        by_label.setdefault(group.label.value, []).extend(group.text)
    return {label: embed_texts(texts) for label, texts in by_label.items() if texts}


def save_jd_cache(path: Path, jd_vecs: dict[str, "np.ndarray"], content_hash: str) -> None:
    """Save per-label JD vectors plus the cache-invalidation metadata."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, hash=content_hash, model=MODEL_NAME, **jd_vecs)


def load_jd_cache(path: Path, content_hash: str) -> dict[str, "np.ndarray"] | None:
    """Return cached per-label JD vectors, or None if the cache is missing/stale."""
    import numpy as np

    if not path.exists():
        return None
    cached = np.load(path, allow_pickle=True)
    if str(cached["hash"]) != content_hash or str(cached["model"]) != MODEL_NAME:
        return None
    return {key: cached[key] for key in cached.files if key not in ("hash", "model")}


def jd_vecs_for(job: Job) -> tuple[dict[str, "np.ndarray"], bool]:
    """Get a job's JD vectors as (vecs, was_cached); embed + cache on a miss."""
    content_hash = job_hash(job)
    path = JD_CACHE_DIR / f"{job.job_id}.npz"
    cached = load_jd_cache(path, content_hash)
    if cached is not None:
        return cached, True
    jd_vecs = embed_job(job)
    save_jd_cache(path, jd_vecs, content_hash)
    return jd_vecs, False


def rank(
    jobs: list[Job],
    resume_vecs: "np.ndarray",
    resume_bullets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score every job and return CSV rows sorted by composite, descending."""
    from tqdm import tqdm

    rows: list[dict[str, Any]] = []
    embedded = 0
    for job in tqdm(jobs, desc="ranking", unit="job"):
        jd_vecs, was_cached = jd_vecs_for(job)
        if not was_cached:
            embedded += 1
        result = score_job(job, resume_vecs, resume_bullets, jd_vecs_by_label=jd_vecs)
        per = result["per_label"]
        rows.append({
            "job_id": job.job_id,
            "position": job.position,
            "company": job.company,
            "url": job.url,
            "composite": round(result["composite"], 4),
            **{label: (round(per[label], 4) if label in per else "")
               for label in _LABELS},
        })

    rows.sort(key=lambda row: row["composite"], reverse=True)
    for rank_no, row in enumerate(rows, start=1):
        row["rank"] = rank_no
    print(f"Embedded {embedded} job(s), {len(jobs) - embedded} from cache")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ranked rows to a CSV with the fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows: list[dict[str, Any]], limit: int = 10) -> None:
    """Print the top-N ranked jobs as a plain table."""
    top = rows[:limit]
    print(f"\nTop {len(top)} of {len(rows)} jobs by composite score:")
    print(f"  {'#':>2}  {'score':>6}  {'company':<24}  position")
    for row in top:
        print(f"  {row['rank']:>2}  {row['composite']:>6.3f}  "
              f"{row['company'][:24]:<24}  {row['position'][:48]}")


# === CLI ======================================================================


def _run_load(argv: list[str]) -> None:
    source = Path(argv[0]) if argv else latest_eval(DATA_DIR)
    print(f"Loading {source.relative_to(ROOT)}")
    load_jobs(source)


def _run_score(argv: list[str]) -> None:
    ids, resume_vecs = load_or_embed()
    resume_bullets = aligned_resume_bullets(ids)
    source = Path(argv[0]) if argv else latest_eval(DATA_DIR)
    jobs = load_jobs(source)
    if not jobs:
        print("no jobs to score")
        return
    job = jobs[0]
    result = score_job(job, resume_vecs, resume_bullets)
    print(f"\nScored: {job.position} @ {job.company}")
    print(json.dumps(result, indent=2))


def _run_rank(argv: list[str]) -> None:
    ids, resume_vecs = load_or_embed()
    resume_bullets = aligned_resume_bullets(ids)
    source = Path(argv[0]) if argv else latest_eval(DATA_DIR)
    jobs = load_jobs(source)
    rows = rank(jobs, resume_vecs, resume_bullets)
    write_csv(rows, OUT_CSV)
    print_top(rows)
    print(f"\nWrote {len(rows)} ranked jobs -> {OUT_CSV.relative_to(ROOT)}")


_STEPS = {"load": _run_load, "score": _run_score, "rank": _run_rank}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _STEPS:
        print(f"usage: python src/score.py {{{'|'.join(_STEPS)}}} [path]")
        raise SystemExit(1)
    _STEPS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
