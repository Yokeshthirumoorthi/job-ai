"""Résumé side of the pipeline: chunk (and later embed) resume.yaml.

resume.yaml is a typed, version-controlled file the user edits by hand. The
chunk step validates it strictly (pydantic, extra keys forbidden) and flattens
it into a deterministic list of chunks the embedding step will consume.

Each chunk: {id, section, text, meta?}. `text` is what gets embedded; `meta`
(present only for experience/projects/education) is for display only.

CLI:  python src/resume.py chunk     # resume.yaml -> data/resume_bullets.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    # numpy / torch / sentence-transformers are imported lazily inside the embed
    # functions, so `import resume` (used by the chunk path) stays fast.
    import numpy as np
    from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RESUME_PATH = ROOT / "resume.yaml"
OUTPUT_PATH = ROOT / "data" / "resume_bullets.json"

CACHE = ROOT / "cache" / "resume_embeddings.npz"
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024-dim, ~1.3GB, ~1.4GB VRAM
BATCH_SIZE = 32


# === Chunk ====================================================================

# --- Input models — typed view of resume.yaml (extra keys forbidden) ----------


class ExperienceItem(BaseModel):
    """One job in the experience section."""

    model_config = ConfigDict(extra="forbid")

    title: str
    company: str
    dates: str
    highlights: list[str]


class ProjectItem(BaseModel):
    """One entry in the optional projects section."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dates: str | None = None
    highlights: list[str]


class EducationItem(BaseModel):
    """One entry in the optional education section."""

    model_config = ConfigDict(extra="forbid")

    institution: str
    degree: str
    dates: str
    highlights: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    """Typed view of resume.yaml."""

    model_config = ConfigDict(extra="forbid")

    summary: list[str]
    experience: list[ExperienceItem]
    skills: list[str]
    projects: list[ProjectItem] | None = None
    education: list[EducationItem] | None = None
    certifications: list[str] | None = None


# --- Output model -------------------------------------------------------------


class Chunk(BaseModel):
    """One atomic resume item — one element of data/resume_bullets.json."""

    id: str
    section: str
    text: str
    meta: dict[str, str] | None = None   # display-only; omitted when absent


# --- Flatten ------------------------------------------------------------------


def flatten(resume: Resume) -> list[Chunk]:
    """Flatten a resume into chunks in a fixed, deterministic order."""
    chunks: list[Chunk] = []

    def add(section: str, text: str, meta: dict[str, str] | None = None) -> None:
        chunks.append(
            Chunk(id=f"r{len(chunks) + 1:03d}", section=section, text=text, meta=meta)
        )

    for text in resume.summary:
        add("summary", text)

    for job in resume.experience:
        meta = {"title": job.title, "company": job.company, "dates": job.dates}
        for text in job.highlights:
            add("experience", text, meta)

    for text in resume.skills:
        add("skills", text)

    for project in resume.projects or []:
        meta = {"name": project.name}
        if project.dates:
            meta["dates"] = project.dates
        for text in project.highlights:
            add("projects", text, meta)

    for edu in resume.education or []:
        meta = {"institution": edu.institution, "degree": edu.degree, "dates": edu.dates}
        add("education", f"{edu.degree}, {edu.institution} ({edu.dates})", meta)
        for text in edu.highlights:
            add("education", text, meta)

    for text in resume.certifications or []:
        add("certifications", text)

    return chunks


# --- Validation error reporting -----------------------------------------------


def _loc(loc: tuple[object, ...]) -> str:
    """Render a pydantic error location as e.g. 'experience[2].titel'."""
    rendered = ""
    for part in loc:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}" if rendered else str(part)
    return rendered or "<root>"


def _report(source: Path, error: ValidationError) -> None:
    """Print every validation problem with its path, then the actual message."""
    print(f"{source.name} is not a valid resume:", file=sys.stderr)
    for err in error.errors():
        print(f"  {_loc(err['loc'])}: {err['msg']}", file=sys.stderr)


# --- IO -----------------------------------------------------------------------


def load_resume(source: Path) -> Resume:
    """Read and strictly validate a resume YAML file (never yaml.load)."""
    return Resume.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))


def save_chunks(chunks: list[Chunk], out: Path) -> None:
    """Write chunks as a UTF-8, indent=2 JSON array (stable byte-for-byte)."""
    out.parent.mkdir(exist_ok=True)
    payload = [chunk.model_dump(exclude_none=True) for chunk in chunks]
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# === Embed ====================================================================
#
# Embed every resume bullet once and cache the result to
# cache/resume_embeddings.npz, keyed by a sha256 of the bullets file and the
# model name, so re-runs are instant until either changes.
#
#   npz keys: ids (N,) str | vecs (N, 1024) float32 | hash str | model str

_model: "SentenceTransformer | None" = None  # singleton, loaded by get_model()


def get_model() -> "SentenceTransformer":
    """Load the embedding model onto the GPU once; reuse the handle thereafter.

    Aborts loudly rather than falling back to CPU — silent CPU fallback (wrong
    torch build) is the classic failure: it works, just ~50x slower.
    """
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — torch is likely the CPU build; "
                "reinstall from the PyTorch CUDA index (see pyproject.toml)"
            )
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"model loaded on {device}, not the GPU — aborting")
        print(f"Loaded {MODEL_NAME} on {torch.cuda.get_device_name(0)}")
        _model = model
    return _model


def embed_texts(texts: list[str]) -> "np.ndarray":
    """Encode texts to normalized float32 vectors via the singleton model."""
    import numpy as np

    vecs = get_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    return vecs.astype(np.float32)


def file_hash(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_is_fresh(cache_path: Path, input_hash: str, model_name: str) -> bool:
    """True if the cache exists and was built from this input and this model."""
    import numpy as np

    if not cache_path.exists():
        return False
    cached = np.load(cache_path, allow_pickle=True)
    return str(cached["hash"]) == input_hash and str(cached["model"]) == model_name


def write_cache(
    cache_path: Path,
    ids: list[str],
    vecs: "np.ndarray",
    input_hash: str,
    model_name: str,
) -> None:
    """Save embeddings plus the fingerprints used to invalidate them."""
    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        ids=np.array(ids),
        vecs=vecs.astype(np.float32),
        hash=input_hash,
        model=model_name,
    )


def read_cache(cache_path: Path) -> "tuple[list[str], np.ndarray]":
    """Load (ids, vecs) from a cache file, ids as a plain list of str."""
    import numpy as np

    cached = np.load(cache_path, allow_pickle=True)
    return cached["ids"].tolist(), cached["vecs"]


def load_or_embed() -> "tuple[list[str], np.ndarray]":
    """Return (ids, vecs), embedding on the GPU only when the cache is stale.

    Input is the chunk step's output (data/resume_bullets.json) — run `chunk`
    first.
    """
    input_hash = file_hash(OUTPUT_PATH)
    if cache_is_fresh(CACHE, input_hash, MODEL_NAME):
        ids, vecs = read_cache(CACHE)
        print(f"Cache hit — {len(ids)} embeddings from {CACHE.relative_to(ROOT)}")
        return ids, vecs

    bullets = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    ids = [bullet["id"] for bullet in bullets]
    texts = [bullet["text"] for bullet in bullets]
    vecs = embed_texts(texts)
    write_cache(CACHE, ids, vecs, input_hash, MODEL_NAME)
    print(f"Embedded and cached {len(ids)} bullets -> {CACHE.relative_to(ROOT)}")
    return ids, vecs


# === CLI ======================================================================


def _run_chunk(argv: list[str]) -> None:
    source = Path(argv[0]) if argv else RESUME_PATH
    try:
        resume = load_resume(source)
    except FileNotFoundError:
        print(f"resume not found: {source}", file=sys.stderr)
        raise SystemExit(1)
    except ValidationError as error:
        _report(source, error)
        raise SystemExit(1)

    chunks = flatten(resume)
    save_chunks(chunks, OUTPUT_PATH)
    print(f"Wrote {len(chunks)} chunks -> {OUTPUT_PATH.relative_to(ROOT)}")


def _run_embed(_argv: list[str]) -> None:
    ids, vecs = load_or_embed()
    print(f"{len(ids)} embeddings, shape {vecs.shape}, dtype {vecs.dtype}")


_STEPS = {"chunk": _run_chunk, "embed": _run_embed}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _STEPS:
        print(f"usage: python src/resume.py {{{'|'.join(_STEPS)}}} [path]")
        raise SystemExit(1)
    _STEPS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
