"""
CASP-compliant HTTP server: FASTA in → PDB out.

**Default pipeline (Lean + ribosome tunnel):** HQIV Lean–aligned co-translational folding
(`fold_lean_ribosome_tunnel`: bulk ε_r(T), pH screening, tunnel extrusion, then 3D EM-field relax
+ discrete tree-torque — not the legacy L-BFGS post-extrusion HKE unless `CASP_LEAN_POST_EXTRUSION_MODE=hke`).
**Ligands:** single-chain folds include them in `fold_lean_ribosome_tunnel`; multi-chain jobs fold
each chain with Lean, merge with chain offsets, then run the same 6-DOF ligand refinement vs the
**full merged backbone** (screening-matched `em_scale`) and append HETATM. Fast-pass does the same
with `quick=True` budgets.

**Legacy pipeline:** Set `CASP_LEGACY_HKE_PIPELINE=1` to restore hierarchical HKE + funnel +
optional tree-torque / extrude cycles.

POST /predict with body = FASTA or form/JSON (sequence=, title=, email=).
If "email" param is set and SMTP is configured, also sends PDB by email.
GET /health → 200 OK.

**PDB model line:** ``REMARK 999 HQIV-QConSi-<n>-step`` plus ``MODEL <n>`` (not plain ``MODEL 1``). Set ``CASP_PDB_QCONSI_STEP`` for ``<n>`` (default ``1``).

Env: SMTP_* for email; SMTP_CC_TO to CC when not recipient; CASP_OUTPUT_DIR (default ./casp_results)
for pending/ and outputs/; USE_FAST_PREDICT=1 to skip main pass (fast PDB only); PREDICTION_TIMEOUT_SEC
(default 3600); CASP_MAX_CONCURRENT_JOBS (default 2); CASP_KNOWN_TARGETS_CACHE (optional) for known
targets + experimental PDBs; CASP_ROUND (default CASP16); CASP_TARGET_ARCHIVE_ROOT (optional) for
Sunday archival of the known-targets cache; CASP_DISABLE_SUNDAY_ARCHIVE=1 to skip.
Thermal refinement (Lean post-tunnel): CASP_LEAN_DISCRETE_METROPOLIS=1 (Metropolis at temperature_k),
CASP_LEAN_DISCRETE_SEED (optional int), CASP_LEAN_POST_ANNEAL=1 (short multi-K Metropolis cool-down vs single pass),
CASP_LEAN_POST_ANNEAL_SCHEDULE=348,330,318,310 (optional comma-separated K, ≥2 values),
CASP_LEAN_POST_LANGEVIN_STEPS, CASP_LEAN_LANGEVIN_NOISE_FRAC.
In-tunnel thermal gradient: CASP_LEAN_TUNNEL_THERMAL_STEPS (default 0), CASP_LEAN_TUNNEL_THERMAL_NOISE_FRAC, CASP_LEAN_TUNNEL_THERMAL_SEED (optional).
**HQIV-native OSHoracle** (Lean ``OSHoracleHQIVNative``): enabled by default after EM/tree-torque; set ``CASP_LEAN_OSH_HQIV_NATIVE=0`` to disable. Optional: ``CASP_LEAN_OSH_N_ITER``, ``CASP_LEAN_OSH_STEP_SIZE``, ``CASP_LEAN_OSH_ANSATZ_DEPTH``, ``CASP_LEAN_OSH_GATE_MIX``, ``CASP_LEAN_OSH_HQIV_REFERENCE_M``.

**Ligand refinement (Lean ``ProteinQCRefinement`` soft clash):** default ``CASP_LEAN_LIGAND_REFINEMENT_MODE=lean_qc``; set to ``horizon`` for legacy ``grad_full`` rigid-body steps. Optional: ``CASP_LEAN_QC_SOFT_CLASH_SIGMA`` (Å), ``CASP_LEAN_QC_CLASH_WEIGHT``.
**Ligand rigid-body steps:** main jobs use ``CASP_LEAN_LIGAND_REFINE_STEPS`` (default **150**). Fast-pass uses ``CASP_LEAN_FAST_LIGAND_REFINE_STEPS`` (default **50**).

Run from repo root: gunicorn -w 1 -b 127.0.0.1:8050 casp_server:app
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr, make_msgid, formatdate

from flask import Flask, request, Response, send_file
import html as _html

# src layout: package lives under ./src (editable install also works)
_root = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from horizon_physics.proteins.casp_submission import _parse_fasta
from horizon_physics.proteins import (
    full_chain_to_pdb,
    full_chain_to_pdb_complex,
    minimize_full_chain,
)
from horizon_physics.proteins.full_protein_minimizer import (
    pdb_hetatm_lines_for_ligands,
    refine_ligands_on_multichain_results,
)
from horizon_physics.proteins.ligands import parse_ligands
from horizon_physics.proteins.hqiv_lean_folding import PHYSIOLOGICAL_PH
from horizon_physics.proteins.lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel
from horizon_physics.proteins.pdb_hqiv_header import hqiv_qconsi_empty_pdb_block, hqiv_qconsi_model_lines
from horizon_physics.proteins.hierarchical import (
    minimize_full_chain_hierarchical,
    hierarchical_result_for_pdb,
)
from horizon_physics.proteins.assembly_dock import (
    run_two_chain_assembly,
    run_two_chain_assembly_hke,
    complex_to_single_chain_result,
)
try:
    from horizon_physics.proteins.temperature_path_search import run_discrete_refinement
except Exception:
    run_discrete_refinement = None  # pyhqiv or deps missing; skip tree-torque

try:
    from horizon_physics.proteins.extrude_hke_treetorque import (
        extrude_hke_treetorque_cycle,
        assembly_hke_treetorque_cycle_two_chains,
    )
except Exception:
    extrude_hke_treetorque_cycle = None
    assembly_hke_treetorque_cycle_two_chains = None

try:
    from horizon_physics.proteins.casp_targets import (
        fetch_known_targets,
        get_target_for_sequence,
        get_target_for_sequences,
        ensure_experimental_ref,
    )
except Exception:
    fetch_known_targets = None
    get_target_for_sequence = None
    get_target_for_sequences = None
    ensure_experimental_ref = None

# Fast path (geometric-only, no minimization): for testing or when USE_FAST_PREDICT=1
try:
    from horizon_physics.proteins import hqiv_predict_structure, hqiv_predict_structure_assembly
except Exception:
    hqiv_predict_structure = None
    hqiv_predict_structure_assembly = None

USE_FAST_PREDICT = os.environ.get("USE_FAST_PREDICT", "").strip().lower() in ("1", "true", "yes")
# CAMEO custom free-parameter key for ligand input (must match registration).
# Multiple ligands: LIGAND_KEY1=code1&LIGAND_KEY2=code2&... (key=value&key2=value2 style).
LIGAND_KEY = os.environ.get("CASP_LIGAND_KEY", "ligand")
FUNNEL_RADIUS = float(os.environ.get("FUNNEL_RADIUS", "10.0"))
FUNNEL_RADIUS_EXIT = float(os.environ.get("FUNNEL_RADIUS_EXIT", "20.0"))
# HKE uses finite-difference gradient (2 * n_dofs evals per step; ~572/step for 141 res). Funnel is on; bottleneck is FD, not funnel.
# Main pipeline: do ONE short minimizing pass in HKE, then hand off to tree-torque (faster) to see if anything changed.
HKE_ONE_PASS_ITER = (
    int(os.environ.get("HKE_ONE_PASS_S1", "2")),
    int(os.environ.get("HKE_ONE_PASS_S2", "3")),
    int(os.environ.get("HKE_ONE_PASS_S3", "5")),
)
# Legacy / fast-pass: can use higher iters for quick geometric fold (Phase 1).
HKE_MAX_ITER = (
    int(os.environ.get("HKE_MAX_ITER_S1", "15")),
    int(os.environ.get("HKE_MAX_ITER_S2", "25")),
    int(os.environ.get("HKE_MAX_ITER_S3", "50")),
)
# Stop prediction after this many seconds; write current PDB, send email, then pick up next job.
PREDICTION_TIMEOUT_SEC = int(os.environ.get("PREDICTION_TIMEOUT_SEC", "3600"))  # 1 hour
# Max number of prediction jobs running at once (subprocesses). Prevents 70+ processes from starving each other.
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("CASP_MAX_CONCURRENT_JOBS", "2")))
_job_concurrency_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
DISABLE_PENDING_STARTUP = os.environ.get("CASP_DISABLE_PENDING_STARTUP", "").strip().lower() in ("1", "true", "yes")

# Optional: cache dir for CASP known targets and experimental PDBs (predictioncenter.org). When set,
# we match request sequences to known targets and fetch experimental refs so we're on equal footing.
CASP_KNOWN_TARGETS_CACHE = (os.environ.get("CASP_KNOWN_TARGETS_CACHE") or "").strip() or None
CASP_ROUND = os.environ.get("CASP_ROUND", "CASP16")
# Default: Lean tunnel pipeline. Set CASP_LEGACY_HKE_PIPELINE=1 for hierarchical HKE + tree-torque stack.
CASP_LEGACY_HKE_PIPELINE = os.environ.get("CASP_LEGACY_HKE_PIPELINE", "").strip().lower() in ("1", "true", "yes")
CASP_TARGET_ARCHIVE_ROOT = (os.environ.get("CASP_TARGET_ARCHIVE_ROOT") or "").strip() or None
CASP_DISABLE_SUNDAY_ARCHIVE = os.environ.get("CASP_DISABLE_SUNDAY_ARCHIVE", "").strip().lower() in ("1", "true", "yes")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB max FASTA

# Result persistence: pending/ (request .txt, then .pdb when done) → on success move both to outputs/
_output_base = os.environ.get("CASP_OUTPUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "casp_results")
PENDING_DIR = os.path.join(_output_base, "pending")
OUTPUTS_DIR = os.path.join(_output_base, "outputs")


def _ensure_output_dirs() -> None:
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _use_lean_casp_pipeline() -> bool:
    """True = default Lean ribosome-tunnel pipeline; False = CASP_LEGACY_HKE_PIPELINE."""
    return not CASP_LEGACY_HKE_PIPELINE


def _lean_fold_env_kwargs(quick: bool) -> dict:
    """Environment-driven parameters for fold_lean_ribosome_tunnel (CASP tuning)."""
    def _i(key: str, default: int) -> int:
        return int(os.environ.get(key, str(default)))

    def _f(key: str, default: float) -> float:
        return float(os.environ.get(key, str(default)))

    def _optional_i(key: str) -> int | None:
        v = os.environ.get(key)
        if v is None or str(v).strip() == "":
            return None
        return int(v)

    ph_def = str(PHYSIOLOGICAL_PH)
    post_mode = (os.environ.get("CASP_LEAN_POST_EXTRUSION_MODE") or "em_treetorque").strip().lower()
    if post_mode not in ("none", "hke", "em_treetorque"):
        post_mode = "em_treetorque"
    disc_seed: int | None = None
    _seed_s = os.environ.get("CASP_LEAN_DISCRETE_SEED", "").strip()
    if _seed_s:
        try:
            disc_seed = int(_seed_s)
        except ValueError:
            disc_seed = None
    if quick:
        post_rounds = _i("CASP_LEAN_FAST_POST_EXTRUSION_MAX_ROUNDS", 12)
        fpc = _i("CASP_LEAN_FAST_FAST_PASS_STEPS", 2)
        mpc = _i("CASP_LEAN_FAST_MIN_PASS_ITER", 5)
    else:
        post_rounds = _i("CASP_LEAN_POST_EXTRUSION_MAX_ROUNDS", 32)
        fpc = _i("CASP_LEAN_FAST_PASS_STEPS_PER_CONNECTION", 0)
        mpc = _i("CASP_LEAN_MIN_PASS_ITER_PER_CONNECTION", 0)

    def _comma_float_tuple(key: str) -> tuple[float, ...] | None:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        return tuple(float(p) for p in parts)

    post_anneal = os.environ.get("CASP_LEAN_POST_ANNEAL", "").strip().lower() in ("1", "true", "yes")
    anneal_sched = _comma_float_tuple("CASP_LEAN_POST_ANNEAL_SCHEDULE")

    _osh_raw = os.environ.get("CASP_LEAN_OSH_HQIV_NATIVE", "").strip().lower()
    post_extrusion_osh_hqiv_native = False if _osh_raw in ("0", "false", "no") else True

    tunnel_th_seed: int | None = None
    _tth_seed_s = os.environ.get("CASP_LEAN_TUNNEL_THERMAL_SEED", "").strip()
    if _tth_seed_s:
        try:
            tunnel_th_seed = int(_tth_seed_s)
        except ValueError:
            tunnel_th_seed = None

    # Ligand 6-DOF refinement: full CASP jobs get a generous default (not capped at 40).
    if quick:
        ligand_refine_steps = _i("CASP_LEAN_FAST_LIGAND_REFINE_STEPS", 50)
    else:
        ligand_refine_steps = _i("CASP_LEAN_LIGAND_REFINE_STEPS", 150)

    return {
        "temperature_k": _f("CASP_LEAN_TEMPERATURE_K", 310.0),
        "ph": _f("CASP_LEAN_PH", float(ph_def)),
        "kappa_dihedral": _f("CASP_LEAN_KAPPA_DIHEDRAL", 0.01),
        "post_extrusion_refine_mode": post_mode,
        "post_extrusion_em_max_steps": _optional_i("CASP_LEAN_POST_EM_MAX_STEPS"),
        "post_extrusion_treetorque_phases": _i("CASP_LEAN_POST_TT_PHASES", 8),
        "post_extrusion_treetorque_n_steps": _i("CASP_LEAN_POST_TT_N_STEPS", 200),
        "post_extrusion_discrete_metropolis": os.environ.get("CASP_LEAN_DISCRETE_METROPOLIS", "")
        .strip()
        .lower()
        in ("1", "true", "yes"),
        "post_extrusion_discrete_seed": disc_seed,
        "post_extrusion_anneal": post_anneal,
        **({"post_extrusion_anneal_schedule_k": anneal_sched} if anneal_sched is not None else {}),
        "tunnel_thermal_gradient_steps": _i("CASP_LEAN_TUNNEL_THERMAL_STEPS", 0),
        "tunnel_thermal_noise_fraction": _f("CASP_LEAN_TUNNEL_THERMAL_NOISE_FRAC", 0.2),
        **({"tunnel_thermal_seed": tunnel_th_seed} if tunnel_th_seed is not None else {}),
        "post_extrusion_langevin_steps": _i("CASP_LEAN_POST_LANGEVIN_STEPS", 0),
        "post_extrusion_langevin_noise_fraction": _f("CASP_LEAN_LANGEVIN_NOISE_FRAC", 0.2),
        "post_extrusion_max_rounds": post_rounds,
        "post_extrusion_osh_hqiv_native": post_extrusion_osh_hqiv_native,
        "post_extrusion_osh_n_iter": _i("CASP_LEAN_OSH_N_ITER", 120),
        "post_extrusion_osh_step_size": _f("CASP_LEAN_OSH_STEP_SIZE", 0.03),
        "post_extrusion_osh_ansatz_depth": _i("CASP_LEAN_OSH_ANSATZ_DEPTH", 2),
        "post_extrusion_osh_gate_mix": _f("CASP_LEAN_OSH_GATE_MIX", 0.55),
        "post_extrusion_osh_hqiv_reference_m": _i("CASP_LEAN_OSH_HQIV_REFERENCE_M", 4),
        "fast_pass_steps_per_connection": fpc,
        "min_pass_iter_per_connection": mpc,
        "hbond_weight": _f("CASP_LEAN_HBOND_WEIGHT", 0.0),
        "hbond_shell_m": _i("CASP_LEAN_HBOND_SHELL_M", 3),
        "ligand_refine_steps": ligand_refine_steps,
        "ligand_refinement_mode": (
            (lambda m: m if m in ("lean_qc", "horizon") else "lean_qc")(
                (os.environ.get("CASP_LEAN_LIGAND_REFINEMENT_MODE") or "lean_qc").strip().lower()
            )
        ),
        "qc_soft_clash_sigma": _f("CASP_LEAN_QC_SOFT_CLASH_SIGMA", 3.0),
        "qc_clash_weight": _f("CASP_LEAN_QC_CLASH_WEIGHT", 1.0),
    }


def _lean_grad_full_kwargs_for_ligand_refine(*, quick: bool) -> dict:
    """EM screening for multichain ligand refinement (same bulk water + pH as fold_lean_ribosome_tunnel)."""
    from horizon_physics.proteins.hqiv_lean_folding import (
        em_scale_aqueous,
        epsilon_r_water,
        ph_em_scale_delta,
    )

    kw = _lean_fold_env_kwargs(quick)
    t_k = float(kw["temperature_k"])
    ph = float(kw["ph"])
    er = epsilon_r_water(t_k)
    em_scale = em_scale_aqueous(t_k, epsilon_r=er) * ph_em_scale_delta(ph)
    return {"em_scale": float(em_scale), "hbond_weight": 0.0}


def _maybe_archive_casp_targets_cache_on_sunday() -> None:
    """
    On each Sunday, move the contents of CASP_KNOWN_TARGETS_CACHE into
    CASP_TARGET_ARCHIVE_ROOT (or ../archived_casp_targets under the cache parent)
    so the next fetch pulls fresh targets from the prediction center.
    """
    global _known_casp_targets
    if CASP_DISABLE_SUNDAY_ARCHIVE or not CASP_KNOWN_TARGETS_CACHE:
        return
    cache = CASP_KNOWN_TARGETS_CACHE
    if not os.path.isdir(cache):
        return
    import datetime

    if datetime.date.today().weekday() != 6:  # Sunday
        return
    today = datetime.date.today().isoformat()
    marker = os.path.join(cache, ".sunday_archive_done")
    try:
        if os.path.isfile(marker):
            with open(marker, encoding="utf-8") as f:
                if f.read().strip() == today:
                    return
    except Exception:
        pass
    root = CASP_TARGET_ARCHIVE_ROOT or os.path.join(
        os.path.dirname(os.path.abspath(cache)), "archived_casp_targets"
    )
    dest = os.path.join(root, f"{today}_targets")
    try:
        os.makedirs(dest, exist_ok=True)
        moved_any = False
        for name in os.listdir(cache):
            if name.startswith(".sunday_archive"):
                continue
            src = os.path.join(cache, name)
            dst = os.path.join(dest, name)
            try:
                shutil.move(src, dst)
                moved_any = True
            except Exception as e:
                app.logger.warning("Sunday archive: could not move %s: %s", src, e)
        if moved_any:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(today)
            _known_casp_targets = None
            app.logger.info(
                "Sunday archive: moved CASP targets cache to %s (fresh fetch on next use).",
                dest,
            )
    except Exception as e:
        app.logger.warning("Sunday archive failed: %s", e)


# Lazy-loaded list of known CASP targets (when CASP_KNOWN_TARGETS_CACHE is set)
_known_casp_targets: list | None = None


def _get_known_casp_targets() -> list:
    """Load known CASP targets once (sequences + PDB codes) when cache dir is set."""
    global _known_casp_targets
    if _known_casp_targets is not None:
        return _known_casp_targets
    if not CASP_KNOWN_TARGETS_CACHE or fetch_known_targets is None:
        _known_casp_targets = []
        return _known_casp_targets
    _maybe_archive_casp_targets_cache_on_sunday()
    try:
        _known_casp_targets = fetch_known_targets(
            casp_round=CASP_ROUND,
            cache_dir=CASP_KNOWN_TARGETS_CACHE,
            fill_pdb_codes=True,
        )
        app.logger.info(
            "Loaded %d known CASP targets from predictioncenter.org (%s)",
            len(_known_casp_targets),
            CASP_ROUND,
        )
    except Exception as e:
        app.logger.warning("Failed to load known CASP targets: %s", e)
        _known_casp_targets = []
    return _known_casp_targets


def _experimental_ref_path_for_job(seqs: list[str], base: str) -> str | None:
    """
    If request sequences match a known CASP target with experimental structure,
    fetch it and return path to copy to outputs (so we're on equal footing).
    Returns path to cached experimental PDB, or None.
    """
    if not CASP_KNOWN_TARGETS_CACHE or ensure_experimental_ref is None:
        return None
    known = _get_known_casp_targets()
    if not known:
        return None
    target = None
    if len(seqs) == 1:
        hit = get_target_for_sequence(seqs[0], known) if get_target_for_sequence else None
        if hit:
            target = hit[0]
    elif len(seqs) >= 2:
        hit = get_target_for_sequences(seqs[:2], known) if get_target_for_sequences else None
        if hit:
            target = hit[0]
    if not target or not target.pdb_code:
        return None
    path = ensure_experimental_ref(target, CASP_KNOWN_TARGETS_CACHE)
    return path


def _gather_ligand_str(request) -> str | None:
    """
    Gather ligand input from request. Supports:
    - Repeated key: LIGAND_KEY=blah&LIGAND_KEY=blah (form getlist order).
    - Numbered keys: LIGAND_KEY1=code1&LIGAND_KEY2=code2&...
    - JSON: LIGAND_KEY as string or array; plus LIGAND_KEY1, LIGAND_KEY2, ...
    Returns newline-joined string for parse_ligands, or None if none present.
    """
    values: list[str] = []

    if request.is_json:
        data = request.get_json(silent=True) or {}
        val = data.get(LIGAND_KEY)
        if isinstance(val, list):
            values = [str(x).strip() for x in val if x is not None and str(x).strip()]
        elif val is not None and str(val).strip():
            values = [str(val).strip()]
        # Numbered keys in order
        numbered = [
            (int(k[len(LIGAND_KEY):]), k)
            for k in data
            if isinstance(k, str) and k.startswith(LIGAND_KEY) and k[len(LIGAND_KEY):].isdigit()
        ]
        numbered.sort(key=lambda x: x[0])
        for _, k in numbered:
            v = data.get(k)
            if v is not None and str(v).strip():
                values.append(str(v).strip())
    elif request.form:
        # Repeated LIGAND_KEY=...&LIGAND_KEY=... (order preserved by getlist)
        values = [s.strip() for s in request.form.getlist(LIGAND_KEY) if s and str(s).strip()]
        # Then numbered LIGAND_KEY1, LIGAND_KEY2, ...
        numbered = [
            (int(k[len(LIGAND_KEY):]), k)
            for k in request.form
            if isinstance(k, str) and k.startswith(LIGAND_KEY) and k[len(LIGAND_KEY):].isdigit()
        ]
        numbered.sort(key=lambda x: x[0])
        for _, k in numbered:
            v = request.form.get(k)
            if v is not None and str(v).strip():
                values.append(str(v).strip())

    if not values:
        return None
    return "\n".join(values)


def _job_id() -> str:
    return f"{int(time.time())}_{random.randint(1000, 9999)}"


def _sanitize_email_for_fs(email: str) -> str:
    """Filesystem-safe string from email (for finding jobs by recipient); not exposed in /status."""
    if not email or not isinstance(email, str):
        return ""
    s = email.strip().lower()
    s = s.replace("@", "_at_").replace(".", "_")
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return s[:80].strip("_") or "no_email"


def _pending_base_name(job_id: str, to_email: str | None) -> str:
    """Base name for pending/output files: {sanitized_email}__{job_id} when email set, else job_id. Not exposed in /status."""
    if to_email and _sanitize_email_for_fs(to_email):
        return f"{_sanitize_email_for_fs(to_email)}__{job_id}"
    return job_id


def _base_to_job_id(base: str) -> str:
    """Extract job_id from file base name (handles both email__job_id and legacy job_id)."""
    return base.split("__", 1)[1] if "__" in base else base


def _find_base_for_job_id(job_id: str, directory: str) -> str | None:
    """Find file base in directory whose _base_to_job_id(base) == job_id (e.g. from .request.json)."""
    if not os.path.isdir(directory):
        return None
    for name in os.listdir(directory):
        if name.endswith(".request.json"):
            base = name[: -len(".request.json")]
            if _base_to_job_id(base) == job_id:
                return base
        if name.endswith(".txt") and not name.endswith(".failed.txt"):
            base = name[:-4]
            if _base_to_job_id(base) == job_id:
                return base
    return None


MAX_ATTEMPTS = 2  # After this many attempts (initial + retries on restart), send failure email instead of retrying


def _write_pending_txt(job_id: str, job_title: str | None, to_email: str | None, num_sequences: int, seq_lengths: list[int], attempts: int = 1) -> None:
    """Write pending/{base}.txt with POST summary and attempts count. base = email__job_id when email set."""
    _ensure_output_dirs()
    base = _pending_base_name(job_id, to_email)
    path = os.path.join(PENDING_DIR, f"{base}.txt")
    lines = [
        f"attempts={attempts}",
        f"title={job_title or ''}",
        f"email={to_email or ''}",
        f"num_sequences={num_sequences}",
        f"lengths={seq_lengths}",
        f"received_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "POST /predict with sequence= (or multiple for assembly), title=, email=.",
        "When prediction completes, this file, .request.json, and .pdb are moved to outputs/.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_pending_request(
    job_id: str,
    sequences: list[str],
    job_title: str | None,
    to_email: str | None,
    ligand_str: str | None = None,
) -> None:
    """Write pending/{base}.request.json so the job can be retried on restart. base = email__job_id when email set."""
    _ensure_output_dirs()
    base = _pending_base_name(job_id, to_email)
    path = os.path.join(PENDING_DIR, f"{base}.request.json")
    payload = {"sequences": sequences, "title": job_title, "email": to_email}
    if ligand_str is not None and ligand_str.strip():
        payload[LIGAND_KEY] = ligand_str.strip()
    with open(path, "w") as f:
        json.dump(payload, f)


# PDB sanity: reject coordinates outside this range (Å) or non-finite
PDB_COORD_MAX = 9999.0


def _backbone_sanity_check(backbone_atoms: list, job_context: str = "") -> None:
    """Raise ValueError if any backbone coordinate is non-finite or |coord| > PDB_COORD_MAX."""
    for i, (name, xyz) in enumerate(backbone_atoms):
        try:
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        except (IndexError, TypeError, ValueError):
            raise ValueError(f"Backbone has invalid coordinates at atom {i} ({name}) {job_context}")
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise ValueError(f"Backbone has non-finite coordinates at atom {i} ({name}) {job_context}")
        if abs(x) > PDB_COORD_MAX or abs(y) > PDB_COORD_MAX or abs(z) > PDB_COORD_MAX:
            raise ValueError(
                f"Backbone coordinates out of range at atom {i} ({name}): |x|,|y|,|z| <= {PDB_COORD_MAX} Å {job_context}"
            )


def _pdb_sanity_check(pdb_content: str) -> tuple[bool, str]:
    """Return (True, '') if PDB coordinates are finite and in [-PDB_COORD_MAX, PDB_COORD_MAX]; else (False, reason)."""
    # Extract x,y,z from standard PDB columns 31-54, or fall back to the last
    # three numbers on overflowed/non-standard lines.
    num_pattern = re.compile(r"-?\d+\.?\d*")
    for line in pdb_content.splitlines():
        if not line.startswith("ATOM  ") and not line.startswith("HETATM"):
            continue
        # Try standard fixed columns first (0-based slices 30:38, 38:46, 46:54).
        if len(line) >= 54:
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                x = y = z = None
        else:
            x = y = z = None
        if x is None or y is None or z is None:
            # Overflowed format: take last three numbers on the line
            nums = num_pattern.findall(line)
            if len(nums) < 3:
                continue
            try:
                x, y, z = float(nums[-3]), float(nums[-2]), float(nums[-1])
            except ValueError:
                return False, "Non-numeric coordinates"
        if not (abs(x) <= PDB_COORD_MAX and abs(y) <= PDB_COORD_MAX and abs(z) <= PDB_COORD_MAX):
            return False, f"Coordinates out of range (|x|,|y|,|z| <= {PDB_COORD_MAX} Å)"
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False, "Non-finite coordinates"
    return True, ""


def _prepend_pdb_model_line(pdb_content: str) -> str:
    """Prepend REMARK with model name for submission (Physical Relational Ontology)."""
    remark = "REMARK   Model: Physical Relational Ontology\n"
    if pdb_content.strip().startswith("REMARK"):
        return remark + pdb_content
    return remark + pdb_content


def _move_to_outputs(base: str, pdb_content: str) -> None:
    """Write pending/{base}.pdb then move .txt, .pdb, and .request.json to outputs/. base = email__job_id or job_id."""
    _ensure_output_dirs()
    pdb_path = os.path.join(PENDING_DIR, f"{base}.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    for name in (f"{base}.txt", f"{base}.pdb", f"{base}.request.json"):
        src = os.path.join(PENDING_DIR, name)
        dst = os.path.join(OUTPUTS_DIR, name)
        if os.path.isfile(src):
            shutil.move(src, dst)


def _move_to_outputs_assembly(
    base: str,
    pdb_complex: str,
) -> None:
    """Write the single combined PDB (chain A + B) and move to outputs/. Same layout as single-chain: {base}.pdb."""
    _move_to_outputs(base, pdb_complex)

# Optional SMTP: send PDB to request's "email" address when set
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
SMTP_FROM_RAW = os.environ.get("SMTP_FROM") or (SMTP_USER if SMTP_USER else "")
SMTP_FROM_DOMAIN = (os.environ.get("SMTP_FROM_DOMAIN") or "").strip()  # e.g. disregardfiat.tech — use for From when Gmail rejects SMTP host–derived domain
SMTP_CC_TO = (os.environ.get("SMTP_CC_TO") or os.environ.get("cc_to") or "").strip()  # CC this address on result emails when not the recipient (e.g. to monitor CAMEO)
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1").strip().lower() in ("1", "true", "yes")


def _smtp_from_address() -> str:
    """Return RFC 5322–compliant From address (user@domain). Gmail rejects some domains; set SMTP_FROM or SMTP_FROM_DOMAIN to use a compliant address."""
    addr = (SMTP_FROM_RAW or "").strip()
    if re.match(r"^[^@]+@[^@]+\.[^@]+", addr):
        return addr
    if SMTP_USER and "@" in SMTP_USER:
        return SMTP_USER
    # Prefer explicit domain (Gmail-friendly, e.g. disregardfiat.tech)
    domain = SMTP_FROM_DOMAIN
    if not domain and SMTP_HOST:
        # Derive from SMTP host: mail.comodomodo.com.py -> comodomodo.com.py
        domain = (SMTP_HOST or "").strip()
        if domain.startswith("mail."):
            domain = domain[5:]
        elif "." in domain:
            domain = domain.split(".", 1)[1]
    if SMTP_USER and domain:
        return f"{SMTP_USER}@{domain}"
    if domain:
        return f"noreply@{domain}"
    return "noreply@localhost"


def _get_email_and_title() -> tuple[str | None, str | None]:
    """Extract email and title from request (form or JSON)."""
    email, title = None, None
    if request.form:
        email = request.form.get("email") or request.form.get("email_to") or request.form.get("results_email")
        title = request.form.get("title")
    if (email is None or title is None) and request.is_json:
        data = request.get_json(silent=True) or {}
        if email is None:
            email = data.get("email") or data.get("email_to") or data.get("results_email")
        if title is None:
            title = data.get("title")
    if email:
        email = email.strip()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            email = None
    return email or None, (title.strip() if title and title.strip() else None)


def _send_pdb_email(to_email: str, pdb: str, title: str | None) -> None:
    """Send PDB as email attachment. No-op if SMTP not configured. Logs but does not raise on failure."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction: {title}" if title else "HQIV structure prediction"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        to_addr_lower = to_email.strip().lower()
        if cc_addr != to_addr_lower:
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText("PDB model attached (CASP format).", "plain"))
    part = MIMEText(pdb, "plain")
    part.add_header("Content-Disposition", "attachment", filename="model.pdb")
    msg.attach(part)
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP send failed: %s", e)


def _send_pdb_email_custom(
    to_email: str,
    pdb: str,
    title: str | None,
    filename: str,
) -> None:
    """Send a single PDB with a custom attachment filename."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction: {title}" if title else "HQIV structure prediction"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        to_addr_lower = to_email.strip().lower()
        if cc_addr != to_addr_lower:
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText("PDB model attached (CASP format).", "plain"))
    part = MIMEText(pdb, "plain")
    part.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(part)
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP send failed: %s", e)


def _send_pdb_email_multi(
    to_email: str,
    models: list[tuple[str, str]],
    title: str | None,
    body_extra: str | None = None,
) -> None:
    """Send multiple PDB models as separate attachments. body_extra is appended to the text body (e.g. time-limit note)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction: {title}" if title else "HQIV structure prediction"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        to_addr_lower = to_email.strip().lower()
        if cc_addr != to_addr_lower:
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    body = "PDB models attached (CASP format)."
    if body_extra:
        body += "\n\n" + body_extra
    msg.attach(MIMEText(body, "plain"))
    for filename, pdb in models:
        part = MIMEText(pdb, "plain")
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP send failed: %s", e)


def _send_assembly_email(
    to_email: str,
    pdb_complex: str,
    job_title: str | None,
) -> None:
    """Send the single combined PDB (chain A+B) as one attachment, same as single-chain."""
    _send_pdb_email(to_email, pdb_complex, job_title or "2-chain assembly")


def _send_job_failure_email(to_email: str, job_id: str, job_title: str | None, error_message: str) -> None:
    """Send email on any single failure (to requester and CC)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction failed (job {job_id})"
    body = f"Job {job_id} (title={job_title or 'n/a'}) failed.\n\nError: {error_message}"
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    msg["To"] = to_email
    recipients = [to_email]
    if SMTP_CC_TO and re.match(r"[^@]+@[^@]+\.[^@]+", SMTP_CC_TO):
        cc_addr = SMTP_CC_TO.strip().lower()
        if cc_addr != (to_email or "").strip().lower():
            msg["Cc"] = SMTP_CC_TO
            recipients.append(SMTP_CC_TO)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText(body, "plain"))
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP failure-email send failed: %s", e)


def _send_failure_email(to_email: str, job_id: str, job_title: str | None) -> None:
    """Notify that the job failed after max attempts (no third try)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return
    subject = f"HQIV prediction failed (job {job_id})"
    body = f"Job {job_id} (title={job_title or 'n/a'}) did not complete after {MAX_ATTEMPTS} attempts. No further retries will be made."
    msg = MIMEMultipart()
    msg["Subject"] = subject
    from_addr = _smtp_from_address()
    msg["From"] = formataddr(("HQIV CASP Server", from_addr))
    # Final failure notifications: send to CC/monitoring address when available.
    primary = SMTP_CC_TO or to_email
    msg["To"] = primary
    recipients = [primary]
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="casp.disregardfiat.tech")
    msg.attach(MIMEText(body, "plain"))
    try:
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.sendmail(from_addr, recipients, msg.as_string())
    except Exception as e:
        app.logger.warning("SMTP failure-email send failed: %s", e)


def _read_pending_attempts(txt_path: str) -> int:
    """Parse attempts=N from first lines of pending .txt; default 1."""
    try:
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("attempts="):
                    return max(1, int(line.split("=", 1)[1].strip() or "1"))
    except Exception:
        pass
    return 1


def _update_pending_attempts(txt_path: str, attempts: int) -> None:
    """Set attempts=N in the .txt file (rewrite first line)."""
    with open(txt_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("attempts="):
            lines[i] = f"attempts={attempts}\n"
            break
    else:
        lines.insert(0, f"attempts={attempts}\n")
    with open(txt_path, "w") as f:
        f.writelines(lines)


def _tail_text_file(path: str, max_chars: int = 2000) -> str:
    """Read the tail of a text file for compact subprocess error reporting."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
    except Exception:
        return ""
    data = data.strip()
    if len(data) > max_chars:
        return "...\n" + data[-max_chars:]
    return data


def _subprocess_env() -> dict[str, str]:
    """Disable startup queue scanning inside worker helper subprocesses."""
    env = os.environ.copy()
    env["CASP_DISABLE_PENDING_STARTUP"] = "1"
    return env


def _write_models_record(
    base: str,
    job_id: str,
    job_title: str | None,
    to_email: str | None,
    models: list[dict],
    *,
    timed_out: bool = False,
) -> None:
    """Write outputs/{base}.models.json describing the delivered models."""
    try:
        record = {
            "job_id": job_id,
            "title": job_title,
            "email": to_email,
            "models": models,
        }
        if timed_out:
            record["timed_out"] = True
        models_path = os.path.join(OUTPUTS_DIR, f"{base}.models.json")
        with open(models_path, "w") as f:
            json.dump(record, f)
    except Exception as e:
        app.logger.warning("Failed to write models.json for job %s: %s", job_id, e)


def _recover_completed_result(
    base: str,
    job_id: str,
    job_title: str | None,
    to_email: str | None,
    seqs: list[str],
    *,
    timed_out: bool = False,
    body_extra: str | None = None,
) -> bool:
    """
    Recover a completed PDB left in pending/ or already written to outputs/.
    This is used when a worker is interrupted after writing the main result but
    before it can finish moving files and emailing the requester.
    """
    out_pdb_path = os.path.join(OUTPUTS_DIR, f"{base}.pdb")
    pending_pdb_path = os.path.join(PENDING_DIR, f"{base}.pdb")
    main_pdb_path = out_pdb_path if os.path.isfile(out_pdb_path) else pending_pdb_path
    if not os.path.isfile(main_pdb_path):
        return False

    try:
        with open(main_pdb_path) as f:
            main_pdb = _prepend_pdb_model_line(f.read())
    except Exception as e:
        app.logger.warning("Could not read recovered PDB for job %s: %s", job_id, e)
        return False

    ok, reason = _pdb_sanity_check(main_pdb)
    if not ok:
        app.logger.warning("Recovered PDB for job %s failed sanity check: %s", job_id, reason)
        return False

    if main_pdb_path == pending_pdb_path:
        _move_to_outputs(base, main_pdb)

    fast_pdb = None
    out_fast_path = os.path.join(OUTPUTS_DIR, f"{base}.fast.pdb")
    pending_fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
    fast_pdb_path = out_fast_path if os.path.isfile(out_fast_path) else pending_fast_path
    if os.path.isfile(fast_pdb_path):
        try:
            with open(fast_pdb_path) as f:
                fast_pdb = _prepend_pdb_model_line(f.read())
            if fast_pdb_path == pending_fast_path:
                shutil.move(pending_fast_path, out_fast_path)
        except Exception as e:
            app.logger.warning("Could not recover fast-pass PDB for job %s: %s", job_id, e)
            fast_pdb = None

    ref_src = _experimental_ref_path_for_job(seqs, base)
    if ref_src and os.path.isfile(ref_src):
        ref_dst = os.path.join(OUTPUTS_DIR, f"{base}.experimental_ref.pdb")
        if not os.path.isfile(ref_dst):
            try:
                shutil.copy2(ref_src, ref_dst)
            except Exception as e:
                app.logger.warning("Could not copy experimental ref during recovery: %s", e)

    if fast_pdb:
        _write_models_record(
            base,
            job_id,
            job_title,
            to_email,
            models=[
                {"name": "Prediction1.pdb", "stage": "hke", "type": "hke", "file": f"{base}.pdb"},
                {"name": "Prediction2.pdb", "stage": "fast", "type": "fast", "file": f"{base}.fast.pdb"},
            ],
            timed_out=timed_out,
        )
        if to_email:
            _send_pdb_email_multi(
                to_email,
                models=[
                    ("Prediction1.pdb", main_pdb),
                    ("Prediction2.pdb", fast_pdb),
                ],
                title=job_title,
                body_extra=body_extra,
            )
    else:
        _write_models_record(
            base,
            job_id,
            job_title,
            to_email,
            models=[
                {"name": "Prediction1.pdb", "stage": "hke", "type": "hke", "file": f"{base}.pdb"},
            ],
            timed_out=timed_out,
        )
        if to_email:
            _send_pdb_email(to_email, main_pdb, job_title)
    return True


def _run_fast_pass(
    sequences: list[str],
    seqs: list[str],
    parsed_ligands: list | None = None,
) -> str:
    """
    Fast-pass prediction: quick Lean tunnel fold (default) or legacy hierarchical HKE.
    Multichain + ligand: each chain folded with Lean, merged, ligands refined vs full complex.
    Returns a PDB string (single or multi-chain).
    """
    if _use_lean_casp_pipeline():
        if len(seqs) > 1:
            return _predict_lean_assembly(
                sequences, quick=True, parsed_ligands=parsed_ligands or None
            )
        use_lig = bool(parsed_ligands)
        lig_chain = (os.environ.get("CASP_LIGAND_CHAIN_ID") or "L").strip() or None
        out = fold_lean_ribosome_tunnel(
            seqs[0],
            quick=True,
            include_ligands=use_lig,
            ligands=parsed_ligands if use_lig else None,
            ligand_chain_id=lig_chain if use_lig else None,
            **_lean_fold_env_kwargs(quick=True),
        )
        _backbone_sanity_check(out.raw_result.get("backbone_atoms") or [], job_context="(lean fast-pass)")
        return out.pdb
    if len(seqs) > 1:
        # Fold each chain quickly and merge without docking
        chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        results = []
        for i, seq in enumerate(seqs):
            pos, z_list = minimize_full_chain_hierarchical(
                seq,
                include_sidechains=False,
                funnel_radius=FUNNEL_RADIUS,
                funnel_stiffness=1.0,
                funnel_radius_exit=FUNNEL_RADIUS_EXIT,
                max_iter_stage1=max(5, HKE_MAX_ITER[0] // 2),
                max_iter_stage2=max(10, HKE_MAX_ITER[1] // 2),
                max_iter_stage3=max(10, HKE_MAX_ITER[2] // 2),
            )
            result = hierarchical_result_for_pdb(pos, z_list, seq, include_sidechains=False)
            _backbone_sanity_check(result.get("backbone_atoms") or [], job_context="(HKE fast-pass)")
            cid = chain_ids[i] if i < len(chain_ids) else "A"
            results.append((result, cid))
        return _merge_pdb_chains(
            results,
            ligands=parsed_ligands or None,
            ligand_refine_quick=True,
        )
    # Single-chain quick hierarchical
    pos, z_list = minimize_full_chain_hierarchical(
        seqs[0],
        include_sidechains=False,
        funnel_radius=FUNNEL_RADIUS,
        funnel_stiffness=1.0,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        max_iter_stage1=max(5, HKE_MAX_ITER[0] // 2),
        max_iter_stage2=max(10, HKE_MAX_ITER[1] // 2),
        max_iter_stage3=max(10, HKE_MAX_ITER[2] // 2),
    )
    result = hierarchical_result_for_pdb(pos, z_list, seqs[0], include_sidechains=False)
    _backbone_sanity_check(result.get("backbone_atoms") or [], job_context="(HKE fast-pass)")
    if parsed_ligands:
        return _merge_pdb_chains(
            [(result, "A")],
            ligands=parsed_ligands,
            ligand_refine_quick=True,
        )
    return full_chain_to_pdb(result, chain_id="A")


def _run_fast_pass_only(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    parsed_ligands: list | None = None,
) -> None:
    """
    Run fast-pass only: write .fast.pdb, send first email. Does NOT increment attempts.
    Used by process_pending_jobs Phase 1 to give everyone a quick result. base = email__job_id or job_id for paths.
    """
    fast_pdb = _run_fast_pass(sequences, seqs, parsed_ligands=parsed_ligands)
    fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
    with open(fast_path, "w") as f:
        f.write(fast_pdb)
    if to_email:
        _send_pdb_email_custom(to_email, fast_pdb, job_title, filename="Prediction1.pdb")


def _run_hke_only(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    parsed_ligands: list,
) -> None:
    """
    Run HKE pipeline only. Assumes .fast.pdb exists in pending.
    On success: move to outputs, send second email (HKE + fast).
    On timeout (1h): write current PDB, send email with time-limit note, move to outputs, pick up next job.
    On HKE failure: fall back to the already-generated fast-pass result so the requester still gets a model.
    """
    fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
    with open(fast_path) as f:
        fast_pdb = f.read()

    # If USE_FAST_PREDICT, treat fast-pass as final (no HKE)
    if USE_FAST_PREDICT and hqiv_predict_structure is not None:
        fast_pdb = _prepend_pdb_model_line(fast_pdb)
        ok, reason = _pdb_sanity_check(fast_pdb)
        if not ok:
            raise ValueError(f"PDB sanity check failed (fast-pass): {reason}")
        _move_to_outputs(base, fast_pdb)
        try:
            shutil.move(fast_path, os.path.join(OUTPUTS_DIR, f"{base}.fast.pdb"))
        except Exception:
            pass
        _write_models_record(
            base,
            job_id,
            job_title,
            to_email,
            models=[
                {"name": "Prediction1.pdb", "stage": "fast", "type": "fast", "file": f"{base}.pdb"},
                {"name": "Prediction2.pdb", "stage": "fast", "type": "fast", "file": f"{base}.fast.pdb"},
            ],
            timed_out=False,
        )
        if to_email:
            _send_pdb_email_multi(
                to_email,
                models=[
                    ("Prediction1.pdb", fast_pdb),
                    ("Prediction2.pdb", fast_pdb),
                ],
                title=job_title,
            )
        return

    deadline = time.time() + PREDICTION_TIMEOUT_SEC
    timed_out = False
    hke_pdb = None
    used_fast_fallback = False
    fallback_reason = ""

    # Full prediction pipeline: Lean tunnel (default) or legacy HKE / assembly
    main_stage = "lean" if _use_lean_casp_pipeline() else "hke"
    try:
        if _use_lean_casp_pipeline():
            if len(seqs) == 2 and os.environ.get("CASP_LEGACY_ASSEMBLY_DOCK", "").strip().lower() in ("1", "true", "yes"):
                assembly = None
                try:
                    assembly = _predict_hke_assembly_with_complex(sequences)
                except Exception as e:
                    app.logger.warning("Legacy 2-chain assembly failed, falling back to Cartesian: %s", e)
                    assembly = _predict_cartesian_assembly_with_complex(sequences)
                if assembly is not None:
                    _pdb_a, _pdb_b, pdb_complex = assembly
                    hke_pdb = pdb_complex
                else:
                    hke_pdb = _predict_lean_assembly(
                        sequences, quick=False, parsed_ligands=parsed_ligands or None
                    )
            elif len(seqs) >= 2:
                hke_pdb = _predict_lean_assembly(
                    sequences, quick=False, parsed_ligands=parsed_ligands or None
                )
            else:
                hke_pdb, timed_out = _predict_lean_single_chain(
                    seqs[0],
                    parsed_ligands,
                    deadline,
                )
                if timed_out and hke_pdb is None:
                    hke_pdb = fast_pdb
                    app.logger.info(
                        "Job %s hit %ds timeout before Lean fold; sending fast-pass as result.",
                        job_id,
                        PREDICTION_TIMEOUT_SEC,
                    )
        elif len(seqs) == 2:
            assembly = None
            try:
                assembly = _predict_hke_assembly_with_complex(sequences)
            except Exception as e:
                app.logger.warning("HKE 2-chain failed, falling back to Cartesian: %s", e)
                assembly = _predict_cartesian_assembly_with_complex(sequences)
            if assembly is not None:
                _pdb_a, _pdb_b, pdb_complex = assembly
                hke_pdb = pdb_complex
            else:
                hke_pdb = _predict_hke_assembly(seqs, parsed_ligands=parsed_ligands or None)
        elif len(seqs) >= 3:
            hke_pdb = _predict_hke_assembly_multichain(sequences, parsed_ligands=parsed_ligands or None)
        else:
            hke_pdb, timed_out = _predict_hke_single(
                seqs[0],
                ligands=parsed_ligands if parsed_ligands else None,
                deadline_sec=deadline,
            )
            if timed_out and hke_pdb is None:
                hke_pdb = fast_pdb
                app.logger.info("Job %s hit %ds timeout before HKE; sending fast-pass as result.", job_id, PREDICTION_TIMEOUT_SEC)
    except Exception as e:
        used_fast_fallback = True
        fallback_reason = str(e).strip()
        hke_pdb = fast_pdb
        app.logger.warning(
            "Job %s full prediction pipeline failed; delivering fast-pass result instead: %s",
            job_id,
            fallback_reason or e,
        )

    # Prepend submission model line and sanity-check before moving or sending
    hke_pdb = _prepend_pdb_model_line(hke_pdb)
    fast_pdb = _prepend_pdb_model_line(fast_pdb)
    ok, reason = _pdb_sanity_check(hke_pdb)
    if not ok:
        raise ValueError(f"PDB sanity check failed: {reason}")

    # Move main HKE PDB + bookkeeping via existing helper
    _move_to_outputs(base, hke_pdb)

    # After move_to_outputs, pending/{base}.pdb is now in outputs/; move fast-pass too
    out_fast = os.path.join(OUTPUTS_DIR, f"{base}.fast.pdb")
    try:
        if os.path.isfile(fast_path):
            shutil.move(fast_path, out_fast)
    except Exception:
        pass

    # If request matches a known CASP target with experimental structure, copy ref to outputs (equal footing)
    ref_src = _experimental_ref_path_for_job(seqs, base)
    if ref_src and os.path.isfile(ref_src):
        ref_dst = os.path.join(OUTPUTS_DIR, f"{base}.experimental_ref.pdb")
        try:
            shutil.copy2(ref_src, ref_dst)
            app.logger.info("Copied experimental ref to %s", ref_dst)
        except Exception as e:
            app.logger.warning("Could not copy experimental ref: %s", e)

    _write_models_record(
        base,
        job_id,
        job_title,
        to_email,
        models=(
            [
                {
                    "name": "Prediction1.pdb",
                    "stage": "fast",
                    "type": "fast",
                    "file": f"{base}.pdb",
                },
                {
                    "name": "Prediction2.pdb",
                    "stage": "fast",
                    "type": "fast",
                    "file": f"{base}.fast.pdb",
                },
            ]
            if used_fast_fallback
            else [
                {
                    "name": "Prediction1.pdb",
                    "stage": main_stage,
                    "type": main_stage,
                    "file": f"{base}.pdb",
                },
                {
                    "name": "Prediction2.pdb",
                    "stage": "fast",
                    "type": "fast",
                    "file": f"{base}.fast.pdb",
                },
            ]
        ),
        timed_out=timed_out,
    )

    # 3) Second email: HKE as Prediction1.pdb, fast-pass as Prediction2.pdb (to requester and CC)
    if to_email:
        body_extra = None
        if timed_out:
            body_extra = (
                f"This prediction was stopped after the {PREDICTION_TIMEOUT_SEC // 3600}-hour time limit. "
                "The attached model is the best structure obtained up to that point."
            )
        elif used_fast_fallback:
            body_extra = (
                "The full main prediction (Lean tunnel or legacy HKE) did not complete successfully, "
                "so the attached model is the fast-pass result."
            )
            if fallback_reason:
                body_extra += f"\n\nRefinement error: {fallback_reason}"
        _send_pdb_email_multi(
            to_email,
            models=[
                ("Prediction1.pdb", hke_pdb),
                ("Prediction2.pdb", fast_pdb),
            ],
            title=job_title,
            body_extra=body_extra,
        )


def _run_hke_from_temp_file(temp_path: str) -> None:
    """Entry point for subprocess: read job payload from temp_path and run _run_hke_only. Used for wall-clock timeout."""
    with open(temp_path) as f:
        payload = json.load(f)
    ligand_str = (payload.get("ligand_str") or "").strip()
    parsed_ligands = parse_ligands(ligand_str) if ligand_str else []
    _run_hke_only(
        payload["job_id"],
        payload["base"],
        payload["sequences"],
        payload["seqs"],
        payload.get("to_email"),
        payload.get("job_title"),
        parsed_ligands,
    )


def _run_one_job_fast_then_hke(temp_path: str) -> None:
    """
    Entry point for subprocess: run fast-pass (if .fast.pdb missing) then HKE for one job.
    Entire job is in one process so a 1h wall-clock timeout covers both; on timeout parent sends fast-pass.
    """
    with open(temp_path) as f:
        payload = json.load(f)
    job_id = payload["job_id"]
    base = payload["base"]
    sequences = payload["sequences"]
    seqs = payload["seqs"]
    to_email = payload.get("to_email")
    job_title = payload.get("job_title")
    ligand_str = (payload.get("ligand_str") or "").strip()
    parsed_ligands = parse_ligands(ligand_str) if ligand_str else []
    fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
    if not os.path.isfile(fast_path):
        _run_fast_pass_only(
            job_id, base, sequences, seqs, to_email, job_title, parsed_ligands=parsed_ligands
        )
    _run_hke_only(job_id, base, sequences, seqs, to_email, job_title, parsed_ligands)


def _run_one_job_with_timeout(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    ligand_str: str,
    txt_path: str,
    attempts: int,
) -> None:
    """
    Run one job (fast-pass if needed + HKE) in a subprocess with PREDICTION_TIMEOUT_SEC wall clock.
    After 1h we kill the process and send fast-pass so the user always gets a result.
    On failure: increment attempts, send failure email; if attempts >= MAX_ATTEMPTS move to .failed.txt.
    Uses _job_concurrency_semaphore so at most MAX_CONCURRENT_JOBS run at once (startup + POST).
    """
    with _job_concurrency_semaphore:
        _run_one_job_with_timeout_impl(
            job_id, base, sequences, seqs, to_email, job_title, ligand_str, txt_path, attempts
        )


def _run_one_job_with_timeout_impl(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    ligand_str: str,
    txt_path: str,
    attempts: int,
) -> None:
    """Actual work for one job (called while holding _job_concurrency_semaphore)."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_src = os.path.join(repo_root, "src")
    payload = {
        "job_id": job_id,
        "base": base,
        "sequences": sequences,
        "seqs": seqs,
        "to_email": to_email,
        "job_title": job_title,
        "ligand_str": ligand_str or "",
    }
    _run_one_job_subprocess_error = ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        temp_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name
    run_cmd = [
        sys.executable,
        "-c",
        "import sys, os, json\n"
        "sys.path.insert(0, %r)\n"
        "sys.path.insert(0, %r)\n"
        "with open(sys.argv[1]) as f: payload = json.load(f)\n"
        "from casp_server import _run_one_job_fast_then_hke\n"
        "_run_one_job_fast_then_hke(sys.argv[1])\n" % (repo_src, repo_root),
        temp_path,
    ]
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.Popen(
                run_cmd,
                cwd=repo_root,
                env=_subprocess_env(),
                start_new_session=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        try:
            proc.wait(timeout=PREDICTION_TIMEOUT_SEC)
            if proc.returncode != 0:
                _run_one_job_subprocess_error = _tail_text_file(log_path) or "exit code %s" % proc.returncode
                raise subprocess.CalledProcessError(proc.returncode, run_cmd)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            else:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            app.logger.warning(
                "Job %s hit wall-clock timeout (%ds); sending fast-pass result.",
                job_id,
                PREDICTION_TIMEOUT_SEC,
            )
            timeout_note = (
                f"This prediction was stopped after the {PREDICTION_TIMEOUT_SEC // 3600}-hour time limit. "
                "A completed model was recovered from the worker state and attached."
            )
            if _recover_completed_result(
                base,
                job_id,
                job_title,
                to_email,
                seqs,
                timed_out=True,
                body_extra=timeout_note,
            ):
                app.logger.info("Recovered completed PDB for timed-out job %s.", job_id)
                return
            fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
            if not os.path.isfile(fast_path):
                _update_pending_attempts(txt_path, attempts + 1)
                if to_email:
                    _send_job_failure_email(
                        to_email,
                        job_id,
                        job_title,
                        f"Prediction timed out after {PREDICTION_TIMEOUT_SEC // 3600}h before initial fold completed.",
                    )
                if attempts + 1 >= MAX_ATTEMPTS:
                    try:
                        shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{base}.failed.txt"))
                    except Exception:
                        pass
                return
            with open(fast_path) as fp:
                fast_pdb = fp.read()
            fast_pdb = _prepend_pdb_model_line(fast_pdb)
            ok, reason = _pdb_sanity_check(fast_pdb)
            if not ok:
                _update_pending_attempts(txt_path, attempts + 1)
                if to_email:
                    _send_job_failure_email(to_email, job_id, job_title, f"Fast-pass sanity check failed after timeout: {reason}")
                if attempts + 1 >= MAX_ATTEMPTS:
                    try:
                        shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{base}.failed.txt"))
                    except Exception:
                        pass
                return
            _move_to_outputs(base, fast_pdb)
            try:
                if os.path.isfile(fast_path):
                    shutil.move(fast_path, os.path.join(OUTPUTS_DIR, f"{base}.fast.pdb"))
            except Exception:
                pass
            try:
                _write_models_record(
                    base,
                    job_id,
                    job_title,
                    to_email,
                    models=[
                        {"name": "Prediction1.pdb", "stage": "fast", "type": "fast", "file": f"{base}.pdb"},
                        {"name": "Prediction2.pdb", "stage": "fast", "type": "fast", "file": f"{base}.fast.pdb"},
                    ],
                    timed_out=True,
                )
            except Exception as e:
                app.logger.warning("Failed to write models.json after timeout: %s", e)
            body_extra = (
                f"This prediction was stopped after the {PREDICTION_TIMEOUT_SEC // 3600}-hour time limit. "
                "The attached model is the fast-pass (quick fold) result."
            )
            if to_email:
                _send_pdb_email_multi(
                    to_email,
                    models=[
                        ("Prediction1.pdb", fast_pdb),
                        ("Prediction2.pdb", fast_pdb),
                    ],
                    title=job_title,
                    body_extra=body_extra,
                )
    except subprocess.CalledProcessError as e:
        if _recover_completed_result(base, job_id, job_title, to_email, seqs):
            app.logger.info("Recovered completed PDB for failed job %s after subprocess exit.", job_id)
            return
        new_attempts = attempts + 1
        _update_pending_attempts(txt_path, new_attempts)
        err_msg = _run_one_job_subprocess_error.strip() if _run_one_job_subprocess_error else "HKE subprocess exited with error (code %s)." % getattr(e, "returncode", "?")
        if to_email:
            _send_job_failure_email(to_email, job_id, job_title, err_msg)
        if new_attempts >= MAX_ATTEMPTS:
            try:
                shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{base}.failed.txt"))
            except Exception:
                pass
    except Exception as e:
        if _recover_completed_result(base, job_id, job_title, to_email, seqs):
            app.logger.info("Recovered completed PDB for job %s after exception.", job_id)
            return
        app.logger.warning("Pending job %s failed: %s", job_id, e)
        new_attempts = attempts + 1
        _update_pending_attempts(txt_path, new_attempts)
        if to_email:
            _send_job_failure_email(to_email, job_id, job_title, str(e))
        if new_attempts >= MAX_ATTEMPTS:
            try:
                shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{base}.failed.txt"))
            except Exception:
                pass
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        try:
            os.unlink(log_path)
        except Exception:
            pass


def _run_hke_only_with_timeout(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    ligand_str: str,
) -> None:
    """
    Run _run_hke_only in a subprocess with wall-clock timeout. If the subprocess exceeds
    PREDICTION_TIMEOUT_SEC, we kill it and send the fast-pass result so the user gets something.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    repo_src = os.path.join(repo_root, "src")
    payload = {
        "job_id": job_id,
        "base": base,
        "sequences": sequences,
        "seqs": seqs,
        "to_email": to_email,
        "job_title": job_title,
        "ligand_str": ligand_str or "",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        temp_path = f.name
    try:
        cmd = [
            sys.executable,
            "-c",
            "import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); "
            "import json; exec(open(sys.argv[1]).read()); __run(__import__('casp_server').__dict__)",
        ]
        # Simpler: run a script that imports and calls
        run_cmd = [
            sys.executable,
            "-c",
            "import sys, os, json\n"
            "sys.path.insert(0, %r)\n"
            "sys.path.insert(0, %r)\n"
            "with open(sys.argv[1]) as f: payload = json.load(f)\n"
            "from casp_server import parse_ligands, _run_hke_only\n"
            "pl = parse_ligands(payload.get('ligand_str') or '') if payload.get('ligand_str') else []\n"
            "_run_hke_only(payload['job_id'], payload['base'], payload['sequences'], payload['seqs'], "
            "payload.get('to_email'), payload.get('job_title'), pl)\n" % (repo_src, repo_root),
            temp_path,
        ]
        proc = subprocess.Popen(
            run_cmd,
            cwd=repo_root,
            env=_subprocess_env(),
            start_new_session=True,
        )
        try:
            proc.wait(timeout=PREDICTION_TIMEOUT_SEC)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, run_cmd)
        except subprocess.TimeoutExpired:
            # Kill entire process group so any child processes (e.g. from JAX/threading) also stop.
            if hasattr(os, "killpg"):
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
            else:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            app.logger.warning(
                "Job %s hit wall-clock timeout (%ds); sending fast-pass result.",
                job_id,
                PREDICTION_TIMEOUT_SEC,
            )
            fast_path = os.path.join(PENDING_DIR, f"{base}.fast.pdb")
            if not os.path.isfile(fast_path):
                raise FileNotFoundError(f"No fast.pdb for timed-out job {job_id}")
            with open(fast_path) as fp:
                fast_pdb = fp.read()
            fast_pdb = _prepend_pdb_model_line(fast_pdb)
            ok, reason = _pdb_sanity_check(fast_pdb)
            if not ok:
                raise ValueError(f"Fast-pass PDB sanity check failed after timeout: {reason}")
            _move_to_outputs(base, fast_pdb)
            try:
                if os.path.isfile(fast_path):
                    shutil.move(fast_path, os.path.join(OUTPUTS_DIR, f"{base}.fast.pdb"))
            except Exception:
                pass
            try:
                models_path = os.path.join(OUTPUTS_DIR, f"{base}.models.json")
                record = {
                    "job_id": job_id,
                    "title": job_title,
                    "email": to_email,
                    "timed_out": True,
                    "models": [
                        {"name": "Prediction1.pdb", "stage": "fast", "type": "fast", "file": f"{base}.pdb"},
                        {"name": "Prediction2.pdb", "stage": "fast", "type": "fast", "file": f"{base}.fast.pdb"},
                    ],
                }
                with open(models_path, "w") as f:
                    json.dump(record, f)
            except Exception as e:
                app.logger.warning("Failed to write models.json after timeout: %s", e)
            body_extra = (
                f"This prediction was stopped after the {PREDICTION_TIMEOUT_SEC // 3600}-hour time limit. "
                "The attached model is the fast-pass (quick fold) result."
            )
            if to_email:
                _send_pdb_email_multi(
                    to_email,
                    models=[
                        ("Prediction1.pdb", fast_pdb),
                        ("Prediction2.pdb", fast_pdb),
                    ],
                    title=job_title,
                    body_extra=body_extra,
                )
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _run_full_job_pipelines(
    job_id: str,
    base: str,
    sequences: list[str],
    seqs: list[str],
    to_email: str | None,
    job_title: str | None,
    parsed_ligands: list,
) -> None:
    """
    For a given job: run fast-pass first (send fast email), then full HKE pipeline.
    Used by _run_job_in_background for new POST jobs. base = email__job_id or job_id for paths.
    """
    _run_fast_pass_only(
        job_id, base, sequences, seqs, to_email, job_title, parsed_ligands=parsed_ligands
    )
    _run_hke_only(job_id, base, sequences, seqs, to_email, job_title, parsed_ligands)


def process_pending_jobs() -> None:
    """
    On startup: one thread per pending job. At most MAX_CONCURRENT_JOBS run at once (semaphore).
    Each job runs in a subprocess with 1h wall-clock timeout (fast-pass if needed, then HKE).
    After 1h we kill the process and send fast-pass. No single job blocks the rest.
    """
    _ensure_output_dirs()
    if not os.path.isdir(PENDING_DIR):
        return

    # Collect valid jobs: have .txt and .request.json, no outputs .pdb (filename base = email__job_id or job_id)
    jobs: list[dict] = []
    for name in os.listdir(PENDING_DIR):
        if not name.endswith(".txt") or name.endswith(".failed.txt"):
            continue
        base = name[:-4]
        job_id = _base_to_job_id(base)
        txt_path = os.path.join(PENDING_DIR, name)
        req_path = os.path.join(PENDING_DIR, f"{base}.request.json")
        out_pdb = os.path.join(OUTPUTS_DIR, f"{base}.pdb")
        if os.path.isfile(out_pdb):
            continue  # Already done
        if not os.path.isfile(req_path):
            # Legacy: .txt only — can't retry
            try:
                with open(txt_path) as f:
                    for line in f:
                        if line.strip().startswith("email="):
                            to_email = line.split("=", 1)[1].strip()
                            if to_email and re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
                                _send_failure_email(to_email, job_id, None)
                            break
            except Exception:
                pass
            try:
                shutil.move(txt_path, os.path.join(OUTPUTS_DIR, f"{base}.failed.txt"))
            except Exception:
                pass
            continue
        try:
            with open(req_path) as f:
                req = json.load(f)
        except Exception:
            continue
        sequences = req.get("sequences") or []
        seqs = [_sequence_from_input(s) for s in sequences if s]
        if not seqs:
            continue
        attempts = _read_pending_attempts(txt_path)
        to_email = (req.get("email") or "").strip()
        job_title = req.get("title")
        ligand_str = (req.get(LIGAND_KEY) or "").strip()
        parsed_ligands = parse_ligands(ligand_str) if ligand_str else []
        if _recover_completed_result(base, job_id, job_title, to_email, seqs):
            app.logger.info("Recovered completed pending job %s on startup.", job_id)
            continue
        jobs.append({
            "job_id": job_id,
            "base": base,
            "txt_path": txt_path,
            "req_path": req_path,
            "sequences": sequences,
            "seqs": seqs,
            "to_email": to_email,
            "job_title": job_title,
            "parsed_ligands": parsed_ligands,
            "ligand_str": ligand_str,
            "attempts": attempts,
        })

    # Handle max-attempts jobs: send failure to CC, move .txt to failed (fast-pass already sent in Phase 1).
    # This runs only at startup; the email is sent when we find a job in *pending* with attempts>=2
    # (e.g. job failed twice and was moved to .failed.txt, then someone reset by copying .failed.txt
    # back to pending but left attempts=2, then server restarted).
    for j in jobs:
        if j["attempts"] >= MAX_ATTEMPTS:
            _send_failure_email(j["to_email"], j["job_id"], j["job_title"])
            try:
                shutil.move(j["txt_path"], os.path.join(OUTPUTS_DIR, f"{j['base']}.failed.txt"))
            except Exception:
                # If move fails, rename in place so we don't re-send this email on every restart
                try:
                    os.rename(j["txt_path"], j["txt_path"] + ".failed")
                except Exception:
                    pass
    jobs = [j for j in jobs if j["attempts"] < MAX_ATTEMPTS]

    def _run_one_job(j: dict) -> None:
        if j["parsed_ligands"]:
            app.logger.info(
                "Ligand detected via key %s → %d ligands added as 6-DOF agents",
                LIGAND_KEY,
                len(j["parsed_ligands"]),
            )
        _run_one_job_with_timeout(
            j["job_id"],
            j["base"],
            j["sequences"],
            j["seqs"],
            j["to_email"],
            j["job_title"],
            j.get("ligand_str", "") or "",
            j["txt_path"],
            j["attempts"],
        )

    for j in jobs:
        threading.Thread(target=_run_one_job, args=(j,), daemon=True).start()


def _sequence_from_input(raw: str) -> str:
    """Extract one-letter sequence from FASTA or raw sequence."""
    s = (raw or "").strip()
    if not s:
        return ""
    return _parse_fasta(s) if ">" in s or "\n" in s else "".join(c for c in s.upper() if c.isalpha())


def _predict_lean_assembly(
    sequences: list[str],
    *,
    quick: bool,
    parsed_ligands: list | None = None,
) -> str:
    """
    Fold each chain with the Lean tunnel pipeline and merge (offset chains in +x).
    Optional ligands: 6-DOF refinement vs full merged backbone, then HETATM (same screening as Lean fold).
    """
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    results: list[tuple[dict, str]] = []
    kw = _lean_fold_env_kwargs(quick)
    for i, raw in enumerate(sequences):
        seq = _sequence_from_input(raw)
        if not seq:
            continue
        out = fold_lean_ribosome_tunnel(
            seq,
            quick=quick,
            include_ligands=False,
            **kw,
        )
        _backbone_sanity_check(out.raw_result.get("backbone_atoms") or [], job_context="(lean assembly)")
        cid = chain_ids[i] if i < len(chain_ids) else "A"
        results.append((out.raw_result, cid))
    if not results:
        return hqiv_qconsi_empty_pdb_block()
    return _merge_pdb_chains(
        results,
        ligands=parsed_ligands or None,
        ligand_refine_quick=quick,
    )


def _predict_lean_single_chain(
    sequence: str,
    parsed_ligands: list,
    deadline_sec: float | None,
) -> tuple[str | None, bool]:
    """
    Default single-chain CASP prediction: Lean ribosome tunnel + solvent/dihedral + optional ligands.
    Returns (pdb_str, timed_out).
    """
    if deadline_sec is not None and time.time() >= deadline_sec:
        return (None, True)
    use_lig = bool(parsed_ligands)
    lig_chain = (os.environ.get("CASP_LIGAND_CHAIN_ID") or "L").strip() or None
    kw = _lean_fold_env_kwargs(quick=False)
    out = fold_lean_ribosome_tunnel(
        sequence,
        quick=False,
        include_ligands=use_lig,
        ligands=parsed_ligands if use_lig else None,
        ligand_chain_id=lig_chain if use_lig else None,
        **kw,
    )
    _backbone_sanity_check(out.raw_result.get("backbone_atoms") or [], job_context="(lean tunnel)")
    result = dict(out.raw_result)
    seq = str(result.get("sequence") or sequence)
    backbone = result.get("backbone_atoms")
    post_mode = str(kw.get("post_extrusion_refine_mode") or "em_treetorque").strip().lower()
    if (
        run_discrete_refinement
        and post_mode != "em_treetorque"
        and os.environ.get("CASP_LEAN_TREE_TORQUE_AFTER", "").strip().lower() in ("1", "true", "yes")
        and backbone
        and len(backbone) == 4 * len(seq)
    ):
        try:
            ref = run_discrete_refinement(
                seq,
                initial_backbone_atoms=list(backbone),
                run_until_converged=True,
                max_phases_cap=100000,
            )
            result = {
                **result,
                "backbone_atoms": ref.backbone_atoms,
                "sequence": ref.sequence,
                "n_res": ref.n_res,
                "include_sidechains": False,
            }
        except Exception:
            pass
    pdb_str = full_chain_to_pdb(
        result,
        chain_id="A",
        ligands=result.get("ligands"),
        ligand_chain_id=lig_chain if use_lig else None,
    )
    return (pdb_str, False)


def _predict_hke_single(
    sequence: str,
    ligands: list | None = None,
    deadline_sec: float | None = None,
) -> tuple[str | None, bool]:
    """
    Run single-chain prediction. Returns (pdb_str, timed_out).
    If timed_out is True, pdb_str may still be the best-so-far model (or None if no step completed).
    """
    def _over_deadline() -> bool:
        return deadline_sec is not None and time.time() >= deadline_sec

    if ligands:
        result = minimize_full_chain(
            sequence,
            include_sidechains=False,
            include_ligands=True,
            ligands=ligands,
        )
        _backbone_sanity_check(result.get("backbone_atoms") or [], job_context="(minimize_full_chain)")
        backbone = result.get("backbone_atoms")
        seq = result.get("sequence", sequence)
        if run_discrete_refinement and backbone and len(backbone) == 4 * len(seq):
            try:
                ref = run_discrete_refinement(
                    seq,
                    initial_backbone_atoms=backbone,
                    run_until_converged=True,
                    max_phases_cap=100000,
                )
                result = {"backbone_atoms": ref.backbone_atoms, "sequence": ref.sequence, "n_res": ref.n_res, "include_sidechains": False, "ligands": result.get("ligands")}
            except Exception:
                pass
        return (full_chain_to_pdb(result, chain_id="A", ligands=result.get("ligands")), False)

    # Prefer full extrusion + HKE + tree-torque cycle when helper and pyhqiv are available.
    if extrude_hke_treetorque_cycle is not None and run_discrete_refinement is not None:
        try:
            cyc = extrude_hke_treetorque_cycle(
                sequence,
                temperature=310.0,
                max_phases_cap=100000,
                hke_max_iter_stages=HKE_ONE_PASS_ITER,
                rmsd_threshold=1.0,
                deadline_sec=deadline_sec,
            )
            _backbone_sanity_check(cyc.backbone_atoms or [], job_context="(extrude+HKE+tree-torque)")
            pdb_str = full_chain_to_pdb(
                {
                    "backbone_atoms": cyc.backbone_atoms,
                    "sequence": cyc.sequence,
                    "n_res": cyc.n_res,
                    "include_sidechains": False,
                },
                chain_id="A",
            )
            return (pdb_str, bool(cyc.meta.get("timed_out")))
        except Exception as e:
            app.logger.warning("Extrude+HKE+tree-torque cycle failed for single-chain: %s", e)

    # Fallback: HKE + tree-torque. Check deadline before starting and before tree-torque.
    if _over_deadline():
        return (None, True)
    pos, z_list = minimize_full_chain_hierarchical(
        sequence,
        include_sidechains=False,
        funnel_radius=FUNNEL_RADIUS,
        funnel_stiffness=1.0,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        max_iter_stage1=HKE_ONE_PASS_ITER[0],
        max_iter_stage2=HKE_ONE_PASS_ITER[1],
        max_iter_stage3=HKE_ONE_PASS_ITER[2],
    )
    result = hierarchical_result_for_pdb(pos, z_list, sequence, include_sidechains=False)
    _backbone_sanity_check(result.get("backbone_atoms") or [], job_context="(HKE)")
    backbone = result.get("backbone_atoms")
    seq = result.get("sequence", sequence)
    if _over_deadline():
        return (full_chain_to_pdb(result, chain_id="A"), True)
    if run_discrete_refinement and backbone and len(backbone) == 4 * len(seq):
        try:
            ref = run_discrete_refinement(
                seq,
                initial_backbone_atoms=backbone,
                run_until_converged=True,
                max_phases_cap=100000,
            )
            result = {"backbone_atoms": ref.backbone_atoms, "sequence": ref.sequence, "n_res": ref.n_res, "include_sidechains": False}
        except Exception:
            pass
    return (full_chain_to_pdb(result, chain_id="A"), False)


def _merge_pdb_chains(
    result_and_chain_ids: list[tuple[dict, str]],
    *,
    ligands: list | None = None,
    ligand_chain_id: str | None = None,
    ligand_refine_quick: bool = False,
    ligand_refine_steps: int | None = None,
) -> str:
    """
    Merge multiple (result, chain_id) into one PDB (HQIV-QConSi REMARK + MODEL) with renumbered atom IDs.
    Offsets each chain along +x. Optional ligands: Lean-screened 6-DOF refinement vs merged backbone, then HETATM.
    """
    from horizon_physics.proteins.casp_submission import AA_1to3
    chain_gap = 50.0  # Å between chains
    # First pass: get min_x per chain for offset calculation
    chain_mins = []
    for result, _ in result_and_chain_ids:
        backbone_atoms = result["backbone_atoms"]
        if not backbone_atoms:
            chain_mins.append(0.0)
            continue
        min_x = min(float(xyz[0]) for _, xyz in backbone_atoms)
        chain_mins.append(min_x)
    lines = list(hqiv_qconsi_model_lines())
    atom_id = 1
    prev_max_x = float("-inf")
    for ci, (result, chain_id) in enumerate(result_and_chain_ids):
        backbone_atoms = result["backbone_atoms"]
        sequence = result["sequence"]
        include_sidechains = result.get("include_sidechains", False)
        if not backbone_atoms:
            continue
        # Offset so this chain's leftmost is to the right of previous chain's rightmost + gap
        offset_x = max(0.0, prev_max_x + chain_gap - chain_mins[ci])
        idx = 0
        chain_max_x = float("-inf")
        for res_id in range(1, result["n_res"] + 1):
            res_1 = sequence[res_id - 1]
            res_3 = AA_1to3.get(res_1, "UNK")
            n_atoms_this = (5 if res_1 != "G" else 4) if include_sidechains else 4
            for _ in range(n_atoms_this):
                name, xyz = backbone_atoms[idx]
                x = float(xyz[0]) + offset_x
                y = float(xyz[1])
                z = float(xyz[2])
                chain_max_x = max(chain_max_x, x)
                lines.append(
                    f"ATOM  {atom_id:5d}  {name:2s}  {res_3:3s} {chain_id}{res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                )
                atom_id += 1
                idx += 1
        prev_max_x = chain_max_x
    if ligands:
        lcopy = copy.deepcopy(ligands)
        if atom_id > 1:
            kw_lean = _lean_fold_env_kwargs(quick=ligand_refine_quick)
            steps = int(
                ligand_refine_steps if ligand_refine_steps is not None else kw_lean["ligand_refine_steps"]
            )
            refine_ligands_on_multichain_results(
                result_and_chain_ids,
                lcopy,
                grad_full_kwargs=_lean_grad_full_kwargs_for_ligand_refine(quick=ligand_refine_quick),
                ligand_refine_steps=steps,
                ligand_refinement_mode=str(kw_lean.get("ligand_refinement_mode", "lean_qc")),
                qc_soft_clash_sigma=float(kw_lean.get("qc_soft_clash_sigma", 3.0)),
                qc_clash_weight=float(kw_lean.get("qc_clash_weight", 1.0)),
            )
        het_c = (ligand_chain_id or os.environ.get("CASP_LIGAND_CHAIN_ID") or "L").strip() or "L"
        het_lines, atom_id = pdb_hetatm_lines_for_ligands(
            lcopy, start_atom_id=atom_id, ligand_chain_id=het_c
        )
        lines.extend(het_lines)
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


def _predict_hke_assembly(sequences: list[str], parsed_ligands: list | None = None) -> str:
    """Run HKE + funnel per chain; return one PDB with chain A, B, C, .... (no docking)."""
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    results = []
    for i, seq in enumerate(sequences):
        pos, z_list = minimize_full_chain_hierarchical(
            seq,
            include_sidechains=False,
            funnel_radius=FUNNEL_RADIUS,
            funnel_stiffness=1.0,
            funnel_radius_exit=FUNNEL_RADIUS_EXIT,
            max_iter_stage1=HKE_ONE_PASS_ITER[0],
            max_iter_stage2=HKE_ONE_PASS_ITER[1],
            max_iter_stage3=HKE_ONE_PASS_ITER[2],
        )
        result = hierarchical_result_for_pdb(pos, z_list, seq, include_sidechains=False)
        _backbone_sanity_check(result.get("backbone_atoms") or [], job_context="(HKE assembly)")
        cid = chain_ids[i] if i < len(chain_ids) else "A"
        results.append((result, cid))
    return _merge_pdb_chains(
        results,
        ligands=parsed_ligands or None,
        ligand_refine_quick=False,
    )


def _predict_hke_assembly_multichain(sequences: list[str], parsed_ligands: list | None = None) -> str:
    """
    Multi-chain with (A+B)+C docking: fold A, fold B, dock A+B; then fold C, dock (A+B)+C;
    repeat for further chains. Returns single PDB with chain A, B, C, ...
    """
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seqs = [_sequence_from_input(s) for s in sequences]
    seqs = [s for s in seqs if s]
    if not seqs:
        return hqiv_qconsi_empty_pdb_block()
    if len(seqs) == 1:
        pdb, _ = _predict_hke_single(
            seqs[0], ligands=parsed_ligands if parsed_ligands else None
        )
        return pdb or hqiv_qconsi_empty_pdb_block()
    if len(seqs) == 2:
        # Prefer full EM-field assembly cycle (HKE + assembly-mode tree-torque + HKE + tree-torque)
        if assembly_hke_treetorque_cycle_two_chains is not None and run_discrete_refinement is not None:
            try:
                cyc = assembly_hke_treetorque_cycle_two_chains(
                    seqs[0],
                    seqs[1],
                    hke_max_iter_stages=HKE_ONE_PASS_ITER,
                    rmsd_threshold=1.0,
                )
                return cyc.pdb_complex
            except Exception as e:
                app.logger.warning("Assembly HKE+tree-torque cycle failed for 2-chain: %s", e)
        assembly = _predict_hke_assembly_with_complex(sequences)
        if assembly is not None:
            return assembly[2]
        return _predict_hke_assembly(sequences, parsed_ligands=parsed_ligands)
    # (A+B)+C pipeline: dock A+B (EM field, connection points), then tree-torque (further from COM first)
    result_a, result_b, result_ab = run_two_chain_assembly_hke(
        seqs[0],
        seqs[1],
        funnel_radius=FUNNEL_RADIUS,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        funnel_stiffness=1.0,
        hke_max_iter_s1=HKE_ONE_PASS_ITER[0],
        hke_max_iter_s2=HKE_ONE_PASS_ITER[1],
        hke_max_iter_s3=HKE_ONE_PASS_ITER[2],
        converge_max_disp_per_100_res=1.0,
        max_dock_iter=600,
    )
    bb_a, bb_b = _refine_assembly_chains_with_tree_torque(result_ab, seqs[0], seqs[1])
    result_ab["backbone_chain_a"] = bb_a
    result_ab["backbone_chain_b"] = bb_b
    chain_lengths = [len(seqs[0]), len(seqs[1])]
    for i in range(2, len(seqs)):
        result_combined = complex_to_single_chain_result(result_ab)
        pos, z_list = minimize_full_chain_hierarchical(
            seqs[i],
            include_sidechains=False,
            funnel_radius=FUNNEL_RADIUS,
            funnel_stiffness=1.0,
            funnel_radius_exit=FUNNEL_RADIUS_EXIT,
            max_iter_stage1=HKE_ONE_PASS_ITER[0],
            max_iter_stage2=HKE_ONE_PASS_ITER[1],
            max_iter_stage3=HKE_ONE_PASS_ITER[2],
        )
        result_c = hierarchical_result_for_pdb(pos, z_list, seqs[i], include_sidechains=False)
        _backbone_sanity_check(result_c.get("backbone_atoms") or [], job_context="(HKE assembly C)")
        _, _, result_ab = run_two_chain_assembly(
            result_combined,
            result_c,
            max_dock_iter=600,
            converge_max_disp_per_100_res=1.0,
        )
        chain_lengths.append(len(seqs[i]))
    # Split result_ab: backbone_chain_a = all but last chain, backbone_chain_b = last chain
    bb_a = result_ab["backbone_chain_a"]
    bb_b = result_ab["backbone_chain_b"]
    n_prev = sum(chain_lengths[:-1])
    atoms_per_res = 4
    results = []
    offset = 0
    for j, n_res in enumerate(chain_lengths[:-1]):
        n_atoms = n_res * atoms_per_res
        results.append({
            "backbone_atoms": bb_a[offset : offset + n_atoms],
            "sequence": seqs[j],
            "n_res": n_res,
        })
        offset += n_atoms
    results.append({
        "backbone_atoms": bb_b,
        "sequence": seqs[-1],
        "n_res": chain_lengths[-1],
    })
    return _merge_pdb_chains(
        [
            (results[j], chain_ids[j] if j < len(chain_ids) else "A")
            for j in range(len(results))
        ],
        ligands=parsed_ligands or None,
        ligand_refine_quick=False,
    )


def _refine_assembly_chains_with_tree_torque(
    result_complex: dict,
    seq_a: str,
    seq_b: str,
) -> tuple[list, list]:
    """Run tree-torque (assembly_mode=True, further-from-COM first) on each chain; return (backbone_a, backbone_b)."""
    bb_a = result_complex["backbone_chain_a"]
    bb_b = result_complex["backbone_chain_b"]
    if not run_discrete_refinement or len(bb_a) != 4 * len(seq_a) or len(bb_b) != 4 * len(seq_b):
        return bb_a, bb_b
    try:
        ref_a = run_discrete_refinement(
            seq_a,
            initial_backbone_atoms=bb_a,
            run_until_converged=True,
            max_phases_cap=100000,
            assembly_mode=True,
        )
        ref_b = run_discrete_refinement(
            seq_b,
            initial_backbone_atoms=bb_b,
            run_until_converged=True,
            max_phases_cap=100000,
            assembly_mode=True,
        )
        return ref_a.backbone_atoms, ref_b.backbone_atoms
    except Exception:
        return bb_a, bb_b


def _predict_hke_assembly_with_complex(sequences: list[str]) -> tuple[str, str, str] | None:
    """
    For exactly 2 chains: run each chain through HKE-with-funnel; model EM field of each
    compacted protein; find most likely connection points; place and minimize complex;
    then tree-torque refinement (assembly mode: vectors from further from COM first) until no allowed moves.
    Returns (pdb_chain_a, pdb_chain_b, pdb_complex) or None if not 2 chains.
    """
    if len(sequences) != 2:
        return None
    seq_a = _sequence_from_input(sequences[0])
    seq_b = _sequence_from_input(sequences[1])
    if not seq_a or not seq_b:
        return None
    result_a, result_b, result_complex = run_two_chain_assembly_hke(
        seq_a,
        seq_b,
        funnel_radius=FUNNEL_RADIUS,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        funnel_stiffness=1.0,
        hke_max_iter_s1=HKE_ONE_PASS_ITER[0],
        hke_max_iter_s2=HKE_ONE_PASS_ITER[1],
        hke_max_iter_s3=HKE_ONE_PASS_ITER[2],
        converge_max_disp_per_100_res=1.0,
        max_dock_iter=600,
    )
    bb_a, bb_b = _refine_assembly_chains_with_tree_torque(result_complex, seq_a, seq_b)
    pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
    pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        bb_a,
        bb_b,
        result_a["sequence"],
        result_b["sequence"],
        chain_id_a="A",
        chain_id_b="B",
    )
    return (pdb_a, pdb_b, pdb_complex)


def _predict_cartesian_assembly_with_complex(sequences: list[str]) -> tuple[str, str, str] | None:
    """
    Fallback for 2-chain: Cartesian minimizer per chain; placement + complex minimization;
    then tree-torque (assembly mode: further from COM first) until no allowed moves.
    """
    if len(sequences) != 2:
        return None
    seq_a = _sequence_from_input(sequences[0])
    seq_b = _sequence_from_input(sequences[1])
    if not seq_a or not seq_b:
        return None
    result_a = minimize_full_chain(
        seq_a, max_iter=100, long_chain_max_iter=80, include_sidechains=False
    )
    result_b = minimize_full_chain(
        seq_b, max_iter=100, long_chain_max_iter=80, include_sidechains=False
    )
    result_a, result_b, result_complex = run_two_chain_assembly(
        result_a, result_b, max_dock_iter=60, converge_max_disp_per_100_res=1.0
    )
    bb_a, bb_b = _refine_assembly_chains_with_tree_torque(result_complex, seq_a, seq_b)
    pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
    pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        bb_a,
        bb_b,
        result_a["sequence"],
        result_b["sequence"],
        chain_id_a="A",
        chain_id_b="B",
    )
    return (pdb_a, pdb_b, pdb_complex)


@app.route("/health", methods=["GET"])
def health():
    return Response("OK\n", status=200, mimetype="text/plain")


def _status_request_info(job_id: str) -> dict | None:
    """Load request.json for job_id from pending or outputs; return request array without email."""
    for directory in (PENDING_DIR, OUTPUTS_DIR):
        base = _find_base_for_job_id(job_id, directory) or job_id
        path = os.path.join(directory, f"{base}.request.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                req = json.load(f)
        except Exception:
            return None
        sequences = req.get("sequences") or []
        seqs = [_sequence_from_input(s) for s in sequences if s]
        lengths = [len(s) for s in seqs]
        return {
            "num_sequences": len(seqs),
            "lengths": lengths,
            "title": req.get("title") or "",
        }
    return None


@app.route("/status", methods=["GET"])
def status():
    """
    List all jobs: job_id, request info (no email), and status (pending / done / failed).
    """
    _ensure_output_dirs()
    job_ids = set()
    if os.path.isdir(PENDING_DIR):
        for name in os.listdir(PENDING_DIR):
            if name.endswith(".request.json"):
                job_ids.add(name[:- len(".request.json")])
    if os.path.isdir(OUTPUTS_DIR):
        for name in os.listdir(OUTPUTS_DIR):
            if name.endswith(".pdb") and not name.endswith(".fast.pdb"):
                job_ids.add(name[:-4])
            if name.endswith(".failed.txt"):
                job_ids.add(name[:- len(".failed.txt")])
    jobs = []
    for job_id in sorted(job_ids):
        out_pdb = os.path.join(OUTPUTS_DIR, f"{job_id}.pdb")
        out_failed = os.path.join(OUTPUTS_DIR, f"{job_id}.failed.txt")
        if os.path.isfile(out_pdb):
            status_val = "done"
        elif os.path.isfile(out_failed):
            status_val = "failed"
        else:
            status_val = "pending"
        request_info = _status_request_info(job_id)
        jobs.append({
            "job_id": job_id,
            "request": request_info if request_info is not None else {},
            "status": status_val,
        })
    return Response(
        json.dumps({"jobs": jobs}, indent=2),
        status=200,
        mimetype="application/json",
    )


def _run_job_in_background(job_id: str) -> None:
    """Run prediction for job_id in a subprocess with 1h timeout; on timeout or failure, send fast-pass or failure email."""
    base = _find_base_for_job_id(job_id, PENDING_DIR) or job_id
    req_path = os.path.join(PENDING_DIR, f"{base}.request.json")
    txt_path = os.path.join(PENDING_DIR, f"{base}.txt")
    if not os.path.isfile(req_path):
        return
    try:
        with open(req_path) as f:
            req = json.load(f)
    except Exception:
        return
    sequences = req.get("sequences") or []
    seqs = [_sequence_from_input(s) for s in sequences if s]
    if not seqs:
        return
    to_email = (req.get("email") or "").strip()
    job_title = req.get("title")
    ligand_str = (req.get(LIGAND_KEY) or "").strip()
    parsed_ligands = parse_ligands(ligand_str) if ligand_str else []
    if parsed_ligands:
        app.logger.info(
            "Ligand detected via key %s → %d ligands added as 6-DOF agents",
            LIGAND_KEY,
            len(parsed_ligands),
        )
    attempts = _read_pending_attempts(txt_path) if os.path.isfile(txt_path) else 0
    try:
        _run_one_job_with_timeout(
            job_id=job_id,
            base=base,
            sequences=sequences,
            seqs=seqs,
            to_email=to_email,
            job_title=job_title,
            ligand_str=ligand_str,
            txt_path=txt_path,
            attempts=attempts,
        )
    except Exception as e:
        app.logger.warning("Background job %s failed: %s", job_id, e)
        _send_job_failure_email(to_email, job_id, job_title, str(e))


@app.route("/predict", methods=["POST"])
def predict():
    """Accept FASTA in body, JSON, or form (sequence= / fasta= / title= / email=). Queues the job, returns 200 immediately with job_id; prediction runs in background. If email set and SMTP configured, PDB is sent by email when done."""
    sequences = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        seq = data.get("fasta") or data.get("sequence")
        if isinstance(seq, list):
            sequences = [s.strip() for s in seq if s and str(s).strip()]
        elif seq:
            sequences = [str(seq).strip()]
    if sequences is None and request.form:
        # CAMEO: sequence can be repeated for assembly (sequence=...&sequence=...)
        sequences = request.form.getlist("sequence") or request.form.getlist("fasta")
        if not sequences:
            single = request.form.get("sequence") or request.form.get("fasta")
            if single:
                sequences = [single.strip()]
        else:
            sequences = [s.strip() for s in sequences if s and s.strip()]
    if sequences is None:
        raw = request.get_data(as_text=True)
        if raw and raw.strip():
            sequences = [raw.strip()]
    if not sequences:
        return Response("Missing FASTA/sequence in body, JSON, or form 'sequence'/'fasta'\n", status=400, mimetype="text/plain")

    # Normalize to one-letter sequences (strip FASTA headers etc.)
    seqs = [_sequence_from_input(s) for s in sequences]
    seqs = [s for s in seqs if s]
    if not seqs:
        return Response("No valid sequence found.\n", status=400, mimetype="text/plain")

    to_email, job_title = _get_email_and_title()
    ligand_str = _gather_ligand_str(request)
    job_id = _job_id()
    _write_pending_txt(job_id, job_title, to_email, len(seqs), [len(s) for s in seqs])
    _write_pending_request(job_id, sequences, job_title, to_email, ligand_str=ligand_str)

    # Run prediction in background; return 200 immediately once queued
    thread = threading.Thread(target=_run_job_in_background, args=(job_id,), daemon=True)
    thread.start()
    return Response(
        json.dumps({"status": "queued", "job_id": job_id}),
        status=200,
        mimetype="application/json",
    )


# Links for / and /help
REPO_URL = "https://github.com/disregardfiat/protein_folder"
PYHQIV_URL = "https://pypi.org/project/pyhqiv/"
PAPER_DOI_URL = "https://zenodo.org/records/18794890"

IMAGE_SRC = "/assets/images/Uu9Hk.jpg"
IMAGE_CREDIT = "Image credit: Grok (from the provided picture)"


def _commodore_html_page(*, title: str, text: str) -> str:
    """Simple green-on-black HTML wrapper; preserves the plain-text vibe."""
    safe_title = _html.escape(str(title))
    safe_text = _html.escape(str(text))
    safe_img = _html.escape(IMAGE_SRC, quote=True)
    safe_alt = _html.escape("HQIV sparse pi-phase gate map")
    safe_credit = _html.escape(IMAGE_CREDIT)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{safe_title}</title>
    <style>
      :root {{ color-scheme: dark; }}
      body {{
        margin: 0;
        padding: 18px 20px;
        background: #000;
        color: #00ff00;
        font-family: "Courier New", Courier, monospace;
      }}
      pre {{
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
        padding: 0;
        font-size: 14px;
        line-height: 1.35;
      }}
      .imgwrap {{
        margin: 0 0 12px 0;
      }}
      img {{
        max-width: 100%;
        height: auto;
        border: 1px solid #003300;
        image-rendering: pixelated;
      }}
      .credit {{
        margin-top: 6px;
        font-size: 12px;
        opacity: 0.95;
      }}
    </style>
  </head>
  <body>
    <div class="imgwrap">
      <img src="{safe_img}" alt="{safe_alt}" />
      <div class="credit">{safe_credit}</div>
    </div>
    <pre>{safe_text}</pre>
  </body>
</html>
"""


@app.route("/assets/images/<path:filename>", methods=["GET"])
def serve_asset_image(filename: str):
    safe = os.path.basename(filename)
    allowed = {"Uu9Hk.jpg", "favicon.png"}
    if safe not in allowed:
        return Response("Not found\n", status=404, mimetype="text/plain")
    ext = os.path.splitext(safe)[1].lower()
    mimetype = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    asset_path = os.path.join(_root, "site", "assets", "images", safe)
    if not os.path.isfile(asset_path):
        return Response("Not found\n", status=404, mimetype="text/plain")
    return send_file(asset_path, mimetype=mimetype, conditional=True)


@app.route("/help", methods=["GET"])
def help_page():
    """API and reference links."""
    body = f"""HQIV CASP prediction server — help

Submit: POST / or POST /predict with FASTA or form/JSON (sequence=, title=, email=).
Multi-chain: send multiple sequence= values for assembly (chain A, B, C, ...).
Results: PDB in response body; if email is set and SMTP configured, also sent by email.
All predictions are refined with tree-torque (no end flag) until no allowed moves.

Endpoints:
  GET  /       — main page and full algorithm description
  GET  /help   — this message
  GET  /health — liveness
  GET  /status — list jobs: job_id, request (num_sequences, lengths, title; no email), status (pending/done/failed)
  POST / or /predict — structure prediction (Lean tunnel default; CASP_LEGACY_HKE_PIPELINE=1 for legacy)

References:
  Repository:  {REPO_URL}
  pyhqiv:      {PYHQIV_URL}
  Paper (DOI): {PAPER_DOI_URL}
  Lean proofs (hqiv-lean): https://github.com/disregardfiat/hqiv-lean
  Paper PDF (GitHub): see hqiv-lean repo (DOI pending on hal.science).
"""
    html = _commodore_html_page(title="HQIV CASP prediction server (help)", text=body)
    return Response(html, status=200, mimetype="text/html; charset=utf-8")


ALGORITHM_DESCRIPTION = r"""
Prediction algorithm (default: HQIV Lean ribosome tunnel + bulk solvent)

1) Initial fold (default)
   Single chain: co-translational ribosome tunnel with Lean-aligned ε_r(T), pH
   screening, dihedral bias toward the HQIV α basin, post-extrusion EM + tree-torque,
   then (by default) an HQIV-native sparse OSHoracle π-phase gate refine on Cα
   (Lean ``OSHoracleHQIVNative``; disable with CASP_LEAN_OSH_HQIV_NATIVE=0); optional
   Sparse gate map: causal expand (i->i and i+1), dense harmonic reconstruction,
   HQIV-native pi-phase pivot, detect flipped support, prune to flipped indices
   before CA update.
   ligands as 6-DOF rigid bodies (included in the tunnel fold for single chain).
   Multi-chain: each chain folded the same way, merged with a chain–chain offset; ligands
   (if provided) are refined against the **entire** merged complex backbone with the same
   Lean bulk-water screening as the fold, then written as HETATM. No inter-chain docking
   unless CASP_LEGACY_ASSEMBLY_DOCK=1.

   Legacy mode (CASP_LEGACY_HKE_PIPELINE=1): hierarchical HKE + funnel; ligands use
   minimize_full_chain; assembly docking as before.

2) Optional tree-torque refinement (CASP_LEAN_TREE_TORQUE_AFTER=1 only)
   When set, discrete φ/ψ refinement may run after the Lean backbone (no end flag — run until no allowed moves)
   All submissions and retries are piped through discrete φ/ψ refinement with
   EM-field-driven unfreezing until convergence (0 accepts in a phase and EM
   unfreeze cannot unlock any DOF).

   • Discrete DOFs: per-residue φ/ψ in a small set of states (pyhqiv profiles).
   • Deterministic downhill: pick the move with lowest ΔE; accept only if ΔE < 0.
   • Locking: DOFs that cannot improve are locked; when all are locked, use EM
     field torque to decide which to unfreeze and nudge toward a lower-energy
     neighbor.
   • Tree from the ends: build an EM field from the current backbone; compute
     forces; torque τ_i = (r_i - r_anchor) × F_i. When the N- and C-termini are
     "free" (not wrapped into the center), anchors are at both ends; torques are
     combined per residue; residues to unfreeze are ordered from the ends inward
     (min(i, n_res-1-i)). When either terminus is buried (within 0.5 R_g of
     center of mass), switch to a single center-of-mass anchor so the torque
     tree still gives appropriate translations.
   • Re-draw the field: after each unfreeze/nudge, rebuild the backbone from
     the current φ/ψ state and recompute the EM field; iterate until no residue
     is above the torque threshold (or cap per phase).
   • Assembly (A+B, (A+B)+C | ligand): same algorithm, but we add vectors from
     residues further from center of mass first (COM anchor, order by distance
     from COM descending). This keeps the EM field and connection-point logic
     while stepping through refinement from the surface inward.

3) Output
   Final PDB is the converged backbone (and ligands if provided). No end flag:
   refinement runs until no allowed translations remain.
"""


@app.route("/", methods=["GET", "POST"])
def index():
    """GET: info and links. POST: same as /predict (submission URL)."""
    if request.method == "POST":
        return predict()
    body = f"""HQIV CASP server — Lean ribosome tunnel (default) + optional legacy HKE

  Repository:  {REPO_URL}
  pyhqiv:      {PYHQIV_URL}
  Paper (DOI): {PAPER_DOI_URL}
  Lean proofs (hqiv-lean): https://github.com/disregardfiat/hqiv-lean
  Paper PDF (GitHub): see hqiv-lean repo (DOI pending on hal.science).

  POST / or /predict with sequence(s); GET /help for API details.
{ALGORITHM_DESCRIPTION}
"""
    html = _commodore_html_page(title="HQIV CASP server", text=body)
    return Response(html, status=200, mimetype="text/html; charset=utf-8")


# On startup, run pending jobs once in background (retry or send failure email after MAX_ATTEMPTS)
def _start_pending_processor() -> None:
    t = threading.Thread(target=process_pending_jobs, daemon=True)
    t.start()


if not DISABLE_PENDING_STARTUP:
    _start_pending_processor()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, threaded=True)
