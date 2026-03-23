#!/usr/bin/env python3
"""
Run the CASP pipeline locally (fast-pass + HKE) with a short sequence to verify
it completes without blowing up. Uses same code path as the server.
Usage: from repo root, CASP_DISABLE_PENDING_STARTUP=1 python scripts/run_pipeline_local.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

# Repo root
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "src"))

# Use a temp output dir so we don't touch real pending/outputs
out_base = os.environ.get("CASP_OUTPUT_DIR")
if not out_base:
    _tmp = tempfile.mkdtemp(prefix="casp_pipeline_test_")
    os.environ["CASP_OUTPUT_DIR"] = _tmp
    _cleanup_tmp = _tmp
else:
    _cleanup_tmp = None

os.environ.setdefault("CASP_DISABLE_PENDING_STARTUP", "1")
# CASP_SMOKE=1: use geometric-only PDB for fast-pass and skip real HKE so run finishes in seconds
SMOKE = os.environ.get("CASP_SMOKE", "").strip().lower() in ("1", "true", "yes")
# USE_FAST_PREDICT=1: skip full HKE in _run_hke_only so run finishes in seconds
if os.environ.get("USE_FAST_PREDICT", "").strip().lower() not in ("1", "true", "yes"):
    os.environ["USE_FAST_PREDICT"] = "1"  # default for local test: quick run
use_fast = True  # we set it above for local test

# Short single-chain sequence (fast to fold)
TEST_SEQ = os.environ.get("CASP_TEST_SEQ", "MKFL")  # 4 residues for quick full run
JOB_ID = "local_test_job"
BASE = JOB_ID  # no email → base same as job_id


def main() -> int:
    import casp_server as server

    pending = server.PENDING_DIR
    outputs = server.OUTPUTS_DIR
    os.makedirs(pending, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    # Pending job files (same layout as POST /predict)
    server._write_pending_txt(JOB_ID, "local pipeline test", None, 1, [len(TEST_SEQ)])
    server._write_pending_request(JOB_ID, [TEST_SEQ], "local pipeline test", None)

    if SMOKE:
        # Pre-create .fast.pdb with geometric-only predictor so we skip slow HKE in fast-pass
        from horizon_physics.proteins.casp_submission import hqiv_predict_structure
        fast_pdb = hqiv_predict_structure(TEST_SEQ, chain_id="A")
        fast_path = os.path.join(pending, f"{BASE}.fast.pdb")
        with open(fast_path, "w") as f:
            f.write(fast_pdb)
        # Run only HKE stage (with USE_FAST_PREDICT it just moves fast-pass to outputs)
        server._run_hke_only(JOB_ID, BASE, [TEST_SEQ], [TEST_SEQ], None, "local pipeline test", [])
        print("Pipeline finished without exception (smoke: geometric fast-pass + HKE path).", flush=True)
    else:
        payload = {
            "job_id": JOB_ID,
            "base": BASE,
            "sequences": [TEST_SEQ],
            "seqs": [TEST_SEQ],
            "to_email": None,
            "job_title": "local pipeline test",
            "ligand_str": "",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            temp_path = f.name

        try:
            mode = "fast-pass only (USE_FAST_PREDICT)" if use_fast else "fast-pass then HKE (single-chain)"
            print(f"Running pipeline: {mode}...", flush=True)
            server._run_one_job_fast_then_hke(temp_path)
            print("Pipeline finished without exception.", flush=True)
        except Exception as e:
            print(f"Pipeline failed: {e}", flush=True)
            raise
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    # Assert outputs
    out_pdb = os.path.join(outputs, f"{BASE}.pdb")
    out_fast = os.path.join(outputs, f"{BASE}.fast.pdb")
    out_models = os.path.join(outputs, f"{BASE}.models.json")
    assert os.path.isfile(out_pdb), f"Missing {out_pdb}"
    assert os.path.isfile(out_fast), f"Missing {out_fast}"
    assert os.path.isfile(out_models), f"Missing {out_models}"

    ok, reason = server._pdb_sanity_check(open(out_pdb).read())
    assert ok, f"Main PDB sanity check failed: {reason}"
    ok2, reason2 = server._pdb_sanity_check(open(out_fast).read())
    assert ok2, f"Fast PDB sanity check failed: {reason2}"

    with open(out_models) as f:
        record = json.load(f)
    assert record.get("job_id") == JOB_ID and record.get("models"), "models.json invalid"

    print("Outputs OK: .pdb, .fast.pdb, .models.json present and valid.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _cleanup_tmp and os.path.isdir(_cleanup_tmp):
            import shutil
            try:
                shutil.rmtree(_cleanup_tmp)
            except Exception:
                pass
