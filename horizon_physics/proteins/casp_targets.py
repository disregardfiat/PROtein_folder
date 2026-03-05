"""
CASP known targets and experimental structures from predictioncenter.org.

Fetches target sequences from the Prediction Center download area and optional
experimental PDB codes (from target list descriptions). Downloads experimental
PDBs from RCSB/wwPDB so we are on equal footing with other teams and can run
our A+B assembly strategy with the same reference data.

Usage:
  - Sync known targets: fetch_known_targets(casp_round="CASP16", cache_dir=...)
  - Look up by sequence: get_target_for_sequence(seq, known_targets) -> (target_id, pdb_code)
  - Get experimental PDB path: fetch_experimental_pdb(pdb_code, cache_dir)
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PREDICTION_CENTER_BASE = "https://www.predictioncenter.org/download_area"
# RCSB/wwPDB: lowercase PDB ID
RCSB_DOWNLOAD = "https://files.rcsb.org/download"


@dataclass
class CASPTarget:
    """One CASP target: id, type, sequence(s), optional PDB code when released."""
    target_id: str
    target_type: str  # Prot, H (multimer), R (RNA), etc.
    sequences: List[str]
    pdb_code: Optional[str] = None
    description: str = ""


def _fetch_url(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "HQIV-CASP-Server/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_seq_file(content: str) -> List[CASPTarget]:
    """
    Parse Prediction Center sequence file (e.g. casp16.T1.seq.txt).
    Lines: >T1201 description|  then sequence (one line per residue block).
    """
    targets: List[CASPTarget] = []
    current_id: Optional[str] = None
    current_desc: str = ""
    current_seq: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None and current_seq:
                seq = "".join(c for c in "".join(current_seq).upper() if c.isalpha())
                if seq:
                    targets.append(CASPTarget(
                        target_id=current_id,
                        target_type="Prot",
                        sequences=[seq],
                        description=current_desc,
                    ))
            # New entry: >T1201 Q9GZX9, Human, 210 residues|
            parts = line[1:].split("|", 1)
            header = parts[0].strip()
            current_desc = parts[1].strip() if len(parts) > 1 else ""
            # First token is target id (e.g. T1201, H1202, T0208s1)
            tok = header.split()
            current_id = tok[0] if tok else ""
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None and current_seq:
        seq = "".join(c for c in "".join(current_seq).upper() if c.isalpha())
        if seq:
            targets.append(CASPTarget(
                target_id=current_id,
                target_type="Prot",
                sequences=[seq],
                description=current_desc,
            ))
    return targets


def _extract_pdb_from_description(desc: str) -> Optional[str]:
    """Extract PDB code from description, e.g. 'Q9GZX9 PDB code [8bwd]' -> 8bwd."""
    m = re.search(r"PDB\s+code\s+\[([a-zA-Z0-9]{4})\]", desc, re.IGNORECASE)
    return m.group(1).lower() if m else None


def fetch_sequences_for_round(
    casp_round: str = "CASP16",
    kinds: Optional[List[str]] = None,
) -> List[CASPTarget]:
    """
    Fetch target sequences from predictioncenter.org for a CASP round.
    kinds: e.g. ["T1", "H1"] for tertiary (monomer) and multimer (H) sequence files.
    Returns list of CASPTarget (pdb_code left None; fill from target list if needed).
    """
    if kinds is None:
        kinds = ["T1", "H1"]
    base_url = f"{PREDICTION_CENTER_BASE}/{casp_round}/sequences"
    # Filenames: casp16.T1.seq.txt, casp16.H1.seq.txt
    prefix = casp_round.lower()
    all_targets: Dict[str, CASPTarget] = {}
    for k in kinds:
        fname = f"{prefix}.{k}.seq.txt"
        url = f"{base_url}/{fname}"
        try:
            content = _fetch_url(url)
        except (urllib.error.URLError, OSError):
            continue
        for t in _parse_seq_file(content):
            t.target_type = "H" if k == "H1" else "Prot"
            all_targets[t.target_id] = t
    return list(all_targets.values())


def fetch_target_list_html(casp_round: str = "CASP16") -> str:
    """Fetch target list HTML to scrape PDB codes from descriptions."""
    url = f"https://www.predictioncenter.org/{casp_round.lower()}/targetlist.cgi?view=all&phase=all"
    return _fetch_url(url)


def extract_pdb_codes_from_target_list(html: str) -> Dict[str, str]:
    """
    Parse target list HTML for target id and PDB code.
    Rows contain target id (e.g. T1201) and description with PDB code as link or [xxxx].
    Returns dict target_id -> pdb_code (lowercase).
    """
    out: Dict[str, str] = {}
    # PDB code appears as "PDB code [8bwd]" or "PDB code <a href=...>8bwd</a>"
    pdb_pattern = re.compile(
        r"PDB\s+code\s+(?:<[^>]+>)?\[?([a-zA-Z0-9]{4})\]?(?:</a>)?",
        re.IGNORECASE,
    )
    # Target id: T/H/R/M/L + digits + optional suffix (s1, v1, etc.)
    target_pattern = re.compile(r"([THRML]\d{4}[a-z0-9v]*(?:s\d+|v\d+)?)")
    # Split by table rows to keep target and PDB in same row
    for row in re.split(r"</tr>", html, flags=re.IGNORECASE):
        targets_in_row = target_pattern.findall(row)
        pdbs_in_row = pdb_pattern.findall(row)
        if targets_in_row and pdbs_in_row:
            # First target id in row, last PDB in row (description column often has one PDB)
            out[targets_in_row[0]] = pdbs_in_row[-1].lower()
    return out


def fetch_known_targets(
    casp_round: str = "CASP16",
    cache_dir: Optional[str] = None,
    fill_pdb_codes: bool = True,
) -> List[CASPTarget]:
    """
    Fetch known CASP targets (sequences + optional PDB codes) and optionally cache.
    If cache_dir is set, writes sequences to cache_dir/known_targets.txt and
    pdb_codes to cache_dir/known_targets_pdb.json (or similar).
    """
    targets = fetch_sequences_for_round(casp_round, kinds=["T1", "H1"])
    if fill_pdb_codes:
        try:
            html = fetch_target_list_html(casp_round)
            pdb_map = extract_pdb_codes_from_target_list(html)
            for t in targets:
                t.pdb_code = pdb_map.get(t.target_id)
        except (urllib.error.URLError, OSError):
            pass
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Write a simple index: target_id, type, seq_len, pdb_code
        index_path = os.path.join(cache_dir, "known_targets_index.txt")
        with open(index_path, "w") as f:
            for t in targets:
                seq0 = t.sequences[0] if t.sequences else ""
                f.write(f"{t.target_id}\t{t.target_type}\t{len(seq0)}\t{t.pdb_code or '-'}\n")
    return targets


def get_target_for_sequence(
    sequence: str,
    known_targets: List[CASPTarget],
    exact_only: bool = True,
) -> Optional[Tuple[CASPTarget, int]]:
    """
    Find a known CASP target whose sequence matches the given one.
    sequence: one-letter amino acid sequence (no spaces).
    Returns (target, chain_index) if found, else None. chain_index is 0 for single-chain.
    """
    seq = "".join(c for c in sequence.upper() if c.isalpha())
    if not seq:
        return None
    for t in known_targets:
        for i, s in enumerate(t.sequences):
            s_clean = "".join(c for c in s.upper() if c.isalpha())
            if exact_only and s_clean == seq:
                return (t, i)
            if not exact_only and seq in s_clean and len(seq) >= 0.9 * len(s_clean):
                return (t, i)
    return None


def get_target_for_sequences(
    sequences: List[str],
    known_targets: List[CASPTarget],
) -> Optional[Tuple[CASPTarget, List[int]]]:
    """
    Find a known multimer target whose chain sequences match (in order).
    Returns (target, [chain_indices]) or None.
    """
    if not sequences:
        return None
    seqs_clean = ["".join(c for c in s.upper() if c.isalpha()) for s in sequences]
    for t in known_targets:
        if len(t.sequences) < len(seqs_clean):
            continue
        indices: List[int] = []
        for sq in seqs_clean:
            found = False
            for i, s in enumerate(t.sequences):
                if i in indices:
                    continue
                s_clean = "".join(c for c in s.upper() if c.isalpha())
                if s_clean == sq:
                    indices.append(i)
                    found = True
                    break
            if not found:
                break
        if len(indices) == len(seqs_clean):
            return (t, indices)
    return None


def fetch_experimental_pdb(
    pdb_code: str,
    cache_dir: str,
    timeout: int = 30,
) -> Optional[str]:
    """
    Download experimental PDB from RCSB and save to cache_dir.
    Returns path to cached PDB file, or None on failure.
    """
    pdb_code = pdb_code.lower().strip()
    if len(pdb_code) != 4:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_code}.pdb")
    if os.path.isfile(path):
        return path
    url = f"{RCSB_DOWNLOAD}/{pdb_code}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HQIV-CASP-Server/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        return path
    except (urllib.error.URLError, OSError):
        return None


def ensure_experimental_ref(
    target: CASPTarget,
    cache_dir: str,
) -> Optional[str]:
    """If target has a PDB code, fetch and return path to experimental PDB; else None."""
    if not target.pdb_code:
        return None
    return fetch_experimental_pdb(target.pdb_code, cache_dir)
