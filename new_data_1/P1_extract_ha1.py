#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HA1 from influenza HA amino-acid sequences (H1/H3 friendly),
with:
  - HA1/HA2 cleavage detection (R|K | G[L/I]F...)
  - Optional Signal Peptide (SP) trimming: auto/off/fixed
  - Tail enforcement: remove accidental HA2 head if present in HA1

Features:
- Robust to '-' gaps / 'X' unknowns
- Reports exact decisions in CSV for auditability
"""

import re
import csv
from pathlib import Path
from typing import List, Tuple, Optional

# ---------- Heuristics ----------
EXPECTED_HA1_RANGE = (280, 360)   # gapless aa length typical for HA1 (A/H1/H3)
CLEAVAGE_WINDOW    = (220, 450)   # gapless search window for HA1 length
SP_SCAN_WINDOW     = 35           # scan the first ~35 aa of HA1 for SP cleavage
HCORE_MIN_RUN      = 7            # >=7 hydrophobics in any 10-aa window
HCORE_WIN          = 10

# Common HA2 N-terminus motifs (fallback fuzzy matching)
FUSION_MOTIFS = [
    "GLFGAIAGFIEGGW", "GLFGAIAGFIENGW", "GLFGAIAGFIEGGWT",
    "GLFGAIAGFI", "GLFGAIAGF", "GLFGAIAGL", "GLFGAIAG"
]

HYDROPHOBIC = set("AILMVFWYV")       # simplified hydrophobic set
SMALL_RES   = set("AVSTG")           # small residues preferred at -3,-1
PROHIBIT    = set("P")               # avoid Pro at -3/-1

# ---------- Core utils ----------
def strip_gaps_upper(seq: str) -> Tuple[str, List[int]]:
    """Remove '-' but track original indices."""
    clean, idx_map = [], []
    for i, c in enumerate(seq):
        if c == '-':
            continue
        clean.append(c.upper())
        idx_map.append(i)
    return ''.join(clean), idx_map

def drop_first_n_nongap(seq: str, n: int) -> str:
    """Drop first n non-gap residues while preserving gap pattern."""
    out, consumed = [], 0
    for ch in seq:
        if ch != '-' and consumed < n:
            consumed += 1
            continue
        out.append(ch)
    return ''.join(out)

def looks_like_ha1_only(seq_gapless: str) -> bool:
    n = len(seq_gapless)
    return (EXPECTED_HA1_RANGE[0] <= n <= EXPECTED_HA1_RANGE[1])

# ---------- HA1/HA2 cleavage detection ----------
def find_cleavage_rk_glf(seq_gapless: str) -> Optional[int]:
    """
    Return cut position (gapless idx) AFTER R/K and BEFORE G of GLF.
    """
    for m in re.finditer(r"[RK]GLF", seq_gapless):
        return m.start() + 1
    for m in re.finditer(r"[RK][GQX][LIXV][FYWX]", seq_gapless):
        return m.start() + 1
    for m in re.finditer(r"[RK].{0,2}GLF", seq_gapless):
        return m.start() + 1
    return None

def hamming_leq1(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    return sum(aa != bb for aa, bb in zip(a, b)) <= 1

def find_cleavage_by_fusion_motif(seq_gapless: str) -> Optional[int]:
    lo, hi = CLEAVAGE_WINDOW
    lo = max(0, lo)
    hi = min(len(seq_gapless), hi)
    best = None
    for motif in FUSION_MOTIFS:
        L = len(motif)
        for i in range(lo, hi - L + 1):
            window = seq_gapless[i:i+L]
            if hamming_leq1(window, motif):
                score = 1.0 - abs(i - 330) / 330.0  # prefer ~330
                if (best is None) or (score > best[0]):
                    best = (score, i)
    return best[1] if best else None

def find_ha1_cut(seq: str) -> Tuple[str, Optional[int], str]:
    """
    Return (note, cut_idx_gapless or None, method).
    method in {"already_HA1","RK_GLF","fusion_motif","FAIL"}
    """
    clean, _ = strip_gaps_upper(seq)
    n = len(clean)

    if looks_like_ha1_only(clean):
        return (f"looks_like_HA1_only(n={n})", n, "already_HA1")

    pos = find_cleavage_rk_glf(clean)
    if pos is not None:
        return (f"found_R/K+GLF_at={pos}", pos, "RK_GLF")

    pos2 = find_cleavage_by_fusion_motif(clean)
    if pos2 is not None:
        return (f"found_fusion_motif_at={pos2}", pos2, "fusion_motif")

    return ("motif_not_found", None, "FAIL")

# ---------- SP auto detection (SignalP-like heuristic) ----------
def has_hydrophobic_core(head: str) -> bool:
    n = len(head)
    if n < HCORE_WIN:
        return sum(aa in HYDROPHOBIC for aa in head) >= max(5, HCORE_MIN_RUN - 2)
    for i in range(0, n - HCORE_WIN + 1):
        win = head[i:i+HCORE_WIN]
        if sum(aa in HYDROPHOBIC for aa in win) >= HCORE_MIN_RUN:
            return True
    return False

def find_sp_cleavage(seq_gapless: str) -> Optional[int]:
    """
    Find SPase cleavage after position 'c' (0-based): keep N-terminus of length c.
    Conditions:
      - within first SP_SCAN_WINDOW aa
      - (-3,-1) ∈ SMALL_RES and not Pro
      - upstream region has hydrophobic core
    Typical HA SP length ~16–17.
    """
    n = min(len(seq_gapless), SP_SCAN_WINDOW)
    if n < 12:
        return None
    best = None
    for c in range(12, n):  # earliest plausible cut
        i_m3, i_m1 = c - 3, c - 1
        if i_m3 < 0 or i_m1 < 0:
            continue
        aa_m3, aa_m1 = seq_gapless[i_m3], seq_gapless[i_m1]
        if aa_m3 in PROHIBIT or aa_m1 in PROHIBIT:
            continue
        if aa_m3 not in SMALL_RES or aa_m1 not in SMALL_RES:
            continue
        head = seq_gapless[1:max(2, i_m3)]  # skip Met; use region 1..i_m3
        if not head or not has_hydrophobic_core(head):
            continue
        pref = 1.0 - abs(c - 17) / 17.0
        best = (pref, c) if (best is None or pref > best[0]) else best
    return best[1] if best else None

# ---------- Tail enforcement (avoid HA2 carryover) ----------
def enforce_tail_cleavage(ha1_gapless: str) -> Optional[int]:
    """
    If HA1 tail still contains HA2 head (R/K followed by G[L/I][F/Y]),
    return the correct HA1 length (gapless), i.e., last residue index + 1
    should be the R/K right before the G of GLF/GIF.
    """
    if not ha1_gapless:
        return None
    tail_len = 40
    tail = ha1_gapless[-tail_len:] if len(ha1_gapless) >= tail_len else ha1_gapless
    base = len(ha1_gapless) - len(tail)

    last_rk_idx = None
    for match in re.finditer(r"([RK])G[LI][FYX]", tail):
        last_rk_idx = base + match.start(1)
    if last_rk_idx is not None:
        return last_rk_idx + 1  # include R/K
    return None

def clip_gapless_len_to_gapped_prefix(seq_with_gaps: str, keep_gapless_len: int) -> str:
    """Keep only the first keep_gapless_len non-gap residues (preserve gaps)."""
    out, count = [], 0
    for ch in seq_with_gaps:
        if ch != '-':
            if count >= keep_gapless_len:
                break
            count += 1
        out.append(ch)
    return ''.join(out)

# ---------- FASTA I/O ----------
def read_fasta(path: Path):
    name, seq = None, []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq)
                name = line[1:].strip()
                seq = []
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, ''.join(seq)

def write_fasta(records, path: Path, ungap: bool = False):
    with path.open('w') as f:
        for name, seq in records:
            if ungap:
                seq = seq.replace('-', '')
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


# ---------- In-code configuration (no CLI) ----------
def main():
    # Base directory: the folder containing this script
    base = Path(__file__).resolve().parent

    # In-code parameters (previously provided via CLI)
    # Matches: --in new_data_1/ali.fasta --out_ha1 new_data_1/ha1.fasta
    #          --report new_data_1/report.csv --sp_mode auto --ungap_out
    in_path = base / "ali.fasta"
    out_csv_path = base / "ha1.csv"
    rep_path = base / "report.csv"

    sp_mode = "auto"                 # choices: "off" | "auto" | "fixed"
    sp_len = 16                      # used when sp_mode == "fixed"
    apply_to_already_ha1 = False     # previously --apply_to_already_ha1
    ungap_out = True                 # previously --ungap_out

    out_seq_rows = []  # for CSV: Virus, HA1_Sequence
    rows = []

    for name, seq in read_fasta(in_path):
        raw = seq.strip().upper().replace('U', 'X')
        note, cut_pos_gapless, method = find_ha1_cut(raw)
        clean, idx_map = strip_gaps_upper(raw)

        if method == "FAIL":
            rows.append({
                "name": name, "len_input": len(clean), "len_HA1": "",
                "cut_idx_in_original": "", "motif": "", "note": note, "status": "FAIL"
            })
            continue

        # Build initial HA1
        if method == "already_HA1":
            ha1 = raw
            cut_idx_original = len(raw)
        else:
            if cut_pos_gapless <= 0:
                rows.append({
                    "name": name, "len_input": len(clean),
                    "len_HA1": "", "cut_idx_in_original": "",
                    "motif": "", "note": "bad_cut_position", "status": "FAIL"
                })
                continue
            prev_orig = idx_map[cut_pos_gapless - 1]
            cut_idx_original = prev_orig + 1
            ha1 = raw[:cut_idx_original]

        # ---- Signal peptide trimming ----
        sp_action = "none"
        if sp_mode != "off" and (method != "already_HA1" or apply_to_already_ha1):
            ha1_clean, _ = strip_gaps_upper(ha1)
            sp_cut = None
            if sp_mode == "fixed":
                sp_cut = sp_len
                sp_action = f"fixed:{sp_len}"
            elif sp_mode == "auto":
                sp_cut = find_sp_cleavage(ha1_clean[:SP_SCAN_WINDOW])
                sp_action = f"auto:{sp_cut}" if sp_cut is not None else "auto:none"
            if sp_cut is not None and sp_cut > 0:
                ha1 = drop_first_n_nongap(ha1, sp_cut)

        # ---- Tail enforcement (avoid HA2 contamination) ----
        ha1_clean2, _ = strip_gaps_upper(ha1)
        clip_len = enforce_tail_cleavage(ha1_clean2)
        tail_note = ""
        if clip_len is not None and clip_len < len(ha1_clean2):
            ha1 = clip_gapless_len_to_gapped_prefix(ha1, clip_len)
            tail_note = "|tail_enforced"

        # Prepare output sequence (optionally ungap) and record for CSV
        ha1_out = ha1.replace('-', '') if ungap_out else ha1
        virus_simple = name.split('|', 1)[0].strip()
        # Only keep sequences of length 329 in the CSV
        if len(ha1_out) == 329:
            out_seq_rows.append({"Virus": virus_simple, "HA1_Sequence": ha1_out})
        rows.append({
            "name": name,
            "len_input": len(clean),
            "len_HA1": len(ha1.replace('-', '')),
            "cut_idx_in_original": cut_idx_original,
            "motif": "R/K+GLF" if method == "RK_GLF" else ("fusion_variant" if method == "fusion_motif" else "already_HA1"),
            "note": note + f"|SP={sp_action}" + tail_note,
            "status": "OK"
        })

    # Write final sequences as CSV instead of FASTA
    with out_csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Virus", "HA1_Sequence"])
        w.writeheader()
        w.writerows(out_seq_rows)

    if rep_path:
        with rep_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                "name","len_input","len_HA1","cut_idx_in_original","motif","note","status"
            ])
            w.writeheader()
            w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "OK")
    fail = sum(1 for r in rows if r["status"] == "FAIL")
    print(f"[DONE] HA1 written: {ok}, failed: {fail}")
    if fail:
        print("Some sequences may be partial/atypical. Inspect 'note' column.")
    print("Tips: For stricter uniformity, use --sp_mode fixed --sp_len 16 (H3 common) + tail enforcement.")

if __name__ == "__main__":
    main()

