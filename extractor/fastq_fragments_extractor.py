from collections import defaultdict
from typing import List, Dict
from Bio import SeqIO
from .fuzzy_search import SequenceMatch
from pathlib import Path


def extract_dG_oxo_dG_fragments(
    fastq_path: Path,
    matches: List[SequenceMatch]
) -> List[Dict]:
    """
    Extract fragments between S1->S2 (dG) and S2->S3 (oxo_dG) for each read.

    Returns a list of dictionaries:
    {
        'read_id': str,
        'dG': str,
        'dG_start': int,
        'dG_end': int,
        'oxo_dG': str,
        'oxo_dG_start': int,
        'oxo_dG_end': int
    }
    """
    # Load sequences
    read_sequences = {r.id: str(r.seq) for r in SeqIO.parse(fastq_path, "fastq")}

    # Group matches by read_id
    matches_by_read = defaultdict(list)
    for m in matches:
        matches_by_read[m.read_id].append(m)

    results = []

    for read_id, match_list in matches_by_read.items():
        if read_id not in read_sequences:
            continue
        seq = read_sequences[read_id]

        # Sort matches by position
        sorted_matches = sorted(match_list, key=lambda x: x.position)

        # Separate S1, S2, S3 positions
        s1_positions = [m.position for m in sorted_matches if m.query_name == "S1"]
        s2_positions = [m.position for m in sorted_matches if m.query_name == "S2"]
        s3_positions = [m.position for m in sorted_matches if m.query_name == "S3"]

        # --- Extract dG fragments (S1 → next S2)
        dG_intervals = []
        s2_idx = 0
        for s1_pos in s1_positions:
            while s2_idx < len(s2_positions) and s2_positions[s2_idx] <= s1_pos:
                s2_idx += 1
            if s2_idx < len(s2_positions):
                s2_pos = s2_positions[s2_idx]
                dG_intervals.append((s1_pos, s2_pos))
                s2_idx += 1  # move to next S2

        # --- Extract oxo_dG fragments (S2 → next S3)
        oxo_dG_intervals = []
        s3_idx = 0
        for s2_pos in s2_positions:
            while s3_idx < len(s3_positions) and s3_positions[s3_idx] <= s2_pos:
                s3_idx += 1
            if s3_idx < len(s3_positions):
                s3_pos = s3_positions[s3_idx]
                oxo_dG_intervals.append((s2_pos, s3_pos))
                s3_idx += 1  # move to next S3

        # Combine fragments as pairs
        # Take min(len(dG_intervals), len(oxo_dG_intervals)) to avoid mismatches
        for i in range(min(len(dG_intervals), len(oxo_dG_intervals))):
            d_start, d_end = dG_intervals[i]
            oxo_start, oxo_end = oxo_dG_intervals[i]
            results.append({
                "read_id": read_id,
                "dG": seq[d_start:d_end],
                "dG_start": d_start,
                "dG_end": d_end,
                "oxo_dG": seq[oxo_start:oxo_end],
                "oxo_dG_start": oxo_start,
                "oxo_dG_end": oxo_end
            })

    return results
