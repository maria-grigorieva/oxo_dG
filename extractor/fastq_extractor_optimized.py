import pandas as pd
import pyarrow.parquet as pq
from Bio import SeqIO


# =====================================
# FAST DNA ops
# =====================================

TRANS = str.maketrans("ATGCatgc", "TACGtacg")

def reverse_complement(seq):
    if seq is None:
        return None
    return seq[::-1].translate(TRANS)


# =====================================
# Load parquet
# =====================================

def load_positions_fast(parquet_path):

    df = pd.read_parquet(parquet_path)

    df.sort_values(["read_id", "position"], inplace=True)

    return df

# =====================================
# FAST extractor
# =====================================

TARGET_PAIRS = [
    ("A1", "A2", "Guanine"),
    ("A2", "A3", "oxoGuanine"),
    ("A3C", "A2C", "realOxoGuanine"),
    ("A2C", "A1C", "realGuanine"),
]

def extract_regions_fast(fastq_path, parquet_path):

    df = load_positions_fast(parquet_path)

    # FASTQ → dict
    reads = {
        r.id: str(r.seq)
        for r in SeqIO.parse(fastq_path, "fastq")
    }

    results = []
    current_read = None
    buffer = []

    for row in df.itertuples(index=False):

        if row.read_id != current_read:

            if buffer:
                process_read(buffer, reads, results)

            buffer = [row]
            current_read = row.read_id

        else:
            buffer.append(row)

    if buffer:
        process_read(buffer, reads, results)

    return results


# =====================================
# Per-read processor
# =====================================

def process_read(rows, reads, results):

    read_id = rows[0].read_id

    seq = reads.get(read_id)
    if seq is None:
        return

    pos = {
        r.query_name: (r.position, r.length)
        for r in rows
    }

    out = {"read_id": read_id}

    for start, end, label in TARGET_PAIRS:

        if start not in pos or end not in pos:
            out[label] = None
            out[label+"_start"] = None
            out[label+"_end"] = None
            continue

        s = pos[start][0] + pos[start][1]
        e = pos[end][0]

        if e <= s:
            out[label] = None
            out[label+"_start"] = None
            out[label+"_end"] = None
            continue

        fragment = seq[s:e]

        # reverse complement только где нужно
        if label in ("realOxoGuanine", "realGuanine"):
            fragment = reverse_complement(fragment)

        out[label] = fragment
        out[label+"_start"] = s
        out[label+"_end"] = e

    results.append(out)
