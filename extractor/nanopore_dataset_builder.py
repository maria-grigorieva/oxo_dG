from pathlib import Path
from pod5 import Reader
import pysam
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm


# -------------------------------------------------
# Нормализация read_id
# -------------------------------------------------

def normalize_read_id(read_id: str) -> str:
    rid = str(read_id).split()[0]

    if rid.startswith("read_id:"):
        rid = rid.replace("read_id:", "")
    return rid


# -------------------------------------------------
# FASTQ -> dict
# -------------------------------------------------

def load_fastq(fastq_path):
    fastq_dict = {}

    print("Loading FASTQ...")

    for record in tqdm(SeqIO.parse(fastq_path, "fastq")):
        rid = normalize_read_id(record.id)

        fastq_dict[rid] = {
            "sequence": str(record.seq),
            "qscore": record.letter_annotations["phred_quality"]
        }

    return fastq_dict


# -------------------------------------------------
# BAM -> dict
# -------------------------------------------------

def load_bam(bam_path):
    bam_dict = {}

    print("Loading BAM...")

    pysam.set_verbosity(0)

    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        for aln in tqdm(bam.fetch(until_eof=True)):

            rid = normalize_read_id(aln.query_name)

            bam_dict[rid] = {
                "ref_name": aln.reference_name,
                "ref_start": aln.reference_start,
                "ref_end": aln.reference_end,
                "strand": "-" if aln.is_reverse else "+"
            }

    return bam_dict


# -------------------------------------------------
# POD5 generator
# (не держим всё в RAM!)
# -------------------------------------------------

def iterate_pod5_reads(pod5_dir):

    pod5_files = list(Path(pod5_dir).glob("*.pod5"))

    for pod5_file in pod5_files:
        with Reader(str(pod5_file)) as reader:

            for read in reader.reads():

                rid = normalize_read_id(read.read_id)

                yield rid, read.signal.astype(np.float32)


# -------------------------------------------------
# Сборка датасета
# -------------------------------------------------

def build_nanopore_table(
        bam_path,
        pod5_dir,
        fastq_path,
        output_path="nanopore_dataset.parquet"
):

    fastq = load_fastq(fastq_path)
    bam = load_bam(bam_path)

    common_ids = set(fastq) & set(bam)

    print(f"\nReads in FASTQ ∩ BAM: {len(common_ids)}")

    rows = []

    print("\nStreaming POD5 and building table...")

    for rid, signal in tqdm(iterate_pod5_reads(pod5_dir)):

        if rid not in common_ids:
            continue

        row = {
            "read_id": rid,
            "signal": signal,
            "sequence": fastq[rid]["sequence"],
            "qscore": fastq[rid]["qscore"],
            **bam[rid]
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\nSaving parquet...")

    df.to_parquet(output_path, engine="pyarrow")

    print(f"\n✅ Dataset saved: {output_path}")
    print("Reads:", len(df))

    return df

# --- Пример использования ---
if __name__ == "__main__":

    BAM_PATH = Path("../input_files/2026_cassette/bam/calls.sorted.bam")
    POD5_DIR = Path("../input_files/2026_cassette/pod5_part")
    FASTQ_PATH = Path("../input_files/2026_cassette/fastq/calls.sorted.fastq")

    df = build_nanopore_table(
        bam_path=BAM_PATH,
        pod5_dir=POD5_DIR,
        fastq_path=FASTQ_PATH,
    )