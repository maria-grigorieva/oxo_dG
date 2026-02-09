from pathlib import Path
import pysam
from pod5 import Reader
from Bio import SeqIO
import numpy as np
import pandas as pd


def explore_nanopore_files(
    bam_path: Path,
    pod5_dir: Path,
    fastq_path: Path,
    max_reads: int = 5
):
    """Печатает информацию о содержимом BAM, POD5 и FASTQ для первых нескольких ридов."""

    print(f"\n🔍 BAM file: {bam_path}")
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    bam_records = []
    for i, read in enumerate(bam.fetch()):
        if i >= max_reads:
            break
        tags = dict(read.tags)
        bam_records.append({
            "read_id": read.query_name,
            "is_reverse": read.is_reverse,
            "query_length": read.query_length,
            "reference_start": read.reference_start,
            "reference_end": read.reference_end,
            "tags": tags
        })
    bam_df = pd.DataFrame(bam_records)
    print("\n🧬 BAM (alignment) overview:")
    print(bam_df)

    print(f"\n📖 FASTQ file: {fastq_path}")
    fastq_records = []
    for i, rec in enumerate(SeqIO.parse(fastq_path, "fastq")):
        if i >= max_reads:
            break
        fastq_records.append({
            "read_id": rec.id,
            "seq_length": len(rec.seq),
            "qual_length": len(rec.letter_annotations["phred_quality"])
        })
    fastq_df = pd.DataFrame(fastq_records)
    print("\n🧾 FASTQ overview:")
    print(fastq_df)

    print(f"\n📂 POD5 directory: {pod5_dir}")
    pod5_records = []
    for pod5_file in Path(pod5_dir).glob("*.pod5"):
        with Reader(pod5_file) as reader:
            for i, read in enumerate(reader):
                if i >= max_reads:
                    break
                move_table = getattr(read, "move_table", None)
                pod5_records.append({
                    "read_id": read.read_id,
                    "signal_length": len(read.signal),
                    "signal_dtype": str(read.signal.dtype),
                    "move_table": len(move_table) if move_table is not None else None
                })
    pod5_df = pd.DataFrame(pod5_records)
    print("\n🔊 POD5 overview:")
    print(pod5_df)

    # 🔗 Попробуем сопоставить read_id между файлами
    print("\n🔗 Cross-file overlap:")
    bam_ids = set(bam_df["read_id"])
    fastq_ids = set(fastq_df["read_id"])
    pod5_ids = set(pod5_df["read_id"])

    print(f"Total BAM reads: {len(bam_ids)}")
    print(f"Total FASTQ reads: {len(fastq_ids)}")
    print(f"Total POD5 reads: {len(pod5_ids)}")
    print(f"Reads shared between BAM & POD5: {len(bam_ids & pod5_ids)}")
    print(f"Reads shared between FASTQ & POD5: {len(fastq_ids & pod5_ids)}")
    print(f"Reads shared between BAM & FASTQ: {len(bam_ids & fastq_ids)}")

    bam.close()

    return {
        "bam_df": bam_df,
        "fastq_df": fastq_df,
        "pod5_df": pod5_df
    }


if __name__ == "__main__":
    # Пример запуска
    bam_path = Path("../input_files/bam_pass/barcode06/FBA01901_bam_pass_c5d75dbd_b3bf4d07_0.bam")
    pod5_dir = Path("../input_files/pod5")
    fastq_path = Path("../input_files/fastq_pass/barcode06/FBA01901_fastq_pass_c5d75dbd_b3bf4d07_0.fastq")

    data = explore_nanopore_files(
        bam_path=bam_path,
        pod5_dir=pod5_dir,
        fastq_path=fastq_path,
        max_reads=5
    )
