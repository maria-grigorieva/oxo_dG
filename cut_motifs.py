import re
import gzip
import pysam
import pod5
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO

# === 1. Референс и мотив ===
reference = "TCGTGCTAAGCTTTAGGGCCNNNNNNNNTGCNGGCCCNNNNGCNNNGCNNNNGCTCTCGNNGCANNCGAGAGCNGGATCCNGCTCTCGNNTGCNNCGAGAGCNNNNGCNNNGCNNNNGGGCCNGCANNNNNNNNGGCCCNAAAGCTTAGCACGA"
motif = "GGGCCNNNNNNNNTGC"  # пример

# Переводим N в regex
motif_regex = re.compile(motif.replace("N", "[ACGT]"))

# === 2. Чтение FASTQ ===

def read_fastq(fastq_file):
    """
    Reads a FASTQ (optionally gzipped) and returns dict {read_id: sequence}.
    """
    reads = {}

    with open(fastq_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            # record.id → read_id (everything up to first space)
            # record.seq → sequence (as Seq object, convert to str)
            reads[record.id] = str(record.seq)

    return reads

# === 3. Move table из BAM ===
def get_move_tables(bam_file):
    tables = {}
    with pysam.AlignmentFile(bam_file, "rb", check_sq=False) as bam:
        for rec in bam.fetch(until_eof=True):
            rid = rec.query_name
            seq = rec.query_sequence
            if rec.has_tag("mv"):
                mv = rec.get_tag("mv")
                tables[rid] = (seq, mv)
    return tables

# === 4. Сигналы из POD5 ===
def get_signals(pod5_file):
    sigs = {}
    with pod5.Reader(pod5_file) as reader:
        for read in reader.reads():
            sigs[str(read.read_id)] = read.signal
    return sigs

# === 5. Поиск мотива в ридах ===
def extract_fragments(fastq_file, bam_file, pod5_file, motif_regex):
    reads = read_fastq(fastq_file)
    moves = get_move_tables(bam_file)
    signals = get_signals(pod5_file)

    results = []

    for rid, seq in reads.items():
        for m in motif_regex.finditer(seq):
            s, e = m.span()
            subseq = seq[s:e]

            # сигналы
            if rid in moves and rid in signals:
                rseq, mv = moves[rid]
                signal = signals[rid]

                # строим карту нуклеотид → сигнал
                sig_pos = []
                pos = 0
                for step in mv:
                    sig_pos.append(pos)
                    pos += step
                sig_pos.append(len(signal))

                sig_start = sig_pos[s] if s < len(sig_pos) else None
                sig_end = sig_pos[e] if e < len(sig_pos) else None

                frag_signal = signal[sig_start:sig_end] if sig_start and sig_end else None

            else:
                frag_signal = None

            results.append({
                "read_id": rid,
                "motif": motif_regex.pattern,
                "start": s,
                "end": e,
                "seq_fragment": subseq,
                "signal_fragment": frag_signal
            })
            print(f"RID={rid} [{s}:{e}] motif={subseq}, signal_len={len(frag_signal) if frag_signal is not None else 0}")

    return pd.DataFrame(results)


# пример вызова
fastq_file = "/Users/maria/MSU/Nanopores/raw_data/transformed/fastq_pass/barcode06/FBA01901_fastq_pass_c5d75dbd_b3bf4d07_0.fastq"
bam_file = "/Users/maria/MSU/Nanopores/raw_data/transformed/bam_pass/barcode06/FBA01901_bam_pass_c5d75dbd_b3bf4d07_0.bam"
pod5_file = "/Users/maria/MSU/Nanopores/raw_data/pod5_barcodes/barcode06/barcode06.pod5"

df = extract_fragments(fastq_file, bam_file, pod5_file, motif_regex)
print(df)
df.to_csv("motif_fragments.csv", index=False)
