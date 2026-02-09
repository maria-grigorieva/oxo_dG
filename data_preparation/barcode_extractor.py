import pod5
import pysam
from pathlib import Path
import pandas as pd

# Пути к исходным данным
pod5_dir = Path("input_files/pod5")
fastq_dir = Path("input_files/fastq_pass")
bam_dir = Path("input_files/bam_pass")

records = []

# 1. Читаем read_id из всех POD5
for pod5_file in pod5_dir.glob("*.pod5"):
    with pod5.Reader(pod5_file) as reader:
        for read in reader.reads():
            rid = str(read.read_id)
            records.append({
                "read_id": rid,
                "barcode": None,   # пока пусто, добавим позже
                "fastq_file": None,
                "bam_file": None,
                "pod5_file": pod5_file.name
            })

# DataFrame с ридами
df = pd.DataFrame(records).set_index("read_id")

# 2. Чтение read_id из FASTQ
def read_fastq_ids(fastq_path):
    ids = []
    with open(fastq_path) as f:
        for line in f:
            if line.startswith("@"):
                rid = line.split()[0][1:]  # убираем "@"
                ids.append(rid)
    return ids

# 3. Чтение read_id из BAM
def read_bam_ids(bam_path):
    ids = []
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)  # <--- добавляем check_sq=False
    for rec in bam:
        ids.append(rec.query_name)
    bam.close()
    return ids

# 4. Обрабатываем FASTQ по баркодам
for barcode_dir in fastq_dir.iterdir():
    if barcode_dir.is_dir():
        barcode = barcode_dir.name
        for fq in barcode_dir.glob("*.fastq"):
            fq_ids = read_fastq_ids(fq)
            for rid in fq_ids:
                if rid in df.index:
                    df.at[rid, "fastq_file"] = fq.name
                    df.at[rid, "barcode"] = barcode
                else:
                    df.loc[rid] = {
                        "barcode": barcode,
                        "fastq_file": fq.name,
                        "bam_file": None,
                        "pod5_file": None
                    }

# 5. Обрабатываем BAM по баркодам
for barcode_dir in bam_dir.iterdir():
    if barcode_dir.is_dir():
        barcode = barcode_dir.name
        for bam in barcode_dir.glob("*.bam"):
            bam_ids = read_bam_ids(bam)
            for rid in bam_ids:
                if rid in df.index:
                    df.at[rid, "bam_file"] = bam.name
                    if pd.isna(df.at[rid, "barcode"]):
                        df.at[rid, "barcode"] = barcode
                else:
                    df.loc[rid] = {
                        "barcode": barcode,
                        "fastq_file": None,
                        "bam_file": bam.name,
                        "pod5_file": None
                    }

# 6. Сохраняем в CSV
df.reset_index().to_csv("service_files/read_mapping.csv", index=False)

print("Сохранено в read_mapping.csv")