import pandas as pd
from Bio import SeqIO

# ==========================
#   Основные функции
# ==========================

def complement(seq: str) -> str:
    """Комплементарная последовательность."""
    table = str.maketrans("ATGCatgc", "TACGtacg")
    return seq.translate(table)

def reverse_complement(seq: str | None) -> str | None:
    """Обратная комплементарная последовательность."""
    if seq is None:
        return None
    return complement(seq[::-1])

def extract_between(seq, start_name, end_name, positions):
    """Возвращает участок между двумя известными последовательностями и его координаты."""
    start_row = positions[positions["query_name"] == start_name]
    end_row = positions[positions["query_name"] == end_name]
    if start_row.empty or end_row.empty:
        return None, None, None
    start = int(start_row["position"].values[0]) + int(start_row["length"].values[0])
    end = int(end_row["position"].values[0])
    if end > start:
        return seq[start:end], start, end
    return None, None, None

# ==========================
#   Главная функция анализа
# ==========================

def extract_regions(fastq_path: str, table_path: str, sep: str = ",") -> list[dict]:
    """
    Извлекает участки G, oxoG, realG, realOxoG из ридов и возвращает список словарей.

    Параметры:
        fastq_path: путь к FASTQ файлу
        table_path: путь к CSV файлу с позициями известных блоков
        sep: разделитель в таблице (по умолчанию — табуляция)

    Возвращает:
        list[dict]: [
            {
                'read_id': str,
                'G': str, 'G_start': int, 'G_end': int,
                'oxoG': str, 'oxoG_start': int, 'oxoG_end': int,
                'realG': str, 'realG_start': int, 'realG_end': int,
                'realOxoG': str, 'realOxoG_start': int, 'realOxoG_end': int
            },
            ...
        ]
    """
    df = pd.read_csv(table_path, sep=sep)
    reads = {record.id: str(record.seq) for record in SeqIO.parse(fastq_path, "fastq")}
    read_groups = df.groupby("read_id")

    results = []

    for read_id, group in read_groups:
        if read_id not in reads:
            continue
        seq = reads[read_id]
        group = group.sort_values("position")

        G, G_start, G_end = extract_between(seq, "A1", "A2", group)
        oxoG, oxoG_start, oxoG_end = extract_between(seq, "A2", "A3", group)
        realOxoG_raw, realOxoG_start, realOxoG_end = extract_between(seq, "A3C", "A2C", group)
        realG_raw, realG_start, realG_end = extract_between(seq, "A2C", "A1C", group)

        realOxoG = reverse_complement(realOxoG_raw) if realOxoG_raw else None
        realG = reverse_complement(realG_raw) if realG_raw else None

        results.append({
            "read_id": read_id,
            "Guanine": G,
            "Guanine_start:": G_start,
            "Guanine_end": G_end,
            "oxoGuanine": oxoG,
            "oxoGuanine_start": oxoG_start,
            "oxoGuanine_end": oxoG_end,
            "realGuanine": realG,
            "realGuanine_start": realG_start,
            "realGuanine_end": realG_end,
            "realOxoGuanine": realOxoG,
            "realOxoGuanine_start": realOxoG_start,
            "realOxoGuanine_end": realOxoG_end,
        })

    return results

