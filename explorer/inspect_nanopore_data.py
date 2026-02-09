from pathlib import Path
from pod5 import Reader
import pysam
from Bio import SeqIO
import numpy as np
import pandas as pd
import warnings


def inspect_nanopore_data(bam_path: Path, pod5_dir: Path, fastq_path: Path, max_reads: int = 5):
    """
    Инспектирует и визуализирует взаимосвязь между данными в файлах pod5, bam и fastq.

    Args:
        bam_path (Path): Путь к BAM-файлу.
        pod5_dir (Path): Путь к директории, содержащей POD5-файлы.
        fastq_path (Path): Путь к FASTQ-файлу.
        max_reads (int): Максимальное количество ридов для анализа из каждого файла.
    """
    print("=== Инспекция набора данных Nanopore ===")
    print(f"BAM-файл: {bam_path}")
    print(f"Директория с POD5-файлами: {pod5_dir}")
    print(f"FASTQ-файл: {fastq_path}\n")

    # Подавляем предупреждения pysam, которые могут возникать при работе с файлами
    pysam.set_verbosity(0)

    # --- 1️⃣ Чтение BAM-файла ---
    try:
        # check_sq=False, чтобы избежать ошибки, если BAM-файл не содержит заголовок
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            bam_reads = []
            for i, aln in enumerate(bam.fetch(until_eof=True)):
                if i >= max_reads:
                    break
                tags = {t[0]: t[1] for t in aln.tags}
                bam_reads.append({
                    "read_id": aln.query_name,
                    "ref_name": aln.reference_name,
                    "ref_start": aln.reference_start,
                    "ref_end": aln.reference_end,
                    "strand": "-" if aln.is_reverse else "+",
                    "tags": tags
                })

        print(f"Проанализировано BAM-ридов: {len(bam_reads)}")
        if bam_reads:
            print(pd.DataFrame(bam_reads))
        else:
            print("В BAM-файле не найдено ридов для анализа.")

    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла: {e}")
        bam_reads = []

    # --- 2️⃣ Чтение POD5-файлов ---
    pod5_reads = []
    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                for read_record in reader.reads():
                    pod5_reads.append(read_record.read_id)
                    if len(pod5_reads) >= max_reads:
                        break
            if len(pod5_reads) >= max_reads:
                break
        except Exception as e:
            print(f"Ошибка чтения POD5-файла {pod5_file}: {e}")

    print(f"\nПроанализировано POD5-ридов: {len(pod5_reads)}")
    print("Идентификаторы первых ридов:", pod5_reads[:5])

    # --- 3️⃣ Чтение FASTQ-файла ---
    fastq_reads = []
    try:
        with open(fastq_path, "r") as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fastq")):
                fastq_reads.append(record.id)
                if i >= max_reads - 1:
                    break
    except FileNotFoundError:
        print(f"Ошибка: FASTQ-файл {fastq_path} не найден.")

    print(f"\nПроанализировано FASTQ-ридов: {len(fastq_reads)}")
    print("Идентификаторы первых ридов:", fastq_reads[:5])

    # --- 4️⃣ Подсчет пересечений ---
    bam_ids = {r["read_id"] for r in bam_reads}
    pod5_ids = set(pod5_reads)
    fastq_ids = set(fastq_reads)

    print("\n=== Сводка по пересечениям ===")
    print(f"Общих ридов между BAM и POD5: {len(bam_ids & pod5_ids)}")
    print(f"Общих ридов между BAM и FASTQ: {len(bam_ids & fastq_ids)}")
    print(f"Общих ридов между POD5 и FASTQ: {len(pod5_ids & fastq_ids)}")
    print(f"Общих ридов во всех трех файлах: {len(bam_ids & pod5_ids & fastq_ids)}")

    # --- 5️⃣ Опционально: показать пример сигнала ---
    if pod5_reads:
        example_id = pod5_reads[0]
        print(f"\nПример сигнала для рида: {example_id}")

        # Находим файл pod5, содержащий этот рид (в больших наборах это необходимо)
        containing_pod5 = None
        for pod5_file in pod5_dir.glob("*.pod5"):
            try:
                with Reader(str(pod5_file)) as reader:
                    # Используем reads с selection для эффективного поиска
                    reads_found = list(reader.reads(selection=[example_id]))
                    if reads_found:
                        containing_pod5 = pod5_file
                        example_record = reads_found[0]
                        break
            except Exception:
                continue

        if containing_pod5 and example_record:
            print(f"Найден в файле: {containing_pod5.name}")
            print("Длина сигнала:", len(example_record.signal))
            print("Сигнал (первые 20 точек):", example_record.signal[:20])
        else:
            print("Не удалось найти рид в POD5-файлах.")


# --- Пример использования ---
if __name__ == "__main__":
    # Укажите правильные пути к вашим файлам
    # Замените эти переменные на ваши реальные пути
    BAM_PATH = Path("../input_files/merged.bam")
    POD5_DIR = Path("../input_files/pod5")
    FASTQ_PATH = Path("../input_files/merged.fastq")

    inspect_nanopore_data(bam_path=BAM_PATH, pod5_dir=POD5_DIR, fastq_path=FASTQ_PATH, max_reads=50)

