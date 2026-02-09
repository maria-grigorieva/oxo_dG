from pod5 import Reader
import pysam
from pathlib import Path


def get_pod5_read_from_bam(bam_path: Path, pod5_dir: Path, bam_read_id: str):
    """
    Находит родительский рид в POD5-файле по дочернему риду из BAM-файла.

    Args:
        bam_path (Path): Путь к BAM-файлу.
        pod5_dir (Path): Путь к директории с POD5-файлами.
        bam_read_id (str): query_name дочернего рида из BAM.

    Returns:
        pod5.ReadRecord: Объект рида из POD5 или None, если не найден.
    """
    # 1. Найти родительский read_id в BAM-файле
    parent_read_id = None
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bamfile:
            for aln in bamfile.fetch(until_eof=True):
                if aln.query_name == bam_read_id:
                    if aln.has_tag("pi"):
                        parent_read_id = aln.get_tag("pi")
                    else:
                        parent_read_id = aln.query_name  # Fallback for older data
                    break
    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла: {e}")
        return None

    if not parent_read_id:
        print(f"Не найден рид '{bam_read_id}' в BAM-файле или отсутствует тег 'pi'.")
        return None

    # 2. Найти и вернуть рид из POD5-файла по родительскому ID
    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                pod5_reads = list(reader.reads(selection=[parent_read_id]))
                if pod5_reads:
                    print(f"Найден родительский рид '{parent_read_id}' в файле {pod5_file.name}")
                    return pod5_reads[0]
        except Exception as e:
            print(f"Ошибка при чтении POD5-файла {pod5_file.name}: {e}")

    print(f"Родительский рид '{parent_read_id}' не найден в POD5-файлах.")
    return None


# Пример использования
if __name__ == "__main__":
    BAM_PATH = Path("../input_files/bam_pass/barcode06/FBA01901_bam_pass_c5d75dbd_b3bf4d07_0.bam")
    POD5_DIR = Path("../input_files/pod5/")

    # Пример дочернего рида из BAM
    child_bam_read_id = "43c19983-75c1-40cd-9a17-aa900e141993"

    pod5_record = get_pod5_read_from_bam(BAM_PATH, POD5_DIR, child_bam_read_id)

    if pod5_record:
        print(f"Успешно извлечены данные из POD5 для рида '{pod5_record.read_id}'")
        print(f"Длина сигнала: {len(pod5_record.signal)}")