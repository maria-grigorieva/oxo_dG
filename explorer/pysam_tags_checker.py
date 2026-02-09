import pysam
from pod5 import Reader
from pathlib import Path
import warnings


def get_bam_query_names(bam_path: Path) -> set:
    """
    Извлекает все query_name из BAM-файла и возвращает их в виде множества.

    Args:
        bam_path (Path): Путь к BAM-файлу.

    Returns:
        set: Множество query_name.
    """
    if not bam_path.is_file():
        print(f"Ошибка: BAM-файл '{bam_path}' не найден.")
        return set()

    query_names = set()

    try:
        # Подавляем предупреждения pysam
        pysam.set_verbosity(0)
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bamfile:
            for read in bamfile.fetch(until_eof=True):
                query_names.add(read.query_name)
    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла {bam_path}: {e}")

    return query_names


def get_pod5_read_ids(pod5_dir: Path) -> set:
    """
    Извлекает все read_id из POD5-файлов в директории и возвращает их в виде множества.

    Args:
        pod5_dir (Path): Путь к директории с POD5-файлами.

    Returns:
        set: Множество read_id.
    """
    if not pod5_dir.is_dir():
        print(f"Ошибка: Директория '{pod5_dir}' не найдена.")
        return set()

    read_ids = set()

    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                for read_record in reader.reads():
                    read_ids.add(str(read_record.read_id))
        except Exception as e:
            print(f"Ошибка чтения файла {pod5_file.name}: {e}")

    return read_ids

# --- Пример использования ---
if __name__ == "__main__":
    # Замените на реальные пути
    BAM_PATH = Path("../input_files/merged.bam")
    POD5_DIR = Path("../input_files/pod5")

    # Шаг 1: Извлечение query_name из BAM
    print("Извлечение query_name из BAM...")
    bam_query_names = get_bam_query_names(BAM_PATH)
    print(f"Всего query_name в BAM: {len(bam_query_names)}")

    # Шаг 2: Извлечение read_id из POD5
    print("\nИзвлечение read_id из POD5...")
    pod5_read_ids = get_pod5_read_ids(POD5_DIR)
    print(f"Всего read_id в POD5: {len(pod5_read_ids)}")

    # Шаг 3: Поиск пересечений
    intersection = bam_query_names.intersection(pod5_read_ids)

    print("\n--- Сводка по пересечениям ---")
    print(f"Количество совпадений query_name (BAM) и read_id (POD5): {len(intersection)}")

    if intersection:
        print("Примеры совпадающих ID:", list(intersection)[:5])


