import pandas as pd
import pysam
from pathlib import Path
import warnings


def get_all_bam_read_data_dynamic_corrected(bam_path: Path) -> pd.DataFrame:
    """
    Извлекает все доступные данные, включая все теги, из BAM-файла для каждого рида,
    и возвращает их в виде DataFrame.

    Args:
        bam_path (Path): Путь к BAM-файлу.

    Returns:
        pandas.DataFrame: DataFrame, содержащий все данные из BAM-файла.
    """
    if not bam_path.is_file():
        print(f"Ошибка: BAM-файл '{bam_path}' не найден.")
        return pd.DataFrame()

    all_read_data = []

    # Подавляем предупреждения pysam
    pysam.set_verbosity(0)

    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bamfile:
            for read in bamfile.fetch(until_eof=True):
                read_info = {}

                # Извлекаем основные, предсказуемые атрибуты
                attributes_to_extract = [
                    'query_name', 'reference_name', 'reference_start', 'reference_end',
                    'is_reverse', 'is_unmapped', 'mapq', 'cigarstring', 'flag'
                ]

                for attr in attributes_to_extract:
                    try:
                        value = getattr(read, attr)
                        read_info[attr] = value
                    except AttributeError:
                        read_info[attr] = None

                # Извлекаем последовательность и качество, если они доступны
                try:
                    read_info['query_sequence'] = read.query_sequence
                except ValueError:
                    read_info['query_sequence'] = None

                try:
                    read_info['query_qualities'] = read.query_qualities
                except ValueError:
                    read_info['query_qualities'] = None

                # Корректно извлекаем все теги, включая "pi"
                try:
                    for tag, value in read.tags:
                        read_info[f"tag_{tag}"] = value
                except Exception:
                    # Некоторые риды могут не содержать теги
                    continue

                all_read_data.append(read_info)

    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла {bam_path}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(all_read_data)

    # Приводим все столбцы к строковому типу, чтобы избежать ошибок при отображении
    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col]):
            df[col] = df[col].astype(str)
        elif pd.api.types.is_list_like(df[col]):
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(str)

    return df

# --- Пример использования ---
if __name__ == "__main__":
    # Замените на реальный путь к вашему BAM-файлу
    BAM_PATH = Path("../input_files/bam_pass/barcode06/FBA01901_bam_pass_c5d75dbd_b3bf4d07_0.bam")

    bam_df = get_all_bam_read_data_dynamic_corrected(BAM_PATH)

    if not bam_df.empty:
        print(f"Всего извлечено ридов из BAM: {len(bam_df)}")
        print("\nПервые 5 ридов:")
        print(bam_df.head())
        print("\nВсе столбцы в DataFrame:")
        print(bam_df.columns.tolist())
        print("\nТипы данных в DataFrame:")
        print(bam_df.info())
        bam_df.to_csv("bam_inspector.csv", index=False)
    else:
        print("Не удалось извлечь данные из BAM-файла.")


