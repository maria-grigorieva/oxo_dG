import pandas as pd
from pathlib import Path
from pod5 import Reader


def get_all_pod5_read_data_dynamic(pod5_dir: Path) -> pd.DataFrame:
    """
    Динамически извлекает все доступные данные из всех POD5-файлов
    и возвращает их в виде DataFrame.

    Args:
        pod5_dir (Path): Путь к директории с POD5-файлами.

    Returns:
        pandas.DataFrame: DataFrame, содержащий все данные из файлов POD5.
    """
    if not pod5_dir.is_dir():
        print(f"Ошибка: Директория '{pod5_dir}' не найдена.")
        return pd.DataFrame()

    all_read_data = []
    attributes_to_exclude = ['signal']  # Исключаем "signal", чтобы не перегружать память

    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                for read_record in reader.reads():
                    read_info = {}

                    # Используем dir() для получения списка всех атрибутов
                    attributes = dir(read_record)

                    for attr in attributes:
                        # Игнорируем внутренние и служебные атрибуты
                        if attr.startswith('_') or attr in attributes_to_exclude:
                            continue

                        try:
                            value = getattr(read_record, attr)

                            # Проверяем, является ли атрибут вызываемым (методом)
                            if not callable(value):
                                # Обрабатываем вложенные объекты, если таковые имеются
                                if hasattr(value, '__dict__'):
                                    nested_data = {
                                        f"{attr}_{k}": v for k, v in vars(value).items()
                                    }
                                    read_info.update(nested_data)
                                else:
                                    # Обрабатываем сигнал отдельно, чтобы не перегружать память
                                    if attr == 'signal':
                                        read_info['signal_length'] = len(value)
                                    else:
                                        read_info[attr] = str(value)
                        except Exception as e:
                            # Пропускаем атрибуты, к которым нет доступа
                            continue

                    all_read_data.append(read_info)

        except Exception as e:
            print(f"Ошибка чтения файла {pod5_file.name}: {e}")

    return pd.DataFrame(all_read_data)


# --- Пример использования ---
if __name__ == "__main__":
    # Замените на реальный путь к вашей директории с POD5-файлами
    POD5_DIR = Path("../input_files/pod5")

    pod5_df = get_all_pod5_read_data_dynamic(POD5_DIR)

    if not pod5_df.empty:
        print(f"Всего извлечено ридов из POD5: {len(pod5_df)}")
        print("\nПервые 5 ридов:")
        print(pod5_df.head())
        print("\nВсе столбцы в DataFrame:")
        print(pod5_df.columns.tolist())
        print("\nТипы данных в DataFrame:")
        print(pod5_df.info())
        pod5_df.to_csv("pod5_inspector.csv", index=False)
    else:
        print("Не удалось извлечь данные из POD5-файлов.")

