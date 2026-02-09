import pysam
from pod5 import Reader
import numpy as np
from pathlib import Path
import pprint


def extract_signal_for_fragment(bam_path: Path, pod5_dir: Path, fragment_info: dict) -> dict:
    """
    Извлекает фрагмент сырого сигнала из POD5 на основе данных BAM (query_name, mv, pi).

    Args:
        bam_path (Path): Путь к BAM-файлу.
        pod5_dir (Path): Путь к директории с POD5-файлами.
        fragment_info (dict): Словарь с информацией о фрагменте.

    Returns:
        dict: Словарь с извлеченными данными о сигнале.
    """
    read_id = fragment_info['read_id']
    dG_start = fragment_info['dG_start']
    dG_end = fragment_info['dG_end']
    oxo_dG_start = fragment_info['oxo_dG_start']
    oxo_dG_end = fragment_info['oxo_dG_end']

    # --- 1. Поиск рида в BAM и извлечение необходимых данных ---
    move_table = None
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bamfile:
            # Ищем рид по query_name
            read = next((r for r in bamfile.fetch(until_eof=True) if r.query_name == read_id), None)

            if read:
                move_table = read.get_tag("mv") if read.has_tag("mv") else None
            else:
                print(f"Рид '{read_id}' не найден в BAM-файле.")
                return None
    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла: {e}")
        return None

    if move_table is None:
        print(f"Тег 'mv' отсутствует в BAM-файле для рида '{read_id}'. Невозможно сопоставить сигнал.")
        return None

    # --- 2. Сопоставление координат баз с сигналами ---
    base_to_signal_map = {}
    current_signal_pos = 0
    for base_idx, move in enumerate(move_table):
        current_signal_pos += move
        base_to_signal_map[base_idx] = current_signal_pos

    print(move_table)
    print(base_to_signal_map)
    #
    # def get_signal_range_from_coords(start_base, end_base, base_map, signal_len):
    #     start_signal_pos = base_map.get(start_base)
    #     end_signal_pos = base_map.get(end_base + 1)
    #
    #     if start_signal_pos is None:
    #         return None, "Начальная позиция сигнала не найдена."
    #
    #     if end_signal_pos is None:
    #         # Если это последняя база, берем до конца сигнала
    #         end_signal_pos = signal_len
    #     print(f"start_signal_pos = {start_signal_pos}, end_signal_pos = {end_signal_pos}")
    #     return start_signal_pos, end_signal_pos
    #
    # # --- 3. Извлечение сигнала из POD5 ---
    # dG_signal = None
    # oxo_dG_signal = None
    #
    # for pod5_file in pod5_dir.glob("*.pod5"):
    #     try:
    #         with Reader(str(pod5_file)) as reader:
    #             for read_record in reader.reads():
    #                 if str(read_record.read_id) == read_id:
    #                     print('Найдено соответствие!')
    #                     print(read_record.read_id)
    #                     print(read_id)
    #                     raw_signal = read_record.signal
    #                     print(raw_signal)
    #
    #                     signal_len = len(raw_signal)
    #
    #                     # Извлекаем сигнал для dG
    #                     dG_start_pos, dG_end_pos = get_signal_range_from_coords(dG_start, dG_end, base_to_signal_map,
    #                                                                             signal_len)
    #                     if dG_start_pos is not None:
    #                         dG_signal = raw_signal[dG_start_pos:dG_end_pos].tolist()
    #                         print(dG_signal)
    #
    #                     # Извлекаем сигнал для oxo_dG
    #                     oxo_dG_start_pos, oxo_dG_end_pos = get_signal_range_from_coords(oxo_dG_start, oxo_dG_end,
    #                                                                                     base_to_signal_map, signal_len)
    #                     if oxo_dG_start_pos is not None:
    #                         oxo_dG_signal = raw_signal[oxo_dG_start_pos:oxo_dG_end_pos].tolist()
    #                         print(oxo_dG_signal)
    #                     return {
    #                         'read_id': read_id,
    #                         'dG': fragment_info['dG'],
    #                         'dG_start': dG_start,
    #                         'dG_end': dG_end,
    #                         'dG_raw': dG_signal,
    #                         'oxo_dG': fragment_info['oxo_dG'],
    #                         'oxo_dG_start': oxo_dG_start,
    #                         'oxo_dG_end': oxo_dG_end,
    #                         'oxo_dG_raw': oxo_dG_signal
    #                     }
    #
    #     except Exception as e:
    #         print(f"Ошибка при чтении файла {pod5_file.name}: {e}")

    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                for read_record in reader.reads():
                    if str(read_record.read_id) == read_id:
                        print('Найдено соответствие!')
                        print(read_record.read_id)
                        print(read_id)
                        raw_signal = read_record.signal

                        # Локальная функция для извлечения фрагмента
                        def get_fragment_signal(start_base, end_base, mv_table, raw_sig):
                            cumulative_moves = np.cumsum(mv_table)

                            start_sig_pos = cumulative_moves[start_base - 1] if start_base > 0 else 0

                            # Конец сигнала для end_base - это начало следующего нуклеотида.
                            # Если end_base - последний нуклеотид, берем до конца сигнала.
                            if end_base >= len(cumulative_moves):
                                end_sig_pos = len(raw_sig)
                            else:
                                end_sig_pos = cumulative_moves[end_base]

                            # Важная проверка: start_sig_pos не должен быть больше end_sig_pos,
                            # что может случиться из-за нулей в move_table
                            if start_sig_pos >= end_sig_pos:
                                return []  # Возвращаем пустой список, если фрагмент имеет нулевую длину

                            return raw_sig[start_sig_pos:end_sig_pos].tolist()

                        dG_signal = get_fragment_signal(dG_start, dG_end, move_table, raw_signal)
                        oxo_dG_signal = get_fragment_signal(oxo_dG_start, oxo_dG_end, move_table, raw_signal)

                        return {
                            'read_id': read_id,
                            'dG': fragment_info['dG'],
                            'dG_start': dG_start,
                            'dG_end': dG_end,
                            'dG_raw': dG_signal,
                            'oxo_dG': fragment_info['oxo_dG'],
                            'oxo_dG_start': oxo_dG_start,
                            'oxo_dG_end': oxo_dG_end,
                            'oxo_dG_raw': oxo_dG_signal
                        }
        except Exception as e:
            print(f"Ошибка при чтении файла {pod5_file.name}: {e}")

# --- Пример использования ---
if __name__ == "__main__":
    # Замените на реальные пути
    BAM_PATH = Path("../input_files/merged.bam")
    POD5_DIR = Path("../input_files/")

    fragment_data = {
        'read_id': 'b5aaedc3-ac68-4529-8493-d3a72415caf6',
        'dG': 'TCGTGCTGAGCTTTAGGGC',
        'dG_start': 44,
        'dG_end': 63,
        'oxo_dG': 'CGAGAGCAGGATCCATGAAGATCGTGCTCTC',
        'oxo_dG_start': 63,
        'oxo_dG_end': 94
    }
    pprint.pprint(fragment_data)

    result = extract_signal_for_fragment(BAM_PATH, POD5_DIR, fragment_data)

    pprint.pprint(result)

    # if result:
    #     print("\n--- Результат ---")
    #     print(f"Read ID: {result['read_id']}")
    #     print(f"dG фрагмент: {result['dG']}")
    #     print(f"Длина сырого сигнала dG: {len(result['dG_raw']) if result['dG_raw'] else 0}")
    #     print(f"oxo_dG фрагмент: {result['oxo_dG']}")
    #     print(f"Длина сырого сигнала oxo_dG: {len(result['oxo_dG_raw']) if result['oxo_dG_raw'] else 0}")

