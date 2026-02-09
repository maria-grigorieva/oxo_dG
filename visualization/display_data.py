import pysam
from pod5 import Reader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm


def extract_signal_for_fragment(bam_path: Path, pod5_dir: Path, fragment_info: dict) -> dict:
    """
    Извлекает фрагмент сырого сигнала из POD5 на основе данных BAM,
    учитывая логику Dorado (теги 'pi' и 'mv').

    Args:
        bam_path (Path): Путь к BAM-файлу.
        pod5_dir (Path): Путь к директории с POD5-файлами.
        fragment_info (dict): Словарь с информацией о фрагменте.

    Returns:
        dict: Словарь с извлеченными данными о сигнале, или None.
    """
    # Этот код полностью взят из предыдущего ответа,
    # так как он уже содержит корректную логику извлечения данных.
    read_id = fragment_info['read_id']
    dG_start, dG_end = fragment_info['dG_start'], fragment_info['dG_end']
    oxo_dG_start, oxo_dG_end = fragment_info['oxo_dG_start'], fragment_info['oxo_dG_end']

    move_table = None
    try:
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bamfile:
            read = next((r for r in bamfile.fetch(until_eof=True) if r.query_name == read_id), None)

            if read:
                move_table = read.get_tag("mv") if read.has_tag("mv") else None
            else:
                return None
    except pysam.SamtoolsError as e:
        print(f"Ошибка чтения BAM-файла: {e}")
        return None

    if move_table is None:
        print(f"Ошибка: Тег 'mv' отсутствует для рида '{read_id}'. Невозможно сопоставить сигнал.")
        return None

    for pod5_file in pod5_dir.glob("*.pod5"):
        try:
            with Reader(str(pod5_file)) as reader:
                pod5_reads = list(reader.reads(selection=[read_id]))
                if pod5_reads:
                    pod5_read = pod5_reads[0]
                    raw_signal = pod5_read.signal

                    def get_fragment_signal_with_coords(start_base, end_base, mv_table, raw_sig):
                        cumulative_moves = np.cumsum(mv_table)

                        start_sig_pos = cumulative_moves[start_base - 1] if start_base > 0 else 0

                        if end_base >= len(cumulative_moves):
                            end_sig_pos = len(raw_sig)
                        else:
                            end_sig_pos = cumulative_moves[end_base]

                        if start_sig_pos >= end_sig_pos:
                            return [], []

                        return raw_sig[start_sig_pos:end_sig_pos].tolist(), cumulative_moves[
                                                                            start_base:end_base + 1] - start_sig_pos

                    dG_signal, dG_signal_coords = get_fragment_signal_with_coords(dG_start, dG_end, move_table,
                                                                                  raw_signal)
                    oxo_dG_signal, oxo_dG_signal_coords = get_fragment_signal_with_coords(oxo_dG_start, oxo_dG_end,
                                                                                          move_table, raw_signal)

                    return {
                        'read_id': read_id,
                        'dG': fragment_info['dG'],
                        'dG_start': dG_start,
                        'dG_end': dG_end,
                        'dG_raw': dG_signal,
                        'dG_coords': dG_signal_coords,
                        'oxo_dG': fragment_info['oxo_dG'],
                        'oxo_dG_start': oxo_dG_start,
                        'oxo_dG_end': oxo_dG_end,
                        'oxo_dG_raw': oxo_dG_signal,
                        'oxo_dG_coords': oxo_dG_signal_coords
                    }

        except Exception as e:
            print(f"Ошибка при чтении файла {pod5_file.name}: {e}")


def plot_nanopore_signal(signal_data: dict, output_path: Path = Path("signal_plot.png")):
    """
    Строит график сырого сигнала, BAM и нуклеотидов.

    Args:
        signal_data (dict): Словарь с данными сигнала, возвращаемый extract_signal_for_fragment_corrected.
        output_path (Path): Путь для сохранения графика.
    """
    if not signal_data:
        print("Данные для построения графика отсутствуют.")
        return

    dG_raw = signal_data['dG_raw']
    dG_coords = signal_data['dG_coords']
    dG_seq = signal_data['dG']

    oxo_dG_raw = signal_data['oxo_dG_raw']
    oxo_dG_coords = signal_data['oxo_dG_coords']
    oxo_dG_seq = signal_data['oxo_dG']

    # Объединяем сигналы и координаты для удобства
    full_raw = dG_raw + oxo_dG_raw
    full_coords = dG_coords.tolist() + [c + dG_coords[-1] for c in oxo_dG_coords.tolist()]
    full_seq = dG_seq + oxo_dG_seq

    fig, ax = plt.subplots(figsize=(20, 8))

    # Построение графика сырого сигнала
    ax.plot(full_raw, color='b', alpha=0.7)

    # Визуализация нуклеотидов
    for i in range(len(full_seq)):
        # Получаем координаты для i-го нуклеотида
        start_coord = full_coords[i - 1] if i > 0 else 0
        end_coord = full_coords[i] if i < len(full_coords) else len(full_raw)

        # Визуализируем прямоугольник для каждого нуклеотида
        rect = patches.Rectangle(
            (start_coord, ax.get_ylim()[0]),
            end_coord - start_coord,
            ax.get_ylim()[1] - ax.get_ylim()[0],
            facecolor='lightgray',
            alpha=0.2
        )
        ax.add_patch(rect)

        # Подписываем нуклеотид
        ax.text(
            start_coord + (end_coord - start_coord) / 2,
            ax.get_ylim()[1] * 0.9,
            full_seq[i],
            horizontalalignment='center',
            verticalalignment='top',
            fontweight='bold',
            fontsize=10
        )

    ax.set_title(f"Сырой сигнал для рида: {signal_data['read_id']}")
    ax.set_xlabel("Номер отсчёта сигнала")
    ax.set_ylabel("Сила тока")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"График сохранен в {output_path}")


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

    signal_data = extract_signal_for_fragment(BAM_PATH, POD5_DIR, fragment_data)

    if signal_data:
        plot_nanopore_signal(signal_data)
