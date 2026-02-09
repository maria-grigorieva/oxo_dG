#!/usr/bin/env python3
import pod5
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

# Входные данные
csv_file = "read_mapping.csv"
pod5_dir = Path("input_files/pod5")
output_dir = Path("input_files/pod5_barcodes")
output_dir.mkdir(exist_ok=True)

# === Загружаем CSV (строки) ===
df = pd.read_csv(csv_file, dtype=str)  # читаем всё как строки
if "read_id" not in df.columns or "barcode" not in df.columns:
    raise SystemExit("CSV должен содержать столбцы 'read_id' и 'barcode'")

id2barcode = dict(zip(df["read_id"], df["barcode"]))

# === Вспомогательные функции ===
def normalize_barcode(barcode):
    """Нормализовать значение barcode: None/empty/'nan' -> 'unclassified'."""
    if barcode is None:
        return "unclassified"
    s = str(barcode).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return "unclassified"
    # при необходимости можно удалить пробелы или неподходящие символы
    return s

writers = {}               # barcode -> pod5.Writer
counts = defaultdict(int)  # barcode -> number of reads written
errors = 0

def get_writer(barcode):
    """Вернуть единственный writer для данного barcode (создать при необходимости)."""
    barcode = normalize_barcode(barcode)
    if barcode not in writers:
        barcode_dir = output_dir / barcode
        barcode_dir.mkdir(parents=True, exist_ok=True)
        out_file = barcode_dir / f"{barcode}.pod5"
        # Открываем writer один раз
        writers[barcode] = pod5.Writer(out_file)
    return writers[barcode]

# === Основной цикл: читаем исходные pod5 и распределяем риды ===
for pod5_file in sorted(pod5_dir.glob("*.pod5")):
    print(f"Processing {pod5_file}...")
    with pod5.Reader(pod5_file) as reader:
        for rec in reader.reads():
            try:
                rid = str(rec.read_id)
                barcode_raw = id2barcode.get(rid, None)
                barcode = normalize_barcode(barcode_raw)

                writer = get_writer(barcode)

                # Ключевая строка: конвертируем ReadRecord -> Read (mutable),
                # затем записываем Read в writer.
                read_obj = rec.to_read()
                writer.add_read(read_obj)

                counts[barcode] += 1

            except Exception as e:
                # Логируем и продолжаем (не прекращаем весь процесс)
                errors += 1
                print(f"Error writing read {locals().get('rid', '?')}: {e}", file=sys.stderr)

# === Закрываем все writer'ы ===
for w in writers.values():
    try:
        w.close()
    except Exception as e:
        print(f"Error closing writer: {e}", file=sys.stderr)

# === Отчёт ===
print("\nFinished. Summary:")
total = 0
for bc, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {bc}: {c} reads")
    total += c
print(f"Total written reads: {total}")
if errors:
    print(f"Errors during processing: {errors}", file=sys.stderr)
print(f"Output directory: {output_dir.resolve()}")