# pip install pod5
import pod5 as p5
import numpy as np
import os
import csv

pod5_path = "input_files/pod5/FBA01901_c5d75dbd_b3bf4d07_0.pod5"
out_dir = "signals_nA"
os.makedirs(out_dir, exist_ok=True)

with p5.Reader(pod5_path) as reader:
    for read in reader.reads():
        # Получаем ток в pA и конвертируем в nA
        nA_signal = read.signal_pa / 1000.0
        sample_rate = read.run_info.sample_rate
        time = np.arange(len(nA_signal)) / sample_rate

        # Формируем имя файла по read_id
        read_id = str(read.read_id)
        out_path = os.path.join(out_dir, f"{read_id}.csv")

        # Сохраняем CSV: time, nA
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "signal_nA"])
            for t, val in zip(time, nA_signal):
                writer.writerow([t, val])

        print(f"Saved: {out_path}")