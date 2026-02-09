import json
import random
import numpy as np
from uuid import uuid4

def generate_signal(base_mean=460, base_std=10, length=20, shift=0):
    """
    Генерация псевдо-сырого сигнала (имитация значений из pod5).
    shift — добавка для отличия oxoG от G.
    """
    signal = np.random.normal(base_mean + shift, base_std, size=length)
    return np.round(signal).astype(int).tolist()

def generate_seq(length=20):
    """Генерация случайной нуклеотидной последовательности."""
    alphabet = "ACGT"
    return ''.join(random.choice(alphabet) for _ in range(length))

def generate_example():
    """
    Генерация одного примера с dG и oxo_dG фрагментами.
    """
    read_id = str(uuid4())

    # параметры для сигнала
    dG_len = random.randint(12, 20)
    oxo_len = random.randint(15, 25)

    dG_raw = generate_signal(length=dG_len, shift=0)
    oxo_raw = generate_signal(length=oxo_len, shift=random.choice([-5, +5]))

    dG_seq = generate_seq(dG_len)
    oxo_seq = generate_seq(oxo_len)

    dG_start = random.randint(0, 100)
    oxo_start = dG_start + dG_len

    return {
        "read_id": read_id,
        "dG": dG_seq,
        "dG_start": dG_start,
        "dG_end": dG_start + dG_len,
        "dG_raw": dG_raw,
        "oxo_dG": oxo_seq,
        "oxo_dG_start": oxo_start,
        "oxo_dG_end": oxo_start + oxo_len,
        "oxo_dG_raw": oxo_raw,
    }

def generate_dataset(n=1000, output_file="examples.jsonl"):
    """
    Генерация n примеров и сохранение в JSONL.
    """
    with open(output_file, "w") as f:
        for _ in range(n):
            rec = generate_example()
            f.write(json.dumps(rec) + "\n")
    print(f"✅ Сгенерировано {n} примеров и сохранено в {output_file}")

if __name__ == "__main__":
    generate_dataset(5000, "examples.jsonl")
