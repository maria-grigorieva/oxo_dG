import random

# --- константы ---
A1 = "GATCAGTCCGATATC"
A2 = "TCGACATGCTAGTGC"
A3 = "GCTATCGGATACGTC"
S1 = "ATGACTGCCA"
S2 = "TGGCAGTCAT"  # reverse complement of S1
A1C = "GATATCGGACTGATC"
A2C = "GCACTAGCATGACGA"
A3C = "GACGTATCCGATAGC"

# --- функции ---
bases = ['A', 'T', 'G', 'C']

def random_N(n=4):
    return ''.join(random.choice(bases) for _ in range(n))

def complement(seq):
    comp = {'A':'T','T':'A','G':'C','C':'G'}
    return ''.join(comp[b] for b in seq)

def generate_sequence():
    # N4 блоки
    N1 = random_N(4)
    N2 = random_N(4)
    N3 = random_N(4)
    N4 = random_N(4)
    # N5 = random_N(4)
    # N6 = random_N(4)

    # Комплементарные блоки
    N4C = complement(N4)
    # N5C = complement(N5)
    # N6C = complement(N6)
    N3C = complement(N3)
    N2C = complement(N2)
    N1C = complement(N1)

    # Собираем по заданной схеме
    seq = (
        A1 +
        N1 + "G" + N2 +
        A2 +
        N3 + "G" + N4 +
        A3 +
        S1 + "TTTTTT" + S2 +
        A3C +
        N4C + "C" + N3C +
        A2C +
        N2C + "C" + N1C +
        A1C
    )
    return seq

# --- создаем FASTQ ---
def random_quality(length):
    # имитация Phred-скоростей (Q=30)
    return ''.join(chr(random.randint(33+25, 33+35)) for _ in range(length))

with open("synthetic_reads.fastq", "w") as f:
    for i in range(1000):
        seq = generate_sequence()
        qual = random_quality(len(seq))
        f.write(f"@synthetic_read_{i}\n")
        f.write(seq + "\n+\n")
        f.write(qual + "\n")

print("✅ Создан файл synthetic_reads.fastq с 1000 синтетическими последовательностями.")
