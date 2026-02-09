from .fuzzy_search import FuzzySearch, SequenceMatch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

class FastqProcessor:
    def __init__(self, fastq_path: Path, query_dict: Dict[str, str], similarity_threshold: float = 0.9):
        self.fastq_path = fastq_path
        self.query_dict = query_dict
        self.sequence_analyzer = FuzzySearch(similarity_threshold)
        self.max_sequence_length = 0
        self.sequences = None
        self.sequence_lengths = None
        self.lengths_stats = None
        self.total_records = 0
        self.similarity_threshold = similarity_threshold

    def process_file(self) -> List[SequenceMatch]:
        """
        Process a FASTQ file and return a flat list of SequenceMatch objects.
        Each match keeps its corresponding read_id (record.id) and query name.
        """
        all_sequence_matches: List[SequenceMatch] = []
        sequence_lengths = []

        # Count total records for tqdm
        total_records = sum(1 for _ in SeqIO.parse(self.fastq_path, "fastq"))

        # Iterate through FASTQ file
        for record in tqdm(SeqIO.parse(self.fastq_path, "fastq"), total=total_records, desc="Processing FASTQ"):
            read_id = record.id
            sequence = str(record.seq)
            sequence_lengths.append(len(sequence))
            self.max_sequence_length = max(self.max_sequence_length, len(sequence))

            # Find matches in this sequence for all queries
            matches = self.sequence_analyzer.find_matches_in_sequences(
                [sequence], query_dict=self.query_dict
            )

            # Flatten matches (assuming find_matches_in_sequences returns a list of lists)
            for match_list in matches:
                for m in match_list:
                    # Replace seq_id with actual read_id
                    all_sequence_matches.append(
                        SequenceMatch(
                            read_id=read_id,
                            query_name=m.query_name,
                            position=int(m.position),
                            length=int(m.length),
                            score=self.similarity_threshold
                        )
                    )

        # Save some stats
        self.lengths_stats = pd.Series(sequence_lengths).describe()
        self.sequence_lengths = sequence_lengths
        self.total_records = total_records

        return all_sequence_matches

    def save_sequence_matches_to_csv(self, matches: List[SequenceMatch], csv_path: Path):
        """
        Convert a list of SequenceMatch objects to CSV and save.
        """
        # Convert to list of dictionaries
        records = [
            {
                "read_id": m.read_id,  # or m.read_id if that's the field
                "query_name": m.query_name,
                "position": int(m.position),
                "length": int(m.length),
                "score": float(m.score)
            }
            for m in matches
        ]

        # Create DataFrame
        df = pd.DataFrame(records)

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} matches to {csv_path}")

        return df

    def remove_empty_records(self, data: List[List[SequenceMatch]]) -> List[List[SequenceMatch]]:
        to_remove = []
        for idx,match in enumerate(data):
            if len(match) <= 1:
                to_remove.append(idx)
        return [item for i,item in enumerate(data) if i not in to_remove]

    def filter_by_distance(self, data: List[List[SequenceMatch]], threshold: int = 40) -> List[List[SequenceMatch]]:
        filtered_data = []

        for sequence_list in data:
            new_sequence = []
            for i in range(len(sequence_list) - 1):
                current = sequence_list[i]
                next_ = sequence_list[i + 1]
                gap = next_.position - (current.position + current.length)

                if gap >= threshold:
                    new_sequence.append(current)
                    new_sequence.append(next_)

            filtered_data.append(new_sequence)

        return filtered_data

    def matches_stats(self, matches):
        # Dictionary to store statistics
        stats = defaultdict(lambda: {'count': 0, 'total_score': 0, 'total_length': 0})

        total_elements = 0

        # Process the data
        for match_list in matches:
            for match in match_list:
                query = match.query_name
                stats[query]['count'] += 1
                stats[query]['total_score'] += match.score
                stats[query]['total_length'] += match.length
                total_elements += 1

        # Calculate averages
        for query in stats:
            stats[query]['average_score'] = stats[query]['total_score'] / stats[query]['count']
            stats[query]['average_length'] = stats[query]['total_length'] / stats[query]['count']
            # Remove intermediate sums
            del stats[query]['total_score']
            del stats[query]['total_length']

        # Add total elements count
        # stats['total_elements'] = total_elements
        stats['number_of_reads'] = self.total_records

        return dict(stats)

    def compute_pairwise_distances(self, records: List[List[SequenceMatch]]) -> List[List[Tuple[str, str, int]]]:
        """
        For each record (list of SequenceMatch), sort by position and compute distances between neighbors.
        Returns list of lists of tuples (query1, query2, distance).
        """
        all_distances = []

        for record in records:

            filtered_record = [m for m in record if "barcode" not in m.query_name]

            # Skip empty records or records with only one element
            if len(filtered_record) < 2:
                continue

            # Sort by position
            sorted_record = sorted(filtered_record, key=lambda match: match.position)

            # Compute distances between each neighbor
            distances = []
            for i in range(len(sorted_record) - 1):
                m1 = sorted_record[i]
                m2 = sorted_record[i + 1]
                dist = m2.position - m1.position
                distances.append((m1.query_name, m2.query_name, dist))

            all_distances.append(distances)

        return [item for sublist in all_distances for item in sublist]