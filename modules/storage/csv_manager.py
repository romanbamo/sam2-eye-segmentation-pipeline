"""
Module: storage/csv_manager.py
Description: Handles reading from and writing to the classification.csv storage file.
             Manages records tracking frame filenames, structural numerical frame IDs,
             and eye visibility state strings ('open' or 'close').
"""

import csv
import os
from typing import Dict, List, Tuple


class CSVManager:
    """Reads and writes the open/close classification CSV file."""

    HEADER = ["frame_filename", "frame_id", "state"]

    @staticmethod
    def write(path: str, rows: List[Tuple[str, int, str]]):
        """
        Pre:
            - path must be a valid string representing a file destination pathway.
            - rows must be a list of tuples containing data matching the structure (frame_filename, frame_id, state).
        Post:
            - Ensures the parent directory path exists, opening the file in write mode.
            - Serializes headers and structured dataset collection strings directly to the target path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSVManager.HEADER)
            for filename, fid, state in rows:
                writer.writerow([filename, fid, state])
        print(f"[CSV] Saved → {path}")

    @staticmethod
    def read(path: str) -> List[Tuple[str, int, str]]:
        """
        Pre:
            - path must be a valid string pointing to an existing and readable target CSV data file.
        Post:
            - Parses the file rows using standard dictionary mappings.
            - Returns a list containing structured data tuples in the format (frame_filename, frame_id, state).
        """
        rows = []
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    (row["frame_filename"], int(row["frame_id"]), row["state"])
                )
        return rows

    @staticmethod
    def read_as_dict(path: str) -> Dict[int, str]:
        """
        Pre:
            - path must be a valid string pointing to an existing and readable target CSV file.
        Post:
            - Converts the file rows into a dictionary structure mapping frame IDs to eye state tokens.
            - Returns a dictionary containing {frame_id: state} pairs for quick runtime lookups.
        """
        result = {}
        for _, fid, state in CSVManager.read(path):
            result[fid] = state
        return result

    @staticmethod
    def read_open_frames(path: str) -> List[str]:
        """
        Pre:
            - path must be a valid string pointing to an existing and readable target CSV file.
        Post:
            - Filters data source contents to isolate records categorized under the 'open' status key.
            - Returns a list containing only the frame filename strings whose classification state is equal to 'open'.
        """
        return [fn for fn, _, state in CSVManager.read(path) if state == "open"]
