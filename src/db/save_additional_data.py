import csv
import os
from typing import Dict, List, Union, Any
from datetime import datetime


class CSVDataSaver:
    """
    A class to save additional data to CSV files while preserving existing data.
    
    This class can handle dictionary data and append it to existing CSV files,
    automatically managing headers and data formatting.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the CSVDataSaver with a file path.
        
        Args:
            file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.headers = None
        self._load_existing_headers()
    
    def _load_existing_headers(self):
        """Load headers from existing CSV file if it exists."""
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            try:
                with open(self.file_path, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    self.headers = next(reader)
            except (IOError, StopIteration):
                self.headers = None
    
    def save_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        Save data to the CSV file.
        
        Args:
            data: Either a single dictionary or a list of dictionaries to save
        """
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return
        
        # Determine headers from the first data item if not already set
        if self.headers is None:
            self.headers = list(data[0].keys())
            self._write_headers()
        
        # Check if we need to add new headers
        new_headers = set()
        for item in data:
            new_headers.update(item.keys())
        
        if not set(new_headers).issubset(set(self.headers)):
            self._update_headers(new_headers)
        
        # Append data to file
        self._append_data(data)
    
    def _write_headers(self):
        """Write headers to a new CSV file."""
        with open(self.file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)
    
    def _update_headers(self, new_headers: set):
        """Update headers if new columns are detected."""
        # Add new headers that don't exist
        for header in new_headers:
            if header not in self.headers:
                self.headers.append(header)
        
        # Read existing data
        existing_data = []
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                existing_data = list(reader)
        
        # Rewrite file with updated headers
        with open(self.file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()
            for row in existing_data:
                # Fill missing columns with empty strings
                complete_row = {header: row.get(header, '') for header in self.headers}
                writer.writerow(complete_row)
    
    def _append_data(self, data: List[Dict[str, Any]]):
        """Append data rows to the CSV file."""
        with open(self.file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            for item in data:
                # Fill missing columns with empty strings
                complete_row = {header: item.get(header, '') for header in self.headers}
                writer.writerow(complete_row)
    
    def get_headers(self) -> List[str]:
        """Get the current headers of the CSV file."""
        return self.headers.copy() if self.headers else []
    
    def file_exists(self) -> bool:
        """Check if the CSV file exists."""
        return os.path.exists(self.file_path)
    
    def get_row_count(self) -> int:
        """Get the number of data rows in the CSV file (excluding header)."""
        if not self.file_exists():
            return 0
        
        try:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = sum(1 for _ in reader)
                return max(0, row_count - 1)  # Subtract 1 for header
        except IOError:
            return 0


# Example usage and utility functions
def save_trading_data(file_path: str, trading_data: Dict[str, Any]):
    """
    Convenience function to save trading data to CSV.
    
    Args:
        file_path (str): Path to the CSV file
        trading_data (dict): Dictionary containing trading data
    """
    saver = CSVDataSaver(file_path)
    saver.save_data(trading_data)


def save_multiple_records(file_path: str, records: List[Dict[str, Any]]):
    """
    Convenience function to save multiple records to CSV.
    
    Args:
        file_path (str): Path to the CSV file
        records (list): List of dictionaries containing data
    """
    saver = CSVDataSaver(file_path)
    saver.save_data(records)


if __name__ == "__main__":
    # Define the output file path
    output_file = r'C:\Users\user\Code\DeepTradeRL\src\db\example.csv'

    # Initial data
    initial_data = {
        'timestamp': '2024-07-24 10:00:00',
        'open': 60000,
        'high': 61000,
        'low': 59000,
        'close': 60500,
        'volume': 100.5
    }

    # Create a new file with initial data
    saver = CSVDataSaver(output_file)
    saver.save_data(initial_data)

    # Additional data to append
    additional_data = {
        'timestamp': '2024-07-24 11:00:00',
        'open': 60500,
        'high': 62000,
        'low': 60300,
        'close': 61500,
        'volume': 150.2
    }

    # Append the additional data
    saver.save_data(additional_data)

    print(f"Data saved to {saver.file_path}")
    print(f"Current headers: {saver.get_headers()}")
    print(f"Total rows: {saver.get_row_count()}")