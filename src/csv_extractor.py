import csv
from typing import List, Dict

class CsvExtractor:
    """
    A class to extract data from a CSV file.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[Dict] = self.extract_data() 

    def extract_data(self):
        """
        Extracts data from the CSV file and returns it as a list of dictionaries.
        """
        extracted_data = []
        with open(self.file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                extracted_data.append(row)
        return extracted_data

    def get_data(self) -> List[Dict]:
        """
        Returns the extracted data.
        """
        return self.data

    def get_column(self, column_name: str) -> List:
        """
        Returns a list of values for the specified column.
        """
        return [row[column_name] for row in self.data if column_name in row]
