from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """DataIngestionArtifact class to store data ingestion artifacts"""
    train_file_path: str
    test_file_path: str