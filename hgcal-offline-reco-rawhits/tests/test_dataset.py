import os
import sys
import time
from tqdm import tqdm
import yaml
import pytest
from pathlib import Path
from dataset.dataset_maskformers import HGCDataset

@pytest.fixture
def base_path() -> Path:
    """Get the current folder of the test"""
    return Path(__file__).parent


def test_dataset_loading(base_path : Path):
    """Test loading HGCDataset from EOS parquet file."""
    
    # Get EOS path from environment variable
    eos_path = os.environ.get('EOS_PATH', None)
    assert eos_path is not None, "EOS_PATH environment variable is not set. Please set it to the directory containing the parquet files."
    eos_path = Path(eos_path)

    # Find the parquet file in the EOS path
    parquet_files = [f for f in os.listdir(eos_path) if f.endswith('.parquet')]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {eos_path}")
    
    filepath = eos_path / parquet_files[0]
    print(f"Testing with file: {filepath}")
    
    # Load config
    config_path = base_path / '..' / 'configs' / 'base.yaml'
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    targets = config["data"]["targets"]
    scale_dict_path = config["data"]["scale_dict_path"]
    particle_scale_dict_path = config["data"]["particle_scale_dict_path"]
    
    # Initialize dataset
    dataset = HGCDataset(
        filepath,
        targets=targets,
        scale_dict_path=scale_dict_path,
        particle_scale_dict_path=particle_scale_dict_path,
        verbose=False
    )

    nevents = len(dataset)
    
    # Test loading 10 events
    print("Testing dataset loading...")
    start_total = time.time()
    for idx in tqdm(range(nevents), desc="Loading events"):
        print(f"Loading event {idx}")
        
        start = time.time()
        inputs, labels = dataset.__getitem__(idx)
        end = time.time()
        
        print(f"Event {idx} loaded in {end - start:.2f} seconds")
        
        # Basic assertions
        assert inputs is not None, f"Inputs for event {idx} should not be None"
        assert labels is not None, f"Labels for event {idx} should not be None"
    
    end_total = time.time()
    print(f"Total time to load {len(dataset)} events: {end_total - start_total:.2f} seconds")
    print(f"Average time per event: {(end_total - start_total) / nevents:.3f} seconds")
    print("✓ Dataset loading test passed!")


if __name__ == "__main__":
    test_dataset_loading()
