from pathlib import Path
import pandas as pd

def load_data(filename, subfolder=None):
    base_path = Path().resolve().parent / 'base'
    
    if subfolder:
        base_path = base_path / subfolder
        
    file_path = base_path / filename
    
    return pd.read_csv(file_path)