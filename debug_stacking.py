import sys
import os
import traceback
import pandas as pd

# Add src to sys.path
sys.path.append('src')

from models.stacking import run_stacking
from utils import load_data

def debug():
    try:
        print("Starting diagnostic run of Stacking model...")
        run_stacking()
    except Exception as e:
        print("\n" + "="*50)
        print("CAUGHT EXCEPTION:")
        print("="*50)
        traceback.print_exc()
        
        # Additional diagnostics
        print("\nChecking DataFrame Dtypes specifically for categorical features:")
        train, valid, _ = load_data()
        cat_cols = ['subjects', 'speaker', 'speaker_job', 'state', 'party', 'context']
        print("\nTrain Dtypes:")
        print(train[cat_cols].dtypes)
        print("\nValid Dtypes:")
        print(valid[cat_cols].dtypes)
        
        for col in cat_cols:
            if train[col].dtype != valid[col].dtype:
                print(f"!!! DTYPE MISMATCH in column '{col}': Train={train[col].dtype}, Valid={valid[col].dtype}")

if __name__ == "__main__":
    debug()
