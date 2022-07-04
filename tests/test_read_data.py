from pathlib import Path

import pandas as pd

from robyn_data_validation import DataReviewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = str(PROJECT_ROOT / "sample_data" / "simulated_data.csv")


def test_read_from_csv():
    dr = DataReviewer(file_path=SAMPLE_DATA_PATH)
    assert dr.data.shape == (208, 6)
    return None


def test_read_from_df():
    df = pd.read_csv(SAMPLE_DATA_PATH)
    dr = DataReviewer(data_frame=df)
    assert dr.data.shape == (208, 6)
    return None
