from pathlib import Path

import pandas as pd

from src.robyn_data_validation import DataReviewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WEEKLY_DATA_PATH = str(
    PROJECT_ROOT / "sample_data" / "simulated_data_weekly.csv"
)
SAMPLE_DAILY_DATA_PATH = str(PROJECT_ROOT / "sample_data" / "simulated_data_daily.csv")


def test_read_from_csv():
    dr_weekly = DataReviewer(file_path=SAMPLE_WEEKLY_DATA_PATH, date_frequency="weekly")
    dr_daily = DataReviewer(file_path=SAMPLE_DAILY_DATA_PATH, date_frequency="daily")
    assert dr_weekly.data.shape == (208, 6)
    assert dr_daily.data.shape == (208, 6)
    return None


def test_read_from_df():
    df_weekly = pd.read_csv(SAMPLE_WEEKLY_DATA_PATH)
    df_daily = pd.read_csv(SAMPLE_DAILY_DATA_PATH)
    dr_weekly = DataReviewer(data_frame=df_weekly, date_frequency="weekly")
    dr_daily = DataReviewer(data_frame=df_daily, date_frequency="daily")
    assert dr_weekly.data.shape == (208, 6)
    assert dr_daily.data.shape == (208, 6)
    return None
