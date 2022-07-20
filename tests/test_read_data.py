from pathlib import Path

import pandas as pd

from src.robyn_data_validation import DataReviewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WEEKLY_DATA_PATH = str(
    PROJECT_ROOT / "sample_data" / "simulated_data_weekly.csv"
)
SAMPLE_DAILY_DATA_PATH = str(PROJECT_ROOT / "sample_data" / "simulated_data_daily.csv")
media_vars = ["tv_S", "radio_S", "paid_search_S"]
extra_vars = ["competitor_sales"]
dependent_var = "revenue"


def test_read_from_csv():
    dr_weekly = DataReviewer(
        paid_media_vars=media_vars,
        paid_media_spends=media_vars,
        extra_vars=extra_vars,
        dep_var=dependent_var,
        file_path=SAMPLE_WEEKLY_DATA_PATH,
        date_frequency="weekly",
    )
    dr_daily = DataReviewer(
        paid_media_vars=media_vars,
        paid_media_spends=media_vars,
        extra_vars=extra_vars,
        dep_var=dependent_var,
        file_path=SAMPLE_DAILY_DATA_PATH,
        date_frequency="daily",
    )
    assert dr_weekly.data.shape == (208, 6)
    assert dr_daily.data.shape == (208, 6)
    return None


def test_read_from_df():
    df_weekly = pd.read_csv(SAMPLE_WEEKLY_DATA_PATH)
    df_daily = pd.read_csv(SAMPLE_DAILY_DATA_PATH)
    dr_weekly = DataReviewer(
        paid_media_vars=media_vars,
        paid_media_spends=media_vars,
        extra_vars=extra_vars,
        dep_var=dependent_var,
        data_frame=df_weekly,
        date_frequency="weekly",
    )
    dr_daily = DataReviewer(
        paid_media_vars=media_vars,
        paid_media_spends=media_vars,
        extra_vars=extra_vars,
        dep_var=dependent_var,
        data_frame=df_daily,
        date_frequency="daily",
    )
    assert dr_weekly.data.shape == (208, 6)
    assert dr_daily.data.shape == (208, 6)
    return None
