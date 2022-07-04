import warnings
from typing import Union

import pandas as pd


class DataReviewer:
    """
    DataReviewer is the class that contains all the functionalities of data review outlined in the Roybyn documentation seen here: https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/#data-review
    Specifically it covers:
    1. Provide descriptive statistics or a basic overview of the collected data inputs (useful to help determine if there is any missing or incomplete data and can help identify the specific variable (e.g. media channel) that requires further investigation)
    2. Help analyse the correlation between all the different variables (multicollinearity detection, expected impact estimation on dependent variable)
    3. Help check for the accuracy of the collected data (spend share and trend)
    """

    def __init__(
        self,
        file_path: str = None,
        data_frame: pd.DataFrame = pd.DataFrame(),
        date_column_name: str = "DATE",
        date_format: Union[str, None] = None,
        date_frequency: str = "weekly",
    ) -> None:
        self.data = self._read_input_data_source(file_path, data_frame)
        # convert the data's date column upfront
        self.date_column_name = date_column_name
        self.data[date_column_name] = pd.to_datetime(
            self.data[date_column_name], format=date_format
        )
        self.date_frequency = date_frequency
        self._check_date_frequency()

    @staticmethod
    def _read_input_data_source(
        file_path: str, data_frame: pd.DataFrame
    ) -> pd.DataFrame:
        if file_path is None:
            if len(data_frame) == 0:
                raise ValueError(
                    "Need to have at least one source of input for the data reviewer to ingest"
                )
            else:
                return data_frame
        else:
            return pd.read_csv(file_path)

    def _check_date_frequency(self, threshold: float = 0.8):
        assert self.date_frequency in [
            "weekly",
            "daily",
        ], "only daily or weekly frequency is currently supported"
        temp_df = self.data.set_index(self.date_column_name)
        if self.date_frequency == "daily":
            counts = temp_df[temp_df.columns[0]].resample("1D").count()
        else:
            counts = temp_df[temp_df.columns[0]].resample("1W").count()
        if (counts == 1).mean() < threshold:
            warnings.warn(
                "Please check if the date frequency is rightly selected or check if there are substantial time gaps in the dataset"
            )
        return None
