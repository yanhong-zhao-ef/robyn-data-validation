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
        date_frequency: str = "weekly",
    ) -> None:
        self.data = self._read_input_data_source(file_path, data_frame)
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

    def _check_date_frequency(self):
        assert self.date_frequency in [
            "weekly",
            "daily",
        ], "only daily or weekly frequency is currently supported"
