import logging
import os
import warnings
from datetime import datetime
from typing import Callable, List, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def number_of_days_in_a_year(year):
    number_of_days = sum([pd.Period(f"{year}-{i}-1").daysinmonth for i in range(1, 13)])
    return number_of_days


def number_of_weeks_in_a_year(year):
    number_of_weeks = pd.Period(f"{year}-12-28").week
    return number_of_weeks


def number_of_days_in_a_month(month, year):
    number_of_days = pd.Period(f"{year}-{month}-1").daysinmonth
    return number_of_days


def number_of_weeks_in_a_month(month, year):
    start_of_month = pd.Timestamp(f"{year}-{month}-1").isocalendar()[1]
    end_of_month = (
        pd.Timestamp(f"{year}-{month}-1") + pd.tseries.offsets.MonthEnd(1)
    ).isocalendar()[1]
    number_of_weeks = end_of_month - start_of_month + 1
    if number_of_weeks < 0:
        number_of_weeks = (
            end_of_month + number_of_weeks_in_a_year(year) - start_of_month + 1
        )
    return number_of_weeks


def color_map(condition):
    if condition:
        return "red"
    else:
        return "grey"


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
        independent_vars: List[str],
        dependent_var: str,
        file_path: str = None,
        data_frame: pd.DataFrame = pd.DataFrame(),
        date_column_name: str = "DATE",
        date_format: Union[str, None] = None,
        date_frequency: str = "weekly",
        review_output_dir: str = "review_output",
    ) -> None:
        self.data = self._read_input_data_source(file_path, data_frame)
        self.independent_vars = independent_vars
        self.dependent_var = dependent_var
        if not os.path.isdir(review_output_dir):
            os.mkdir(review_output_dir)
        output_folder = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = os.path.join(review_output_dir, output_folder)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        # copy raw data over so there is a reference on the data that generated this
        self.data.to_csv(
            os.path.join(self.output_dir, "original_data.csv"), index=False
        )
        # convert the data's date column upfront
        self.date_column_name = date_column_name
        self.data[date_column_name] = pd.to_datetime(
            self.data[date_column_name], format=date_format
        )
        self.date_frequency = date_frequency
        self._check_date_frequency()
        self.logger = logging.getLogger("robyn_data_review")

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

    def _check_date_frequency(self, threshold: float = 0.8) -> None:
        assert pd.api.types.is_datetime64_any_dtype(self.data[self.date_column_name])
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

    def plot_missing_values(self) -> None:
        percent_missing = self.data.isnull().sum() / len(self.data)
        percent_nonmissing = 1 - percent_missing
        data_completeness = pd.DataFrame(
            {
                "Data is complete": percent_nonmissing,
                "Has missing data": percent_missing,
            }
        )
        data_completeness = data_completeness * 100
        ax = data_completeness.plot.barh(
            color={"Data is complete": "grey", "Has missing data": "red"},
            stacked=True,
            figsize=(20, 40),
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        for p in ax.containers[0].patches:
            ax.annotate(
                str(round(p.get_width())) + "%",
                ((p.get_x() + 100) * 1.005, p.get_y() * 1.005),
            )
        plt.xlabel("Examine the variables with missing data (highlighted in red above)")
        plt.ylabel("Varaibles")
        plt.figtext(0.5, 0.01, f"total number of observations is {len(self.data)}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "missing_values_overall.png"))
        return None

    def plot_missing_data_in_a_year(
        self, threshold: float = 0.05, color_map_fun: Callable = color_map
    ) -> None:
        yearly_observation_count = self.data.groupby(
            self.data[self.date_column_name].dt.year
        )[self.date_column_name].count()
        yearly_observation = pd.DataFrame(yearly_observation_count).rename(
            columns={self.date_column_name: "number_of_observations"}
        )
        if self.date_frequency == "weekly":
            yearly_observation["max_number_of_date_unit"] = [
                number_of_weeks_in_a_year(x) for x in yearly_observation.index
            ]
        else:
            yearly_observation["max_number_of_date_unit"] = [
                number_of_days_in_a_year(x) for x in yearly_observation.index
            ]
        yearly_observation["diff_perc"] = (
            yearly_observation.max_number_of_date_unit
            - yearly_observation.number_of_observations
        ) / yearly_observation.max_number_of_date_unit
        colors = list(
            map(color_map_fun, (yearly_observation["diff_perc"] >= threshold).tolist())
        )
        ax = yearly_observation.plot.bar(
            y="number_of_observations",
            rot=0,
            color=colors,
            legend=False,
            figsize=(10, 6),
        )
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),
                (p.get_x() + p.get_width() / 2, p.get_height() * 1.01),
            )
        red_patch = mpatches.Patch(color="red", label=f">= {threshold}%")
        blue_patch = mpatches.Patch(color="grey", label=f"< {threshold}%")
        plt.legend(
            title="% difference vs. max number of observations per year",
            title_fontsize=13,
            prop={"size": 13},
            bbox_to_anchor=(1.02, 1),
            handles=[red_patch, blue_patch],
        )
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "missing_yearly_values.png"))
        return None

    def plot_monthly_tally_of_observations(self, year_to_investigate: int):
        full_df_year = self.data[
            self.data[self.date_column_name].dt.year == year_to_investigate
        ]
        monthly_observation_count = full_df_year.groupby(
            full_df_year[self.date_column_name].dt.month
        )[self.date_column_name].count()
        monthly_observation = pd.DataFrame(monthly_observation_count).rename(
            columns={"full_date": "number_of_observations"}
        )
        if self.date_frequency == "weekly":
            monthly_observation["max_number_of_date_unit"] = [
                number_of_weeks_in_a_month(x, year_to_investigate)
                for x in monthly_observation.index
            ]
        else:
            monthly_observation["max_number_of_date_unit"] = [
                number_of_days_in_a_month(x, year_to_investigate)
                for x in monthly_observation.index
            ]
        ax = monthly_observation.plot.bar(rot=0, figsize=(10, 6))
        ax.legend(
            labels=[
                "Number of observations in a month",
                "Max number of observations in a month",
            ],
            bbox_to_anchor=(1.02, 1),
        )
        for container in ax.containers:
            for p in container.patches:
                ax.annotate(
                    str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005)
                )
        plt.xlabel("Month")
        plt.title(f"Total count of observation for input data in {year_to_investigate}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"missing_monthly_values_of_year_{year_to_investigate}.png",
            )
        )
        return None

    def plot_correlation_heat_map_for_independent_vars(self, fig_size=(16, 12)):
        data_for_indep_correlation = self.data[self.independent_vars]
        # Calculate pairwise-correlation
        matrix = data_for_indep_correlation.corr()
        # Create a mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        # Create a custom divergin palette
        cmap = sns.diverging_palette(
            250, 15, s=75, l=40, n=9, center="light", as_cmap=True
        )
        plt.figure(figsize=fig_size)
        sns.heatmap(
            matrix, mask=mask, center=0, annot=True, fmt=".2f", square=True, cmap=cmap
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"independent_variable_correlation_heatmap.png",
            )
        )
        return None

    def compute_vif(self, threshold=10):
        """Compute Variance Inflation Factor (VIF) for the independent variable to detect multi-colinearity"""
        data_for_vif = self.data[self.independent_vars]
        # VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["feature"] = data_for_vif.columns

        # calculating VIF for each feature
        vif_data["VIF"] = [
            variance_inflation_factor(data_for_vif.values, i)
            for i in range(len(data_for_vif.columns))
        ]
        pct_of_vars_violations = round((vif_data["VIF"] > threshold).mean() * 100)
        self.logger.info(
            f"{pct_of_vars_violations}% of the independent variables violate the vif threshold out of {len(vif_data['VIF'])} variables"
        )
        vif_data.to_csv(
            os.path.join(
                self.output_dir,
                f"vif_independent_vars.csv",
            )
        )
        return vif_data

    def plot_correlation_dep_var(self, fig_size=(16, 12)):
        columns_to_include = self.independent_vars + [self.dependent_var]
        correlation_to_dep = self.data[columns_to_include].corr()[self.dependent_var]
        plt.figure(figsize=fig_size)
        ax = (
            correlation_to_dep[correlation_to_dep.index != self.dependent_var]
            .round(2)
            .sort_values(ascending=False)
            .plot.bar()
        )
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height()))
        plt.title("Correlation with dependent varialbe")
        plt.xlabel(
            "Examine the variables with high correlation to see if it is expected or not"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"correlation_between_dependent_vars_and_independent_vars.png",
            )
        )
        return None

    def run_review(self):
        self.plot_missing_values()
        self.plot_missing_data_in_a_year()
        unique_years = self.data[self.date_column_name].dt.year.unique()
        for year in unique_years:
            self.plot_monthly_tally_of_observations(year)
        self.plot_correlation_heat_map_for_independent_vars()
        self.compute_vif()
        self.plot_correlation_dep_var()
        return None
