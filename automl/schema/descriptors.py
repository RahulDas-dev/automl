from typing import Any, List, Optional, Union

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pydantic import BaseModel, Field, model_validator

from .enums import ColumnType, Dtypes, FeatureType
from .stratagy import ImputationScheme, TargetStat


class ColumnDescriptor(BaseModel):
    name: str
    col_type: ColumnType = Field(default=ColumnType.FEATURES)
    dtype: Dtypes
    feature_type: FeatureType
    count: int
    mean: Optional[float] = Field(default=None)
    median: Optional[float] = Field(default=None)
    mode: Optional[float | int | str] = Field(default=None)
    null_count: int = Field(default=0, ge=0)
    unique_valus: int = Field(default=0)
    imputation_scheme: Optional[ImputationScheme]
    imputation_value: Optional[Union[float, int, str]]
    is_selected: bool = Field(default=True)

    @classmethod
    def build_from_dataset(cls, dataset: pd.DataFrame, colname: str, target_columns: List[str]) -> "ColumnDescriptor":
        count = dataset[colname].count().item()

        if is_numeric_dtype(dataset[colname]):
            mean = dataset[colname].mean().item()
            median = dataset[colname].median().item()
            mode = dataset[colname].mode()[0]
            feature_type = FeatureType.Continuous
            dtypes = Dtypes.from_nptype(dataset[colname].dtype)
        elif is_string_dtype(dataset[colname]) or is_bool_dtype(dataset[colname]):
            mode = dataset[colname].mode()[0]
            mean = None
            median = None
            feature_type = FeatureType.Ordinal
            if is_categorical_dtype(dataset[colname]):
                dtypes = Dtypes.CATEGORICAL
            else:
                dtypes = Dtypes.from_nptype(dataset[colname].dtype)
        elif is_datetime64_any_dtype(dataset[colname]):
            mode = None
            mean = None
            median = None
            feature_type = FeatureType.Nominal
            dtypes = Dtypes.from_nptype(dataset[colname].dtype)
        else:
            raise ValueError(f"Unsupported dtype {dataset[colname].dtype}")
        column_type = ColumnType.TARGET if colname in target_columns else ColumnType.FEATURES
        null_count = dataset[colname].isna().sum().item()
        unique_valus = dataset[colname].nunique()
        if unique_valus == 1:
            if column_type == FeatureType.TARGET:
                raise ValueError(f"Target column {colname} has only one unique value")
            feature_type = FeatureType.constant

        return cls(
            name=colname,
            col_type=column_type,
            dtype=dtypes,
            feature_type=feature_type,
            count=count,
            mean=mean,
            median=median,
            mode=mode,
            null_count=null_count,
            unique_valus=unique_valus,
            imputation_scheme=None,
            imputation_value=None,
        )

    @model_validator(mode="after")
    @classmethod
    def imputation_scheme(cls, values) -> Any:
        dtype = values.dtype
        imp_scheme = values.imputation_scheme
        imp_value = values.imputation_value
        if values.dtype in [Dtypes.INTERGER, Dtypes.FLOAT]:
            if imp_scheme == ImputationScheme.MODE:
                raise ValueError(f"imputation_scheme {imp_scheme.value} is not valid for Dtypes {dtype.name}")
            if imp_scheme == ImputationScheme.VALUE:  # noqa: SIM102
                if not isinstance(imp_value, (int, float)):
                    raise ValueError(f"imputation_value {imp_value} is not valid for Dtypes {dtype.name}")
        elif values.dtype == Dtypes.CATEGORICAL:
            if imp_scheme in [ImputationScheme.MEAN, ImputationScheme.MEDIAN]:
                raise ValueError(f"imputation_scheme {imp_scheme.value} is not valid for Dtypes {dtype.name}")
            if imp_scheme == ImputationScheme.VALUE:  # noqa: SIM102
                if not isinstance(imp_value, str):
                    raise ValueError(f"imputation_value {imp_value} is not valid for Dtypes {dtype.name}")
        elif values.dtype in (Dtypes.BOOLEAN, Dtypes.DATE):
            if imp_scheme in [ImputationScheme.MEAN, ImputationScheme.MEDIAN]:
                raise ValueError(f"imputation_scheme {imp_scheme.value} is not valid for Dtypes {dtype.name}")
            if imp_scheme == ImputationScheme.VALUE:  # noqa: SIM102
                if not isinstance(imp_value, bool):
                    raise ValueError(f"imputation_value {imp_value} is not valid for Dtypes {dtype.name}")
        return values


class DatasetDescriptor(BaseModel):
    row_count: int = Field(default=0, ge=0)
    cloumns_info: List[ColumnDescriptor]
    duplicate_rows: List[int] = Field(default_factory=list)
    duplicate_columns: List[str]

    @classmethod
    def build_from_dataset(cls, dataset: pd.DataFrame, target_columns: List[str]) -> "DatasetDescriptor":
        row_count, col_count = dataset.shape
        duplicate_rows = list(dataset[dataset.duplicated()].index)
        duplicate_columns = list(dataset.columns[dataset.columns.duplicated()])
        cloumns_info = []
        for column in dataset.columns:
            coldesc = ColumnDescriptor.build_from_dataset(dataset, column, target_columns)
            cloumns_info.append(coldesc)
        return DatasetDescriptor(
            row_count=row_count,
            cloumns_info=cloumns_info,
            duplicate_rows=duplicate_rows,
            duplicate_columns=duplicate_columns,
        )

    def outlier_count(self) -> int:
        pass

    def is_imbalance(self) -> bool:
        pass

    def target_statistics(self) -> TargetStat:
        pass
