from enum import Enum
from typing import List

import numpy as np
from pandas.api.types import is_categorical_dtype


def get_valid_values(item: Enum) -> List[str]:
    return [x.value for x in item]


class Dtypes(str, Enum):
    INTERGER = "integer"
    FLOAT = "float"
    BOOLEAN = "bool"
    CATEGORICAL = "categorical"
    DATE = "date"

    @classmethod
    def from_nptype(cls, dtype: np.dtype) -> "Dtypes":
        if is_categorical_dtype(dtype):
            return cls.CATEGORICAL
        if np.issubdtype(dtype, np.integer):
            return cls.INTERGER
        if np.issubdtype(dtype, np.floating):
            return cls.FLOAT
        if np.issubdtype(dtype, np.bool_):
            return cls.BOOLEAN
        if np.issubdtype(dtype, np.datetime64):
            return cls.DATE
        if np.issubdtype(dtype, np.object_) or np.issubdtype(dtype, np.str_):
            return cls.CATEGORICAL
        raise ValueError(f"Unsupported dtype: {dtype}")


class ColumnType(str, Enum):
    FEATURES = "features"
    TARGET = "target"
    INDEX = "index"
    UNIQUEID = "unique-id"


class FeatureType(str, Enum):
    Ordinal = "Ordinal"
    Nominal = "Nominal"
    Continuous = "Continuous"
    constant = "constant"


class ImputationScheme(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    VALUE = "value"


class OutlierDetectingScheme(str, Enum):
    ZSCORE = "Z_Score"
    IQR = "IQR"


class ProjectStatus(str, Enum):
    INIT = "INIT"
    DATALOAD = "DATALOAD"
    VISULIZATION_PRE = "VISULIZATION"
    CLEANING = "CLEANING"
    PREPROCESSING = "PREPROCESSING"
    VISULIZATION_POST = "VISULIZATION_POST"
    MODELING = "MODELING"
    ANALYTICS = "ANALYTICS"
    TESTING = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"


class TaskType(str, Enum):
    REGRESSION_SINGLE_TARGET = "REGRESSION_SINGLE_TARGET"
    REGRESSION_MULTIPLE_TARGET = "REGRESSION_MULTIPLE_TARGET"
    CLASSIFICATION_SINGLE_TARGET_BINARY_CLASS = "CLASSIFICATION_SINGLE_TARGET_BINARY_CLASS"
    CLASSIFICATION_SINGLE_TARGET_MULTI_CLASS = "CLASSIFICATION_SINGLE_TARGET_MULTI_CLASS"
    CLASSIFICATION_MULTIPLE_TARGET = "CLASSIFICATION_MULTIPLE_TARGET"
    CLUSTERING = "CLUSTERING"
    FORECASTING_SINGLE_SERIES = "FORECASTING_SINGLE_SERIES"
    FORECASTING_MULTIPLE_SERIES = "FORECASTING_MULTIPLE_SERIES"


class EXPLAINER(str, Enum):
    FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
    PERMUTACENCE_IMPORTANCE = "PERMUTACENCE_IMPORTANCE"
    SHAP_LOCAL_IMPORTANCE = "SHAP_LOCAL_IMPORTANCEs"
    SHAP_GLOBAL_IMPORTANCE = "SHAP_GLOBAL_IMPORTANCE"
    LIME_IMPORTANCE = "LIME_IMPORTANCE"
