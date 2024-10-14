from pydantic import BaseModel

from .enums import ImputationScheme, OutlierDetectingScheme


class NAStratagy(BaseModel):
    drop_na: bool
    imputation_scheme: ImputationScheme


class ImbalanceStratagy(BaseModel):
    sampling: str


class OutlierStaratagy(BaseModel):
    detection_scheme: OutlierDetectingScheme
    drop_outlier: bool


class CleaningStratagy(BaseModel):
    drop_duplicate_rows: bool
    drop_duplicate_column: bool
    rename_duplicate_columns: bool
    na_handling_stratagy: NAStratagy
    outlier_handling_stratagy: OutlierStaratagy
    imbalance_handling_stratagy: ImbalanceStratagy


class PreprocessingStratagy(BaseModel):
    trnsfrom_binary_column: bool
    remove_datetime_column: bool
    split_datetime_column: bool


class TargetStat(BaseModel):
    target_distributations: str
