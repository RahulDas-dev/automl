from typing import List, Union

from common import as_list, check_column_length, check_X_for_type
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Allows dropping specific columns from a pandas DataFrame by name.
    :param columns: column name ``str`` or list of column names to be selected
    .. note::
        Raises a ``TypeError`` if input provided is not a DataFrame
        Raises a ``ValueError`` if columns provided are not in the input DataFrame
    :Example:
    """

    def __init__(self, columns: Union[str, List[str]]):
        self.columns = columns

    def fit(self, X, y=None):
        """
        Checks
        1) if input is a DataFrame,
        2) if column names are in this DataFrame

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """
        self.columns_ = as_list(self.columns)
        check_X_for_type(X)
        self._check_column_names(X)
        self.feature_names_ = list(X.drop(columns=self.columns_).columns)
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :returns: ``pd.DataFrame`` with only the selected columns
        """
        check_is_fitted(self, ["feature_names_"])
        check_X_for_type(X)
        if self.columns_:
            return X.drop(columns=self.columns_)
        return X

    def get_feature_names(self):
        return self.feature_names_

    def _check_column_length(self):
        """Check if all columns are dropped"""
        if len(self.feature_names_) == 0:
            raise ValueError(f"Dropping {self.columns_} would result in an empty output DataFrame")

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
        non_existent_columns = set(self.columns_).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")


class PandasTypeSelector(BaseEstimator, TransformerMixin):
    """
    The `PandasTypeSelector` transformer allows to select columns in a pandas DataFrame based on their type.
    Can be useful in a sklearn Pipeline.

    It uses
    [pandas.DataFrame.select_dtypes]
    (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
    method.

    Parameters
    ----------
    include : scalar or list-like
        Column type(s) to be selected
    exclude : scalar or list-like
        Column type(s) to be excluded from selection

    Attributes
    ----------
    feature_names_ : list[str]
        The names of the features to keep during transform.
    X_dtypes_ : pd.Series
        The dtypes of the columns in the input DataFrame.

    !!! warning

        Raises a `TypeError` if input provided is not a DataFrame.
    """

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """Fit the transformer by saving the column names to keep during transform.

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : PandasTypeSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        ValueError
            If provided type(s) results in empty dataframe.
        """
        check_X_for_type(X)
        self.X_dtypes_ = X.dtypes
        self.feature_names_ = X.select_dtypes(include=self.include, exclude=self.exclude).columns.tolist()

        if len(self.feature_names_) == 0:
            raise ValueError("Provided type(s) results in empty dataframe")

        return self

    def get_feature_names(self, *args, **kwargs):
        """Alias for `.feature_names_` attribute"""
        return self.feature_names_

    def transform(self, X):
        """Returns a pandas DataFrame with columns (de)selected based on their dtype.

        Parameters
        ----------
        X : pd.DataFrame
            The data to select dtype for.

        Returns
        -------
        pd.DataFrame
            The data with the specified columns selected.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        ValueError
            If column dtypes were not equal during fit and transform.
        """
        check_is_fitted(self, ["X_dtypes_", "feature_names_"])

        try:
            if (self.X_dtypes_ != X.dtypes).any():
                raise ValueError(
                    f"Column dtypes were not equal during fit and transform. Fit types: \n"
                    f"{self.X_dtypes_}\n"
                    f"transform: \n"
                    f"{X.dtypes}"
                )
        except ValueError as e:
            raise ValueError("Columns were not equal during fit and transform") from e

        check_X_for_type(X)
        transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

        return transformed_df


class ColumnSelector(BaseEstimator, TransformerMixin):
    """The `ColumnSelector` transformer allows selecting specific columns from a pandas DataFrame by name.
    Can be useful in a sklearn Pipeline.

    Parameters
    ----------
    columns : str | list[str]
        Column name(s) to be selected.

    Attributes
    ----------
    columns_ : list[str]
        The names of the features to keep during transform.

    !!! warning

        Raises a `TypeError` if input provided is not a DataFrame.

        Raises a `ValueError` if columns provided are not in the input DataFrame.
    """

    def __init__(self, columns: Union[str, List[str]]):
        # if the columns parameter is not a list, make it into a list
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer by storing the column names to keep during transform.

        Checks:

        1. If input is a `pd.DataFrame` object
        2. If column names are in such DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        KeyError
            If one or more of the columns provided doesn't exist in the input DataFrame.
        ValueError
            If dropping the specified columns would result in an empty output DataFrame.
        """
        self.columns_ = as_list(self.columns)
        check_column_length(self.columns_)
        check_X_for_type(X)
        self._check_column_names(X)
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.

        Returns
        -------
        pd.DataFrame
            The data with the specified columns dropped.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        """
        check_X_for_type(X)
        if self.columns:
            return X[self.columns_]
        return X

    def get_feature_names(self):
        """Alias for `.columns_` attribute"""
        return self.columns_

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
        non_existent_columns = set(self.columns_).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """The `IdentityTransformer` returns what it is fed. Does not apply any transformation.

    The reason for having it is because you can build more expressive pipelines.

    Parameters
    ----------
    check_X : bool, default=False
        Whether to validate `X` to be non-empty 2D array of finite values and attempt to cast `X` to float.
        If disabled, the model/pipeline is expected to handle e.g. missing, non-numeric, or non-finite values.

    Attributes
    ----------
    n_samples_ : int
        The number of samples seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    shape_ : tuple[int, int]
        Deprecated, please use `n_samples_` and `n_features_in_` instead.
    """

    def __init__(self, check_X: bool = False):
        self.check_X = check_X

    def fit(self, X, y=None):
        """Check the input data if `check_X` is enabled and and records its shape.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : IdentityTransformer
            The fitted transformer.
        """
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        self.n_samples_, self.n_features_in_ = X.shape
        return self

    def transform(self, X):
        """Performs identity "transformation" on `X` - which is no transformation at all.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            Unchanged input data.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        if self.check_X:
            X = check_array(X, copy=True, estimator=self)
        check_is_fitted(self, "n_features_in_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Wrong shape is passed to transform. Trained on {self.n_features_in_} cols got {X.shape[1]}"
            )
        return X

    @property
    def shape_(self):
        """Returns the shape of the estimator."""
        return (self.n_samples_, self.n_features_in_)
