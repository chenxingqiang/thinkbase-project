"""Advanced transformation strategies for data processing.

This module provides the data-transformation strategies used by the
:class:`~agentflow.transformations.pipeline.TransformationPipeline`:

* :class:`OutlierRemovalStrategy` -- drop outlier rows (z-score / IQR /
  modified z-score / isolation forest).
* :class:`FeatureEngineeringStrategy` -- polynomial expansion and
  standardisation of numeric columns.
* :class:`TextTransformationStrategy` -- normalisation (optionally
  stop-word removal, lemmatisation and TF-IDF vectorisation) of text
  columns in a :class:`pandas.DataFrame`.
* :class:`TimeSeriesTransformationStrategy` -- trend/seasonality
  decomposition, rolling windows, lags and differencing.
* :class:`AnomalyDetectionStrategy` -- isolation forest / local outlier
  factor / statistical / ensemble anomaly scoring.
* :class:`StringTextTransformationStrategy` -- the config-dict driven,
  string-level text strategy that used to live in the
  ``agentflow.transformations.text`` module (kept for the compatibility
  alias served by :mod:`agentflow.transformations`).

All strategies follow the same contract:

.. code-block:: python

    strategy = OutlierRemovalStrategy(method="z_score", threshold=3.0)
    cleaned = strategy.transform(data)
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

__all__ = [
    "AdvancedTransformationStrategy",
    "OutlierRemovalStrategy",
    "FeatureEngineeringStrategy",
    "TextTransformationStrategy",
    "TimeSeriesTransformationStrategy",
    "AnomalyDetectionStrategy",
    "StringTextTransformationStrategy",
]

# Small built-in stop-word list used when the NLTK corpus is not installed.
_FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how i if in into is isn't it its itself let's me
    more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your
    yours yourself yourselves
    """.split()
)


def _get_stopwords() -> frozenset:
    """Return the English stop-word set, with a built-in fallback.

    The NLTK corpus files are frequently not downloaded in minimal
    environments, so fall back to a small built-in list instead of
    crashing with a ``LookupError``.
    """
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except Exception:  # LookupError, OSError, nltk not fully installed, ...
        return _FALLBACK_STOPWORDS


def _get_lemmatizer():
    """Return a ``WordNetLemmatizer`` or ``None`` when WordNet is missing."""
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("testing")  # probe for the WordNet corpus
        return lemmatizer
    except Exception:
        return None


def _text_columns(data: pd.DataFrame) -> List[Any]:
    """Return the columns of *data* that hold text.

    pandas 3 stores strings in the new ``str`` dtype while pandas 2 uses
    ``object``; select both (guarding for versions that do not know the
    ``str`` selector yet).
    """
    try:
        return list(data.select_dtypes(include=["object", "str", "string"]).columns)
    except TypeError:
        return list(data.select_dtypes(include=["object"]).columns)


class AdvancedTransformationStrategy:
    """Base class for advanced transformation strategies."""

    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize transformation strategy.

        Args:
            logger: Optional logger instance. When omitted a logger named
                after the concrete class is used.
            **kwargs: Additional strategy-specific parameters, kept in
                :attr:`params` for later use by :meth:`transform`.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.params: Dict[str, Any] = kwargs

    def transform(self, data: Any) -> Any:
        """Transform input data.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement transform method")


class OutlierRemovalStrategy(AdvancedTransformationStrategy):
    """Strategy for removing outliers from data."""

    def __init__(self, method: str = "z_score", threshold: float = 3.0, **kwargs):
        """Initialize outlier removal strategy.

        Args:
            method: Method to use for outlier detection. One of
                ``'z_score'``, ``'iqr'``, ``'modified_z_score'`` or
                ``'isolation_forest'``.
            threshold: Threshold for outlier detection (z-score cutoff,
                IQR multiplier or modified z-score cutoff).
            **kwargs: Extra parameters, e.g. ``contamination`` and
                ``random_state`` for the isolation forest method.
        """
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.type = "outlier_removal"
        self.params = {
            "method": method,
            "threshold": threshold,
            **kwargs,
        }

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Remove outliers from data.

        Args:
            data: Input data (DataFrame or ndarray)

        Returns:
            Data with outlier rows removed, in the same container type
            as the input.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        self.logger.info(
            "Applying strategy: %s outlier removal with threshold %s",
            self.method,
            self.threshold,
        )

        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = pd.DataFrame(data)

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return data.values if is_numpy else data

        if self.method == "z_score":
            z_scores = np.abs(StandardScaler().fit_transform(numeric_data))
            mask = (z_scores < self.threshold).all(axis=1)
        elif self.method == "iqr":
            first_quartile = numeric_data.quantile(0.25)
            third_quartile = numeric_data.quantile(0.75)
            interquartile = third_quartile - first_quartile
            lower = first_quartile - self.threshold * interquartile
            upper = third_quartile + self.threshold * interquartile
            mask = ~((numeric_data < lower) | (numeric_data > upper)).any(axis=1)
        elif self.method == "modified_z_score":
            median = numeric_data.median()
            median_absolute_deviation = (numeric_data - median).abs().median()
            # Guard against a zero MAD (constant column).
            mad = median_absolute_deviation.replace(0, np.nan)
            modified_z_scores = 0.6745 * (numeric_data - median) / mad
            mask = (np.abs(modified_z_scores.fillna(0)) < self.threshold).all(axis=1)
        elif self.method == "isolation_forest":
            contamination = self.params.get("contamination", 0.1)
            random_state = self.params.get("random_state", None)
            detector = IsolationForest(contamination=contamination, random_state=random_state)
            predictions = detector.fit_predict(numeric_data)
            mask = predictions == 1  # 1 marks inliers, -1 outliers
        else:
            raise ValueError(f"Unknown outlier removal method: {self.method}")

        result = data[mask]
        return result.values if is_numpy else result


class FeatureEngineeringStrategy(AdvancedTransformationStrategy):
    """Strategy for feature engineering."""

    def __init__(self, strategy: str = "polynomial", **kwargs):
        """Initialize feature engineering strategy.

        Args:
            strategy: Strategy to use, either ``'polynomial'`` or
                ``'standard'``.
            **kwargs: ``degree`` for polynomial expansion, ``with_mean``
                and ``with_std`` for standardisation.
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.degree = kwargs.get("degree", 2)
        self.with_mean = kwargs.get("with_mean", True)
        self.with_std = kwargs.get("with_std", True)
        self.type = "feature_engineering"
        self.params = {
            "strategy": strategy,
            "degree": self.degree,
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            **kwargs,
        }

    def transform(self, data: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Union[pd.DataFrame, np.ndarray]:
        """Apply feature engineering transformations.

        Args:
            data: Input data (DataFrame, ndarray or dictionary)

        Returns:
            Transformed data. Dictionary input is converted to a
            single-row DataFrame; ndarray input produces an ndarray.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        self.logger.info("Applying strategy: %s feature engineering", self.strategy)

        is_numpy = isinstance(data, np.ndarray)
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif is_numpy:
            data = pd.DataFrame(data)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature engineering requires a DataFrame, ndarray or dict input")

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            # Nothing to engineer; return the input unchanged.
            return data.values if is_numpy else data

        if self.strategy == "standard":
            scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            transformed = scaler.fit_transform(numeric_data)
            transformed_df = pd.DataFrame(transformed, columns=numeric_data.columns, index=data.index)

            result = data.copy()
            for col in numeric_data.columns:
                result[col] = transformed_df[col]
            return result.values if is_numpy else result

        elif self.strategy == "polynomial":
            degree = self.params.get("degree", 2)
            poly = PolynomialFeatures(degree=degree)
            transformed = poly.fit_transform(numeric_data)
            feature_names = [f"poly_{i}" for i in range(transformed.shape[1])]
            transformed_df = pd.DataFrame(transformed, columns=feature_names, index=data.index)

            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
            if not non_numeric_columns.empty:
                result = pd.concat([data[non_numeric_columns], transformed_df], axis=1)
            else:
                result = transformed_df
            return result.values if is_numpy else result

        else:
            raise ValueError(f"Unknown feature engineering strategy: {self.strategy}")


class TextTransformationStrategy(AdvancedTransformationStrategy):
    """Strategy for transforming text columns of a DataFrame."""

    def __init__(self, method: str = "normalize", **kwargs):
        """Initialize text transformation strategy.

        Args:
            method: Transformation method; only ``'normalize'`` is
                supported.
            **kwargs: ``remove_stopwords`` (default ``True``),
                ``lemmatize`` (default ``False``), ``vectorize``
                (default ``False``) and ``max_features`` (default ``100``)
                control the normalisation pipeline.
        """
        super().__init__(**kwargs)
        self.method = method
        self.remove_stopwords = kwargs.get("remove_stopwords", True)
        self.lemmatize = kwargs.get("lemmatize", False)
        self.vectorize = kwargs.get("vectorize", False)
        self.max_features = kwargs.get("max_features", 100)
        self.type = "text_transformation"

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Transform the text columns of *data*.

        Args:
            data: Input data (DataFrame or ndarray)

        Returns:
            Data with normalised text columns (plus a ``<col>_word_count``
            feature and, when enabled, TF-IDF columns), in the same
            container type as the input.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        self.logger.info("Applying strategy: %s text transformation", self.method)

        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = pd.DataFrame(data)

        text_cols = _text_columns(data)
        if not text_cols:
            return data.values if is_numpy else data

        if self.method != "normalize":
            raise ValueError(f"Unknown text transformation method: {self.method}")

        result = data.copy()
        for col in text_cols:
            # Basic text cleaning.
            result[col] = result[col].astype("str").str.lower()
            result[col] = result[col].str.replace(r"[^\w\s]", "", regex=True)

            if self.remove_stopwords:
                stop_words = _get_stopwords()
                result[col] = result[col].apply(
                    lambda text: " ".join(
                        word for word in str(text).split() if word not in stop_words
                    )
                )

            if self.lemmatize:
                lemmatizer = _get_lemmatizer()
                if lemmatizer is not None:
                    result[col] = result[col].apply(
                        lambda text: " ".join(
                            lemmatizer.lemmatize(word) for word in str(text).split()
                        )
                    )

            # Add a word-count feature.
            result[f"{col}_word_count"] = result[col].str.split().str.len()

            if self.vectorize:
                vectorizer = TfidfVectorizer(max_features=self.max_features)
                tfidf_matrix = vectorizer.fit_transform(result[col].fillna(""))
                feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=data.index)
                result = pd.concat([result, tfidf_df], axis=1)

        return result.values if is_numpy else result


class TimeSeriesTransformationStrategy(AdvancedTransformationStrategy):
    """Strategy for transforming (univariate) time series data."""

    def __init__(self, strategy: str = "decomposition", **kwargs):
        """Initialize time series transformation strategy.

        Args:
            strategy: Strategy to use: ``'decomposition'``,
                ``'rolling_features'``, ``'lag_features'`` or
                ``'difference'``.
            **kwargs: ``period``, ``window``, ``lags`` or ``order``
                depending on the chosen strategy.
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.type = "time_series_transformation"

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform time series data.

        Args:
            data: Input DataFrame holding exactly one value column
                (optionally indexed by timestamp).

        Returns:
            DataFrame containing the original series (column
                ``'original'``) plus the strategy-specific features.

        Raises:
            ValueError: If the data is not one-dimensional or the
                strategy is unknown.
        """
        self.logger.info("Applying strategy: %s time series transformation", self.strategy)

        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 1:
                raise ValueError("Time series data must be 1-dimensional")
            series = data.iloc[:, 0]
        else:
            series = pd.Series(data)

        result = pd.DataFrame(index=series.index)
        result["original"] = series  # Always include original series

        if self.strategy == "decomposition":
            # Classical additive decomposition, implemented with plain
            # pandas/numpy so it works without statsmodels.
            period = int(self.params.get("period", 7))
            trend = series.rolling(window=period, center=True, min_periods=1).mean()
            detrended = series - trend
            phase = np.arange(len(detrended)) % period
            seasonal = detrended.groupby(phase).transform("mean")
            result["trend"] = trend
            result["seasonal"] = seasonal
            result["residual"] = detrended - seasonal

        elif self.strategy == "rolling_features":
            window = int(self.params.get("window", 14))
            result["rolling_mean"] = series.rolling(window=window).mean()
            result["rolling_std"] = series.rolling(window=window).std()
            result["rolling_min"] = series.rolling(window=window).min()
            result["rolling_max"] = series.rolling(window=window).max()

        elif self.strategy == "lag_features":
            lags = self.params.get("lags", [1, 7, 14])
            for lag in lags:
                result[f"lag_{lag}"] = series.shift(int(lag))

        elif self.strategy == "difference":
            order = int(self.params.get("order", 1))
            result[f"diff_{order}"] = series.diff(order)
            result["diff_abs"] = result[f"diff_{order}"].abs()
            result["diff_sign"] = result[f"diff_{order}"].apply(
                lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
            )

        else:
            raise ValueError(f"Unknown time series transformation strategy: {self.strategy}")

        return result


class AnomalyDetectionStrategy(AdvancedTransformationStrategy):
    """Strategy for detecting anomalies in data."""

    def __init__(self, strategy: str = "isolation_forest", **kwargs):
        """Initialize anomaly detection strategy.

        Args:
            strategy: Strategy to use: ``'isolation_forest'``,
                ``'local_outlier_factor'``, ``'statistical'`` or
                ``'ensemble'``.
            **kwargs: Detector parameters forwarded verbatim (e.g.
                ``contamination``, ``n_neighbors``, ``random_state``).
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.type = "anomaly_detection"

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Detect anomalies in data.

        Args:
            data: Input data (ndarray or DataFrame)

        Returns:
            The input augmented with anomaly information: for DataFrames
            the columns ``'predictions'``, ``'anomalies'`` and
            ``'anomaly_scores'`` are added; for ndarrays three extra
            columns are appended.

        Raises:
            ValueError: If the strategy is unknown.
        """
        self.logger.info("Applying strategy: %s anomaly detection", self.strategy)

        is_dataframe = isinstance(data, pd.DataFrame)
        data_array = data.values if is_dataframe else np.asarray(data)

        if self.strategy == "isolation_forest":
            detector = IsolationForest(**self.params)
            scores = detector.fit_predict(data_array)
            anomaly_scores = detector.score_samples(data_array)

        elif self.strategy == "local_outlier_factor":
            detector = LocalOutlierFactor(**self.params)
            scores = detector.fit_predict(data_array)
            anomaly_scores = detector.negative_outlier_factor_

        elif self.strategy == "statistical":
            mean = np.mean(data_array, axis=0)
            std = np.std(data_array, axis=0)
            std = np.where(std == 0, 1.0, std)  # guard constant columns
            z_scores = np.abs((data_array - mean) / std)
            scores = np.where(np.any(z_scores > 3, axis=1), -1, 1)
            anomaly_scores = -np.max(z_scores, axis=1)

        elif self.strategy == "ensemble":
            iforest = IsolationForest(**self.params)
            lof = LocalOutlierFactor(**self.params)

            iforest_scores = iforest.fit_predict(data_array)
            lof_scores = lof.fit_predict(data_array)

            # Majority voting across detectors.
            scores = np.where((iforest_scores + lof_scores) < 0, -1, 1)

            # Average anomaly scores.
            iforest_anomaly_scores = iforest.score_samples(data_array)
            lof_anomaly_scores = lof.negative_outlier_factor_
            anomaly_scores = (iforest_anomaly_scores + lof_anomaly_scores) / 2

        else:
            raise ValueError(f"Unknown anomaly detection strategy: {self.strategy}")

        if is_dataframe:
            result = data.copy()
            result["predictions"] = scores
            result["anomalies"] = scores == -1  # True for anomalies (-1)
            result["anomaly_scores"] = anomaly_scores
            return result

        return np.column_stack([data_array, scores, scores == -1, anomaly_scores])


class StringTextTransformationStrategy:
    """Config-dict driven, string-level text transformation.

    This is the strategy historically served by the
    ``agentflow.transformations.text`` module. It is configured with a
    plain dictionary and transforms a single string::

        strategy = StringTextTransformationStrategy({"lowercase": True})
        strategy.transform("  HELLO WORLD!  ")  # -> "hello world!"

    Supported config keys:

    * ``lowercase`` -- lower-case the text.
    * ``strip`` -- strip leading/trailing whitespace.
    * ``replacements`` -- mapping of substrings to replace.

    It is exposed as ``agentflow.transformations.text
    .TextTransformationStrategy`` for backward compatibility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with a configuration dictionary."""
        self.config: Dict[str, Any] = dict(config or {})

    def transform(self, text: str) -> str:
        """Transform text data.

        Args:
            text: Text to transform

        Returns:
            Transformed text

        Raises:
            ValueError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        transformed = text

        if self.config.get("lowercase", False):
            transformed = transformed.lower()

        if self.config.get("strip", False):
            transformed = transformed.strip()

        replacements = self.config.get("replacements") or {}
        for old, new in replacements.items():
            transformed = transformed.replace(old, new)

        return transformed
