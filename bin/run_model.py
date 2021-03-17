#!/usr/bin/env bash

import argparse
import sys

from copy import deepcopy

from math import floor

import h5py
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
    ndcg_score
)

from sklearn.preprocessing import RobustScaler, MinMaxScaler

from sklearn.kernel_approximation import Nystroem

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LassoLars, ElasticNet

from sklearn.model_selection import GroupKFold

import xgboost as xgb

from ngboost.distns.normal import Normal
from ngboost.distns.cauchy import Cauchy
from ngboost.distns.exponential import Exponential
from ngboost.distns.lognormal import LogNormal
from ngboost import NGBRegressor
from ngboost.learners import DecisionTreeRegressor

import optuna


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (
    check_is_fitted,
    FLOAT_DTYPES,
    _deprecate_positional_args
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    Lambda,
    Conv1D,
    Flatten,
    MaxPool1D,
    Reshape,
    BatchNormalization,
    LocallyConnected1D,
    Add,
    Concatenate
)

from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


class MarkerMAFScaler(TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a give """

    @_deprecate_positional_args
    def __init__(self, ploidy=2, *, copy=True):
        self.ploidy = ploidy
        self.copy = copy
        return

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'n_samples_seen_'):
            del self.n_samples_seen_
            del self.allele_counts_
            del self.P

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        all_ok = np.isin(X[~np.isnan(X)], np.arange(self.ploidy + 1)).all()

        if not all_ok:
            raise ValueError(
                "Encountered a value less than 0 or greater "
                f"than {self.ploidy}."
            )

        # Maybe raise a warning if no 0s or 2s i.e. maybe smaller ploidy than
        # specified.

        if first_pass:
            self.n_samples_seen_ = (~np.isnan(X)).sum(axis=0)
            self.allele_counts_ = (self.ploidy - X).sum(axis=0)
        else:
            self.n_samples_seen_ += (~np.isnan(X)).sum(axis=0)
            self.allele_counts_ += (self.ploidy - X).sum(axis=0)

        # Frequency of alternate allele
        p_i = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)
        self.P = self.ploidy * (p_i - 0.5)
        return self

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(X, copy=self.copy, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan", reset=False)

        X -= ((self.ploidy / 2) + self.P)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X += (self.ploidy / 2) + self.P
        return X


class DropDuplicates(TransformerMixin, BaseEstimator):

    @_deprecate_positional_args
    def __init__(self, inner, copy=True):
        self.inner = inner
        return

    def _reset(self):
        self.inner._reset()
        return

    def fit(self, X, y=None, individuals=None):
        # Reset internal state before fitting
        if individuals is None:
            raise ValueError("Drop duplicates must be provided with individuals")

        X_trans = pd.DataFrame(X).groupby(individuals).first().values
        if y is None:
            y_trans = None
        else:
            y_trans = pd.Series(y).groupby(individuals).mean().values

        self.inner.fit(X_trans)

        return self

    def transform(self, X, y=None):
        return self.inner.transform(X)


    def fit_transform(self, X, y=None, individuals=None):
        return self.fit(X, y, individuals).transform(X, y)


class VanRadenSimilarity(TransformerMixin, BaseEstimator):

    @_deprecate_positional_args
    def __init__(self, ploidy=2, *, distance=False, scale=True, copy=True):
        self.distance = distance
        self.scale = scale
        self.ploidy = ploidy
        self.copy = copy
        return

    def _reset(self):
        if hasattr(self, 'n_samples_seen_'):
            del self.n_samples_seen_
            del self.allele_counts_
            del self.X
            del self.denom
            del self.P
            del self.max
            del self.min
        return

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        all_ok = np.isin(X[~np.isnan(X)], np.arange(self.ploidy + 1)).all()

        if not all_ok:
            raise ValueError(
                "Encountered a value less than 0 or greater "
                f"than {self.ploidy}."
            )

        # Maybe raise a warning if no 0s or 2s i.e. maybe smaller ploidy than
        # specified.

        if first_pass:
            self.n_samples_seen_ = (~np.isnan(X)).sum(axis=0)
            self.allele_counts_ = (self.ploidy - X).sum(axis=0)
            self.X = X
            self.max = float("-inf")
            self.min = float("inf")
        else:
            self.n_samples_seen_ += (~np.isnan(X)).sum(axis=0)
            self.allele_counts_ += (self.ploidy - X).sum(axis=0)
            self.X = np.concatenate([self.X, X])

        # Frequency of alternate allele
        p_i = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)
        self.denom = (
            (self.ploidy * np.sum(p_i * (1 - p_i))) +
            np.finfo(np.float).eps  # Just to avoid zero division
        )
        self.P = self.ploidy * (p_i - 0.5)

        # This is just so we can get the unscaled max
        scale = self.scale
        self.scale = False
        results = self.transform(X)
        self.scale = scale

        self.max = max([self.max, np.max(np.abs(results))])
        self.min = min([self.min, np.min(np.abs(results))])
        return self

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def transform(self, X, y=None):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(X, copy=self.copy, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan", reset=False)

        Xtrain = self.X - ((self.ploidy / 2) + self.P)
        X -= ((self.ploidy / 2) + self.P)

        dists = X.dot(Xtrain.T) / self.denom

        # Min max scale
        if self.scale:
            diff = self.max - self.min

            if diff == 0:
                diff = 1e-15

            dists -= self.min
            dists /= diff

        if self.distance:
            return -1 * dists
        else:
            return dists


class PercentileRankTransformer(object):
    """Convert a continuous y scale into a quantised rank """

    def __init__(self, percentiles, reverse=False, copy=True):
        self.percentiles = percentiles
        self.reverse = reverse
        self.copy = copy
        return

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        del self.thresholds
        return self

    def fit(self, y, individuals=None):
        if individuals is not None:
            y = pd.Series(y).groupby(individuals).mean().values

        if self.reverse:
            self.thresholds = sorted([
                np.percentile(y, 100 - p)
                for p
                in self.percentiles
            ], reverse=True)
        else:
            self.thresholds = sorted([
                np.percentile(y, p) for p in self.percentiles
            ])
        return self

    def transform(self, y, individuals=None):
        assert hasattr(self, "thresholds")

        if individuals is None:
            individuals = np.arange(len(y))

        df = pd.DataFrame({"y": y, "individual": individuals})
        df_means = df.groupby("individual")["y"].mean().reset_index()
        df_means.rename(columns={"y": "mean"}, inplace=True)

        y_ranks = df_means
        y_ranks["rank"] = 0

        if self.reverse:
            for i, th in enumerate(self.thresholds, 1):
                y_ranks.loc[y_ranks["mean"] < th, "rank"] = i
        else:
            for i, th in enumerate(self.thresholds, 1):
                y_ranks.loc[y_ranks["mean"] > th, "rank"] = i

        y_ranks.set_index("individual", inplace=True)
        y_ranks = y_ranks.loc[individuals]
        y_ranks = y_ranks["rank"].values.astype("int64")
        return y_ranks

    def fit_transform(self, y, individuals=None):
        return self.fit(y, individuals).transform(y, individuals)


def ranking_stats(
    y_train,
    y_train_preds,
    y_train_means,
    y_train_means_preds,
    y_test,
    y_test_preds,
    y_test_means,
    y_test_means_preds
):
    # Used for NDCG@N computations
    y_train_preds_length = y_train_preds.shape[0]
    y_train_means_preds_length = y_train_means_preds.shape[0]
    y_test_preds_length = y_test_preds.shape[0]
    y_test_means_preds_length = y_test_means_preds.shape[0]

    # These are used for NDCG
    y_train_preds = y_train_preds.reshape((1, -1))
    y_train_means_preds = y_train_means_preds.reshape((1, -1))
    y_test_preds = y_test_preds.reshape((1, -1))
    y_test_means_preds = y_test_means_preds.reshape((1, -1))

    # These are used for NDCG
    y_train = y_train.reshape((1, -1))
    y_train_means = y_train_means.reshape((1, -1))
    y_test = y_test.reshape((1, -1))
    y_test_means = y_test_means.reshape((1, -1))

    results = {
        "ndcg_test": ndcg_score(y_test, y_test_preds),
        "ndcg_test_means": ndcg_score(y_test_means, y_test_means_preds),
        "ndcg_train": ndcg_score(y_train, y_train_preds),
        "ndcg_train_means": ndcg_score(y_train_means, y_train_means_preds),
        "ndcgat50_test": ndcg_score(
            y_test,
            y_test_preds,
            k=floor(y_train_preds_length * 0.5)
        ),
        "ndcgat50_test_means": ndcg_score(
            y_test_means,
            y_test_means_preds,
            k=floor(y_train_means_preds_length * 0.5)
        ),
        "ndcgat50_train": ndcg_score(
            y_train,
            y_train_preds,
            k=floor(y_test_preds_length * 0.5)
        ),
        "ndcgat50_train_means": ndcg_score(
            y_train_means,
            y_train_means_preds,
            k=floor(y_test_means_preds_length * 0.5)
        ),
        "ndcgat90_test": ndcg_score(
            y_test,
            y_test_preds,
            k=floor(y_train_preds_length * 0.9)
        ),
        "ndcgat90_test_means": ndcg_score(
            y_test_means,
            y_test_means_preds,
            k=floor(y_train_means_preds_length * 0.9)
        ),
        "ndcgat90_train": ndcg_score(
            y_train,
            y_train_preds,
            k=floor(y_test_preds_length * 0.9)
        ),
        "ndcgat90_train_means": ndcg_score(
            y_train_means,
            y_train_means_preds,
            k=floor(y_test_means_preds_length * 0.9)
        ),
    }
    return results


def regression_stats(
    y_train,
    y_train_preds,
    y_train_means,
    y_train_means_preds,
    y_test,
    y_test_preds,
    y_test_means,
    y_test_means_preds,
    rank_transformer,
    rank_reverse=False,
):
    results = {
        "mae_train": mean_absolute_error(y_train, y_train_preds),
        "mae_train_means": mean_absolute_error(
            y_train_means,
            y_train_means_preds
        ),
        "mae_test": mean_absolute_error(y_test, y_test_preds),
        "mae_test_means": mean_absolute_error(
            y_test_means,
            y_test_means_preds
        ),
        "median_ae_train": median_absolute_error(y_train, y_train_preds),
        "median_ae_train_means": median_absolute_error(
            y_train_means,
            y_train_means_preds
        ),
        "median_ae_test": median_absolute_error(y_test, y_test_preds),
        "median_ae_test_means": median_absolute_error(
            y_test_means,
            y_test_means_preds
        ),
        "mse_train": mean_squared_error(y_train, y_train_preds),
        "mse_train_means": mean_squared_error(
            y_train_means,
            y_train_means_preds
        ),
        "mse_test": mean_squared_error(y_test, y_test_preds),
        "mse_test_means": mean_squared_error(
            y_test_means,
            y_test_means_preds
        ),
        "pearsons_train": np.corrcoef(y_train, y_train_preds)[0][1],
        "pearsons_train_means": np.corrcoef(
            y_train_means,
            y_train_means_preds
        )[0][1],
        "pearsons_test": np.corrcoef(y_test, y_test_preds)[0][1],
        "pearsons_test_means": np.corrcoef(
            y_test_means,
            y_test_means_preds
        )[0][1],
        "explained_variance_train": explained_variance_score(
            y_train,
            y_train_preds
        ),
        "explained_variance_train_means": explained_variance_score(
            y_train_means,
            y_train_means_preds
        ),
        "explained_variance_test": explained_variance_score(
            y_test,
            y_test_preds
        ),
        "explained_variance_test_means": explained_variance_score(
            y_test_means,
            y_test_means_preds
        ),
        "r2_train": r2_score(y_train, y_train_preds),
        "r2_train_means": r2_score(y_train_means, y_train_means_preds),
        "r2_test": r2_score(y_test, y_test_preds),
        "r2_test_means": r2_score(y_test_means, y_test_means_preds),
    }

    # Just multiply by negative 1 to make small numbers higher in sorted list.
    rank_coef = -1 if rank_reverse else 1
    results.update(ranking_stats(
        rank_transformer.transform(y_train),
        y_train_preds * rank_coef,
        rank_transformer.transform(y_train_means),
        y_train_means_preds * rank_coef,
        rank_transformer.transform(y_test),
        y_test_preds * rank_coef,
        rank_transformer.transform(y_test_means),
        y_test_means_preds * rank_coef,
    ))
    return results


def eval_ranking_stats(
    y,
    preds,
    y_means,
    means_preds
):
    # Used for NDCG@N computations
    preds_length = preds.shape[0]
    means_preds_length = means_preds.shape[0]

    # These are used for NDCG
    preds = preds.reshape((1, -1))
    means_preds = means_preds.reshape((1, -1))

    # These are used for NDCG
    y = y.reshape((1, -1))
    y_means = y_means.reshape((1, -1))

    results = {
        "ndcg": ndcg_score(y, preds),
        "ndcg_means": ndcg_score(y_means, means_preds),
        "ndcgat50": ndcg_score(
            y,
            preds,
            k=floor(preds_length * 0.5)
        ),
        "ndcgat50_means": ndcg_score(
            y_means,
            means_preds,
            k=floor(means_preds_length * 0.5)
        ),
        "ndcgat90": ndcg_score(
            y,
            preds,
            k=floor(preds_length * 0.9)
        ),
        "ndcgat90_means": ndcg_score(
            y_means,
            means_preds,
            k=floor(means_preds_length * 0.9)
        )
    }
    return results


def eval_regression_stats(
    y,
    preds,
    indivs,
    y_means,
    means_preds,
    rank_transformer,
    rank_reverse=False,
):
    results = {
        "mae": mean_absolute_error(y, preds),
        "mae_means": mean_absolute_error(
            y_means,
            means_preds
        ),
        "median_ae": median_absolute_error(y, preds),
        "median_ae_means": median_absolute_error(
            y_means,
            means_preds
        ),
        "mse": mean_squared_error(y, preds),
        "mse_means": mean_squared_error(
            y_means,
            means_preds
        ),
        "pearsons": np.corrcoef(y, preds)[0][1],
        "pearsons_means": np.corrcoef(
            y_means,
            means_preds
        )[0][1],
        "explained_variance": explained_variance_score(
            y,
            preds
        ),
        "explained_variance_means": explained_variance_score(
            y_means,
            means_preds
        ),
        "r2": r2_score(y, preds),
        "r2_means": r2_score(y_means, means_preds),
    }

    # Just multiply by negative 1 to make small numbers higher in sorted list.
    rank_coef = -1 if rank_reverse else 1
    results.update(eval_ranking_stats(
        rank_transformer.transform(y, indivs),
        preds * rank_coef,
        rank_transformer.transform(y_means),
        means_preds * rank_coef,
    ))
    return results


def eval_generations(
    df,
    model,
    best_model,
    response,
    rank_trans,
    rank_reverse
):
    X = df.drop(columns=["individual", response])
    y = df[response]
    indivs = df["individual"]
    X_means = (
        df
        .drop(columns=["individual", response])
        .groupby(df["individual"])
        .mean()
    )

    y_means = (
        df
        [response]
        .groupby(df["individual"])
        .mean()
        .loc[X_means.index.values, ]
    )

    preds = model.predict(best_model, X)
    means_preds = model.predict(best_model, X_means)
    stats = eval_regression_stats(
        y.values,
        preds,
        indivs.values,
        y_means.values,
        means_preds,
        rank_trans,
        rank_reverse
    )

    preds = pd.DataFrame({
        "y": y,
        "preds": preds,
        "individual": df["individual"]
    })

    return stats, preds


def prep_h5_training_dataset(path, response, chip=0):
    bp = h5py.File(path, "r")

    phenotypes = pd.DataFrame(bp["phenotypes"][:])
    phenotypes = phenotypes[phenotypes["generation"] == b"training"]
    phenotypes.drop(columns=["generation", "rep"], inplace=True)
    phenotypes = phenotypes[["individual", response]]

    genotypes = None
    for i, chipname in enumerate(bp["snp_chips"]):
        if chip == i:
            gen_mask = bp["generations"][:] == b"training"
            genotypes = pd.DataFrame(bp["snp_chips"][chipname][:, gen_mask]).T
            break

    assert genotypes is not None

    genotypes["individual"] = bp["sample_names"][gen_mask]
    genotypes = genotypes[
        ["individual"] + list(genotypes.columns[:-2])]

    merged = pd.merge(phenotypes, genotypes, on="individual")
    bp.close()

    return merged


def get_h5_test_dataset(path, response, chip=0):
    bp = h5py.File(path, "r")
    
    try:
        generations = np.unique(bp["generations"][:])

        for generation in generations:
            phenotypes = pd.DataFrame(bp["phenotypes"][:])
            phenotypes = phenotypes[phenotypes["generation"] == generation]
            phenotypes.drop(columns=["generation", "rep"], inplace=True)
            phenotypes = phenotypes[["individual", response]]

            genotypes = None
            for i, chipname in enumerate(bp["snp_chips"]):
                if i == chip:
                    gen_mask = bp["generations"][:] == generation
                    genotypes = pd.DataFrame(bp["snp_chips"][chipname][:, gen_mask]).T
                    break

            assert genotypes is not None

            genotypes["individual"] = bp["sample_names"][gen_mask]
            genotypes = genotypes[
                ["individual"] + list(genotypes.columns[:-2])]

            merged = pd.merge(phenotypes, genotypes, on="individual")
            yield generation, merged

    finally:
        bp.close()

    return


def variance_weights(X, y, samples):
    df = pd.DataFrame({"y": y, "samples": samples})
    x = df.groupby("samples")["y"].var()[samples]
    counts = df.groupby("samples")["y"].count()[samples]
    y = 1 / (x * counts)
    return y.values


def distance_weights(X, y, samples=None):
    from scipy.spatial.distance import pdist, squareform

    x = pd.DataFrame({
        "index": np.arange(X.shape[0]),
        "genotypes": np.apply_along_axis(
            lambda z: "".join(str(z_i) for z_i in z), 1, X)
    })
    firsts = pd.DataFrame(X).groupby(x["genotypes"]).first()
    groups = (
        x
        .groupby("genotypes")["index"]
        .unique()
        .apply(pd.Series)
        .unstack()
        .reset_index(level=0, drop=True)
        .reset_index()
        .rename(columns={0: "index"})
    )

    dist = squareform(pdist(firsts.values, "cityblock"))
    np.fill_diagonal(dist, 0)

    corr = pd.DataFrame(dist, index=firsts.index.values)
    corr = pd.merge(
        groups,
        corr,
        left_on="genotypes",
        right_index=True
    ).drop(
        columns="genotypes"
    ).set_index("index", drop=True)
    corr = corr.loc[np.arange(X.shape[0])]
    return corr.sum(axis=1).values


def cluster_weights(X, y, samples=None):
    from fastcluster import average
    from scipy.cluster.hierarchy import cut_tree
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist, squareform

    x = pd.DataFrame({
        "index": np.arange(X.shape[0]),
        "genotypes": np.apply_along_axis(
            lambda z: "".join(str(z_i) for z_i in z), 1, X)
    })
    firsts = pd.DataFrame(X).groupby(x["genotypes"]).first()
    groups = (
        x
        .groupby("genotypes")["index"]
        .unique()
        .apply(pd.Series)
        .unstack()
        .reset_index(level=0, drop=True)
        .reset_index()
        .rename(columns={0: "index"})
    )

    dist = pdist(firsts.values, "cityblock")
    hier = average(dist)
    coph = squareform(cophenet(hier))

    height = np.percentile(coph[coph > 0], 0.5)
    clusters = pd.DataFrame({
        "genotypes": firsts.index.values,
        "clusters": cut_tree(hier, height=height)[:, 0]
    })
    clusters = (
        pd.merge(groups, clusters, left_on="genotypes", right_on="genotypes")
        .drop(columns="genotypes")
    )

    cluster_counts = (
        clusters.groupby("clusters").count()["index"]
        .apply(lambda x: (clusters.shape[0] - x) / x)
        .reset_index()
        .rename(columns={"index": "weight"})
    )

    clusters = pd.merge(
        clusters,
        cluster_counts,
        on="clusters"
    ).set_index("index")
    return clusters.loc[np.arange(X.shape[0]), "weight"].values


def sample_weight_fns(weight):
    if weight is None:
        return None
    elif weight == "variance":
        return variance_weights
    elif weight == "distance":
        return distance_weights
    elif weight == "cluster":
        return cluster_weights


def preprocessing_trial(
    trial,
    options=["maf", "maf_dist", "dist",
             "dist_rbf", "dist_laplacian", "dist_poly"]
):
    params = {}
    preprocessor = trial.suggest_categorical("preprocessor", options)
    params["preprocessor"] = preprocessor

    if preprocessor == "dist_rbf":
        params["dist_rbf_gamma"] = trial.suggest_float(
            "dist_rbf_gamma", 1e-15, 0.5)
    elif preprocessor == "dist_laplacian":
        params["dist_laplacian_gamma"] = trial.suggest_float(
            "dist_laplacian_gamma", 1e-15, 0.5)
    elif preprocessor == "dist_poly":
        params["dist_poly_gamma"] = trial.suggest_float(
            "dist_poly_gamma", 0.1, 20)
    elif preprocessor == "maf_rbf":
        params["maf_rbf_gamma"] = trial.suggest_float(
            "maf_rbf_gamma", 1e-15, 0.5)
    elif preprocessor == "maf_laplacian":
        params["maf_laplacian_gamma"] = trial.suggest_float(
            "maf_laplacian_gamma", 1e-15, 0.5)
    elif preprocessor == "maf_poly":
        params["maf_poly_gamma"] = trial.suggest_float(
            "maf_poly_gamma", 0.1, 20)
    return params


def preprocessing_fns(params):

    preprocessor = params["preprocessor"] 

    if preprocessor == "maf":
        return MarkerMAFScaler()

    elif preprocessor == "maf_dist":
        return FeatureUnion([
            ("maf", MarkerMAFScaler()),
            ("dist", VanRadenSimilarity(scale=True))
        ])
    elif preprocessor == "dist":
        return VanRadenSimilarity()
    elif preprocessor == "dist_rbf":
        return FeatureUnion([
            ("dist", VanRadenSimilarity(scale=True)),
            ("rbf", Nystroem(
                kernel="rbf",
                gamma=params["dist_rbf_gamma"],
                n_components=100))
        ])
    elif preprocessor == "dist_laplacian":
        return FeatureUnion([
            ("dist", VanRadenSimilarity(scale=True)),
            ("laplacian", Nystroem(
                kernel="laplacian",
                gamma=params["dist_laplacian_gamma"],
                n_components=100))
        ])
    elif preprocessor == "dist_poly":
        return FeatureUnion([
            ("dist", VanRadenSimilarity(scale=True)),
            ("poly", Nystroem(
                kernel="poly",
                gamma=params["dist_poly_gamma"],
                n_components=100))
        ])

    elif preprocessor == "maf_rbf":
        return FeatureUnion([
            ("maf", MarkerMAFScaler()),
            ("rbf", Nystroem(
                kernel="rbf",
                gamma=params["maf_rbf_gamma"],
                n_components=100))
        ])

    elif preprocessor == "maf_laplacian":
        return FeatureUnion([
            ("maf", MarkerMAFScaler()),
            ("laplacian", Nystroem(
                kernel="laplacian",
                gamma=params["maf_laplacian_gamma"],
                n_components=100))
        ])
    elif preprocessor == "maf_poly":
        return FeatureUnion([
            ("maf", MarkerMAFScaler()),
            ("poly", Nystroem(
                kernel="poly",
                gamma=params["maf_poly_gamma"],
                n_components=100))
        ])

class TestModel(object):

    use_weights = False

    def __init__(
        self,
        data,
        response,
        indiv,
        stat,
        rank_reverse,
        seed,
        model_name
    ):
        self.data = data
        self.response = response
        self.indiv = indiv
        self.stat = stat
        self.rank_reverse = rank_reverse
        self.seed = seed
        self.model_name = model_name
        return

    def __call__(self, trial):
        params = self.sample_params(trial)
        params["train_means"] = trial.suggest_categorical(
            "train_means",
            [True, False]
        )

        if self.use_weights:
            params["weight"] = trial.suggest_categorical(
                "weight",
                ["none", "variance", "distance", "cluster"]
            )
        else:
            params["weight"] = "none"

        try:
            stats = self.cv_eval(params)
        except ValueError as e:
            raise e
            # Sometimes calculating mae etc doesn't work.
            return np.nan

        mae = stats["mae_test_means"].mean()
        pearsons = stats["pearsons_test_means"].mean()

        if np.isnan(pearsons):
            pearsons = -1

        ndcg = stats["ndcg_test_means"].mean()
        ndcgat50 = stats["ndcgat50_test_means"].mean()
        ndcgat90 = stats["ndcgat90_test_means"].mean()

        trial.set_user_attr("mae", mae)
        trial.set_user_attr("pearsons", pearsons)
        trial.set_user_attr("ndcg", ndcg)
        trial.set_user_attr("ndcgat90", ndcgat90)
        trial.set_user_attr("ndcgat50", ndcgat50)

        return stats[self.stat].mean()

    def best_model(self, study, **kwargs):
        params = study.best_params

        if params["train_means"]:
            X = (
                self.data
                .drop(columns=[self.indiv, self.response])
                .groupby(self.data[self.indiv])
                .mean()
            )
            y = (
                self.data[self.response]
                .groupby(self.data[self.indiv])
                .mean()
                .loc[X.index.values, ]
            )
            indiv = np.arange(len(y))
        else:
            X = self.data.drop(columns=[self.indiv, self.response])
            y = self.data[self.response]
            indiv = self.data[self.indiv].values

        params["nsamples"] = len(np.unique(indiv))

        if "weight" in params:
            weight_fn = sample_weight_fns(params["weight"])
        else:
            weight_fn = None

        if weight_fn is None:
            weights = None
        else:
            # Need to do this because variance needed.
            weights = weight_fn(
                self.data.drop(columns=[self.indiv, self.response]).values,
                self.data[self.response].values,
                self.data[self.indiv].values
            )

            if params["train_means"]:
                weights = (
                    pd.Series(weights)
                    .groupby(self.data[self.indiv].values)
                    .mean()
                    .values
                )


        if isinstance(weights, pd.Series):
            weights = weights.values

        model = self.fit(params, X.values, y.values, weights, indiv, **kwargs)
        return model

    def sample_params(self, trial):
        raise NotImplementedError()

    def model(self, params):
        raise NotImplementedError()

    def cv(self, k=5):
        if self.seed is not None:
            np.random.seed(self.seed)

        for i, (train_idx, test_idx) in enumerate(GroupKFold(5).split(
            self.data.drop(columns=[self.indiv, self.response]).values,
            self.data[self.response].values,
            self.data[self.indiv].values,
        )):
            train = self.data.iloc[train_idx]
            X_train = train.drop(columns=[self.indiv, self.response])
            y_train = train[self.response]
            indiv_train = train[self.indiv]

            X_train_means = (
                train
                .drop(columns=[self.indiv, self.response])
                .groupby(train[self.indiv])
                .mean()
            )
            y_train_means = (
                train[self.response]
                .groupby(train[self.indiv])
                .mean()
                .loc[X_train_means.index.values, ]
            )

            test = self.data.iloc[test_idx]
            X_test = test.drop(columns=[self.indiv, self.response])
            y_test = test[self.response]

            X_test_means = (
                test
                .drop(columns=[self.indiv, self.response])
                .groupby(test[self.indiv])
                .mean()
            )

            y_test_means = (
                test[self.response]
                .groupby(test[self.indiv])
                .mean()
                .loc[X_test_means.index.values, ]
            )

            yield (
                i,
                X_train.values,
                y_train.values,
                indiv_train.values,
                X_train_means.values,
                y_train_means.values,
                X_test.values,
                y_test.values,
                X_test_means.values,
                y_test_means.values
            )
        return

    @classmethod
    def fit(cls, model, X, y, weights, individuals, **kwargs):
        raise NotImplementedError()

    @classmethod
    def predict(cls, model, X):
        raise NotImplementedError()

    def cv_eval(self, params, **kwargs):
        out = []
        weight_fn = sample_weight_fns(params["weight"])
        base_params = deepcopy(params)
        
        for (
            i,
            X_train,
            y_train,
            indiv_train,
            X_train_means,
            y_train_means,
            X_test,
            y_test,
            X_test_means,
            y_test_means
        ) in self.cv():

            if weight_fn is None:
                weights = None
            else:
                weights = weight_fn(X_train, y_train, indiv_train)            
                if params["train_means"]:
                    weights = (
                        pd.Series(weights)
                        .groupby(indiv_train)
                        .mean()
                        .values
                    )

            if params["train_means"]:
                X = X_train_means
                y = y_train_means
            else:
                X = X_train
                y = y_train

            params["nsamples"] = len(np.unique(indiv_train))

            if params["train_means"]:
                indiv_fit = np.arange(len(y))
            else:
                indiv_fit = indiv_train
            model = self.fit(params, X, y, weights, indiv_fit, **kwargs)

            train_preds = self.predict(model, X_train)
            train_means_preds = self.predict(model, X_train_means)
            test_preds = self.predict(model, X_test)
            test_means_preds = self.predict(model, X_test_means)
                
            rank_trans = PercentileRankTransformer(
                [50, 90, 99],
                self.rank_reverse
            )
            
            rank_trans.fit(y_train, indiv_train)

            these_stats = regression_stats(
                y_train,
                train_preds,
                y_train_means,
                train_means_preds,
                y_test,
                test_preds,
                y_test_means,
                test_means_preds,
                rank_trans,
                self.rank_reverse
            )

            these_stats.update({"cv": i, "name": self.model_name})
            out.append(these_stats)

        return pd.DataFrame(out)


class SKModel(TestModel):

    def predict(self, model, X):
        preprocessor, model = model
        X_trans = preprocessor.transform(X)
        return model.predict(X_trans)

    def fit(self, params, X, y, weights, individuals, **kwargs):
        """ Default is suitable for sklearn compatible models. """

        preprocessor, model = self.model(params)
        X_trans = preprocessor.fit_transform(X, individuals=individuals)

        if weights is None:
            model.fit(X_trans, y, **kwargs)
        else:
            model.fit(
                X_trans,
                y,
                sample_weight=weights,
                **kwargs
            )

        return preprocessor, model


class XGBBaseModel(TestModel):

    def predict(self, model, X):
        dtrain = xgb.DMatrix(X)
        return model.predict(dtrain)

    def fit(self, params, X, y, weights, individuals, **kwargs):
        """ Default is suitable for sklearn compatible models. """
        from copy import copy
        params = copy(params)

        if weights is None:
            dtrain = xgb.DMatrix(X, label=y)
        else:
            dtrain = xgb.DMatrix(
                X,
                label=y,
                weight=weights
            )

        
        num_boost_round = params.pop("num_boost_round")

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=True
        )

        return model


class XGBModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(trial, options=["maf", "maf_dist"])
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "gamma": trial.suggest_float("gamma", 0, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        })
        return params

    def model(self, params):
        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
                xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=params["n_estimators"],
                booster=params["booster"],
                gamma=params["gamma"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                colsample_bylevel=params["colsample_bylevel"],
                colsample_bynode=params["colsample_bynode"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=self.seed,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_jobs=1,
                verbosity=0,
            )
        )


class KNNModel(SKModel):

    use_weights = False

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["dist", "dist_rbf",
                     "dist_laplacian", "dist_poly"]
        )
        params.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 100),
            "weights": trial.suggest_categorical("weights", ["distance", "uniform"]),
            "leaf_size": trial.suggest_int("leaf_size", 10, 80),
            "algorithm": trial.suggest_categorical("algorithm", ["kd_tree", "ball_tree"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)

        return (
            DropDuplicates(preprocessor),
            KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                leaf_size=params["leaf_size"],
                algorithm=params["algorithm"],
                p=params["p"],
                n_jobs=1
            )
        )


class RFModel(SKModel):
    
    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(trial, options=["maf", "maf_dist"])
        params.update({
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_features": trial.suggest_int("max_features", 10, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0, 1),
        })
    
        if params["bootstrap"]:
            params["oob_score"] = trial.suggest_categorical("oob_score", [True, False])
        else:
            params["oob_score"] = False

        return params

    def model(self, params):
        preprocessor = preprocessing_fns(params)

        # Prevents key error when selecting from "best params"
        if "oob_score" in params:
            oob_score = params["oob_score"]
        else:
            oob_score = False

        return (
            DropDuplicates(preprocessor),
            RandomForestRegressor(
                criterion="mae",
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                max_features=params["max_features"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_impurity_decrease=params["min_impurity_decrease"],
                bootstrap=params["bootstrap"],
                oob_score=oob_score,
                n_jobs=1,
                random_state=self.seed
            )
        )


class ExtraTreesModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(trial, options=["maf", "maf_dist"])
        params.update({
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "max_features": trial.suggest_int("max_features", 10, 100),
            "max_samples": trial.suggest_int("max_samples", 1, 127),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0, 1),
        })
    
        if params["bootstrap"]:
            params["oob_score"] = trial.suggest_categorical("oob_score", [True, False])
        else:
            params["oob_score"] = False

        return params

    def model(self, params):
        preprocessor = preprocessing_fns(params)
        
        if "oob_score" in params:
            oob_score = params["oob_score"]
        else:
            oob_score = False

        return (
            DropDuplicates(preprocessor),
            ExtraTreesRegressor(
                criterion="mae",
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                max_features=params["max_features"],
                max_samples=params["max_samples"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_impurity_decrease=params["min_impurity_decrease"],
                bootstrap=params["bootstrap"],
                oob_score=oob_score,
                n_jobs=1,
                random_state=self.seed
            )
        )


class NGBModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(trial, options=["maf", "maf_dist"])

        params.update({
            "Dist": trial.suggest_categorical("Dist", ["normal", "exponential"]),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_features": trial.suggest_int("max_features", 10, 200),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0, 1),
            "col_sample": trial.suggest_float("col_sample", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            "natural_gradient": trial.suggest_categorical("natural_gradient", [True, False]),
        })
    
        return params


    def model(self, params):
        preprocessor = preprocessing_fns(params)

        dist = {
            "normal": Normal,
            "cauchy": Cauchy,
            "exponential": Exponential,
            "lognormal": LogNormal,
        }[params["Dist"]]

        return (
            DropDuplicates(preprocessor),
            NGBRegressor(
                Dist=dist,
                n_estimators=params["n_estimators"],
                Base=DecisionTreeRegressor(
                    criterion='friedman_mse',
                    max_depth=params["max_depth"],
                    max_features=params["max_features"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_impurity_decrease=params["min_impurity_decrease"]
                ),
                verbose=False,
                col_sample=params["col_sample"],
                learning_rate=params["learning_rate"],
                natural_gradient=params["natural_gradient"],
                random_state=self.seed
            )
        )


class SVRModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["dist", "dist_rbf", "dist_laplacian", "dist_poly"]
        )

        params.update({
            "loss": trial.suggest_categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            "epsilon": trial.suggest_float("epsilon", 0, 500),
            "C": trial.suggest_float("C", 1e-10, 10),
            "intercept_scaling": trial.suggest_float("intercept_scaling", 1e-10, 5),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
            LinearSVR(
                random_state=self.seed,
                fit_intercept=True,
                max_iter=100000,
            )
        )
    
    
class ElasticNetDistModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["dist", "dist_rbf", "dist_laplacian", "dist_poly"]
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
            ElasticNet(
                random_state=self.seed,
                fit_intercept=True,
                max_iter=100000,
                selection="random",
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
            )
        )


class LassoLarsDistModel(SKModel):

    use_weights = False

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["dist", "dist_rbf", "dist_laplacian", "dist_poly"]
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
            LassoLars(
                alpha=params["alpha"],
                fit_intercept=True,
                max_iter=100000,
                random_state=self.seed,
            )
        )


class LassoLarsModel(SKModel):

    use_weights = False

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["maf", "maf_dist", "maf_rbf", "maf_laplacian", "maf_poly"]
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
            LassoLars(
                alpha=params["alpha"],
                fit_intercept=True,
                max_iter=500000,
                random_state=self.seed,
            )
        )


class ElasticNetModel(SKModel):

    use_weights = True

    def sample_params(self, trial):
        params = preprocessing_trial(
            trial,
            options=["maf", "maf_dist", "maf_rbf", "maf_laplacian", "maf_poly"]
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        })
        return params

    def model(self, params):

        preprocessor = preprocessing_fns(params)
        return (
            DropDuplicates(preprocessor),
            ElasticNet(
                random_state=self.seed,
                fit_intercept=True,
                max_iter=500000,
                selection="random",
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
            )
        )


class TFModel(TestModel):

    use_weights = True

    def sample_params(self, trial):

        params = preprocessing_trial(
            trial,
            options=["maf", "maf_dist"]
        )

        params["nhidden"] = trial.suggest_int("nhidden", 0, 4)

        if params["nhidden"] > 0:
            params["layer_size"] = trial.suggest_int("layer_size", 5, 50)

        for i in range(params["nhidden"]):
            i += 1
            params[f"l{i}_l1"] = trial.suggest_float(f"l{i}_l1", 0, 200)
            params[f"l{i}_l2"] = trial.suggest_float(f"l{i}_l2", 0, 200)
            params[f"l{i}_dropout"] = trial.suggest_float(f"l{i}_dropout", 0, 1)
            params[f"l{i}_batchnorm"] = trial.suggest_categorical(f"l{i}_batchnorm", [True, False])

        params["output_l2"] = trial.suggest_float("output_l2", 0, 200)
        params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-2, 0.1)
        params["epochs"] = trial.suggest_int("epochs",  5, 50)
        return params

    def model(self, params):
        preprocessor = DropDuplicates(preprocessing_fns(params))
        model = Sequential()
        # Subtract 2 because of response and invididual column
        
        if "dist" in params["preprocessor"]:
            in_shape = len(self.data.columns) - 2 + params["nsamples"]
        else:
            in_shape = len(self.data.columns) - 2

        model.add(Input(shape=(in_shape, ), name="input")) 

        for i in range(params["nhidden"]):
            i += 1
            model.add(Dense(
                params["layer_size"],
                name=f"l{i}",
                activation="relu",
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(
                    params[f"l{i}_l1"],
                    params[f"l{i}_l2"]
                )
            ))

            if params[f"l{i}_batchnorm"]:
                model.add(BatchNormalization())

            model.add(Dropout(params[f"l{i}_dropout"]))

        model.add(Dense(
            1,
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"output_l2"],
            )
        ))

        model.compile(
            optimizer=Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=["mae"]
        )
        return preprocessor, model

    def predict(self, model, X):

        preprocessor, model = model

        X = preprocessor.transform(X)
        y = model(X).numpy().reshape(-1)
        return y

    def fit(self, params, X, y, weights, individuals, **kwargs):
        # In lieu of a good way to reinitialise models, we do this.
        tf.keras.backend.clear_session()

        preprocessor, model = self.model(params)
        X_trans = preprocessor.fit_transform(X, individuals=individuals)

        if weights is not None:
            weights = weights.reshape(-1, 1)

        model.fit(
            X_trans,
            y.reshape(-1, 1),
            epochs=params["epochs"],
            verbose=0,
            batch_size=100,
            shuffle=True,
            sample_weight=weights,
            **kwargs
        )
        return preprocessor, model 


class TFSumModel(TestModel):

    use_weights = True

    def sample_params(self, trial):

        params = dict()
        params["nhidden"] = trial.suggest_int("nhidden", 0, 4)

        if params["nhidden"] > 0:
            params["layer_size"] = trial.suggest_int("layer_size", 5, 50)

        for i in range(params["nhidden"]):
            i += 1
            params[f"marker_l{i}_l1"] = trial.suggest_float(f"marker_l{i}_l1", 0, 200)
            params[f"marker_l{i}_l2"] = trial.suggest_float(f"marker_l{i}_l2", 0, 200)
            params[f"marker_l{i}_dropout"] = trial.suggest_float(f"marker_l{i}_dropout", 0, 1)
            params[f"marker_l{i}_batchnorm"] = trial.suggest_categorical(f"marker_l{i}_batchnorm", [True, False])

        params["join_size"] = trial.suggest_int("join_size", 2, 10)
        params["join_marker_l1"] = trial.suggest_float("join_marker_l1", 0, 200)
        params["join_marker_l2"] = trial.suggest_float("join_marker_l2", 0, 200)
        params["join_marker_dropout"] = trial.suggest_float("join_marker_dropout", 0, 1)
        params["join_dist_l2"] = trial.suggest_float("join_dist_l2", 0, 200)
        params["join_dist_dropout"] = trial.suggest_float("join_dist_dropout", 0, 1)
        
        params["output_l2"] = trial.suggest_float("output_l2", 0, 200)
        params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-2, 0.1)
        params["epochs"] = trial.suggest_int("epochs",  5, 50)
        return params

    def model(self, params):

        maf_pre = DropDuplicates(MarkerMAFScaler())
        dist_pre = DropDuplicates(VanRadenSimilarity(scale=True))
 
        # Subtract 2 because of response and invididual column
        markers_input = Input(shape=(len(self.data.columns) - 2, ), name="marker_input")
        markers = markers_input
        for i in range(params["nhidden"]):
            i += 1
            markers = Dense(
                params["layer_size"],
                name=f"marker_l{i}",
                activation="relu",
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(
                    params[f"marker_l{i}_l1"],
                    params[f"marker_l{i}_l2"]
                )
            )(markers)

            if params[f"marker_l{i}_batchnorm"]:
                markers = BatchNormalization()(markers)

            markers = Dropout(params[f"marker_l{i}_dropout"])(markers)

        markers = Dense(
            params["join_size"],
            name="marker_join",
            activation="relu",
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(
                params[f"join_marker_l1"],
                params[f"join_marker_l2"]
            )
        )(markers)

        markers_model = Model(inputs=markers_input, outputs=markers)

        dists_input = Input(shape=(params["nsamples"], ), name="dist_input")
        dists = Dense(
            params["join_size"],
            name="dist_join",
            activation="relu",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"join_dist_l2"],
            )
        )(dists_input)
        dists_model = Model(inputs=dists_input, outputs=dists)

        join = Add()([markers, dists])

        y = Dense(
            1,
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"output_l2"],
            )
        )(join)

        model = Model(inputs=[markers_model.input, dists_model.input], outputs=y)

        model.compile(
            optimizer=Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=["mae"]
        )
        return maf_pre, dist_pre, model

    def predict(self, model, X):

        maf_pre, dist_pre, model = model

        markers = maf_pre.transform(X)
        dists = dist_pre.transform(X)
        y = model([markers, dists]).numpy().reshape(-1)
        return y

    def fit(self, params, X, y, weights, individuals, **kwargs):
        # In lieu of a good way to reinitialise models, we do this.
        tf.keras.backend.clear_session()

        maf_pre, dist_pre, model = self.model(params)
        maf_trans = maf_pre.fit_transform(X, individuals=individuals)
        dist_trans = dist_pre.fit_transform(X, individuals=individuals)

        if weights is not None:
            weights = weights.reshape(-1, 1)

        model.fit(
            [maf_trans, dist_trans],
            y.reshape(-1, 1),
            epochs=params["epochs"],
            verbose=0,
            batch_size=100,
            shuffle=True,
            sample_weight=weights,
            **kwargs
        )
        return maf_pre, dist_pre, model 


class TFConcatModel(TestModel):

    use_weights = True

    def sample_params(self, trial):

        params = dict()
        params["nhidden"] = trial.suggest_int("nhidden", 0, 4)

        if params["nhidden"] > 0:
            params["layer_size"] = trial.suggest_int("layer_size", 5, 50)

        for i in range(params["nhidden"]):
            i += 1
            params[f"marker_l{i}_l1"] = trial.suggest_float(f"marker_l{i}_l1", 0, 200)
            params[f"marker_l{i}_l2"] = trial.suggest_float(f"marker_l{i}_l2", 0, 200)
            params[f"marker_l{i}_dropout"] = trial.suggest_float(f"marker_l{i}_dropout", 0, 1)
            params[f"marker_l{i}_batchnorm"] = trial.suggest_categorical(f"marker_l{i}_batchnorm", [True, False])

        params["join_size"] = trial.suggest_int("join_size", 2, 10)
        params["join_marker_l1"] = trial.suggest_float("join_marker_l1", 0, 200)
        params["join_marker_l2"] = trial.suggest_float("join_marker_l2", 0, 200)
        params["join_marker_dropout"] = trial.suggest_float("join_marker_dropout", 0, 1)
        params["join_dist_l2"] = trial.suggest_float("join_dist_l2", 0, 200)
        params["join_dist_dropout"] = trial.suggest_float("join_dist_dropout", 0, 1)
        
        params["output_l2"] = trial.suggest_float("output_l2", 0, 200)
        params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-2, 0.1)
        params["epochs"] = trial.suggest_int("epochs",  5, 50)
        return params

    def model(self, params):

        maf_pre = DropDuplicates(MarkerMAFScaler())
        dist_pre = DropDuplicates(VanRadenSimilarity(scale=True))
 
        # Subtract 2 because of response and invididual column
        markers_input = Input(shape=(len(self.data.columns) - 2, ), name="marker_input")
        markers = markers_input
        for i in range(params["nhidden"]):
            i += 1
            markers = Dense(
                params["layer_size"],
                name=f"marker_l{i}",
                activation="relu",
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(
                    params[f"marker_l{i}_l1"],
                    params[f"marker_l{i}_l2"]
                )
            )(markers)

            if params[f"marker_l{i}_batchnorm"]:
                markers = BatchNormalization()(markers)

            markers = Dropout(params[f"marker_l{i}_dropout"])(markers)

        markers = Dense(
            params["join_size"],
            name="marker_join",
            activation="relu",
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(
                params[f"join_marker_l1"],
                params[f"join_marker_l2"]
            )
        )(markers)

        markers_model = Model(inputs=markers_input, outputs=markers)

        dists_input = Input(shape=(params["nsamples"], ), name="dist_input")
        dists = Dense(
            params["join_size"],
            name="dist_join",
            activation="relu",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"join_dist_l2"],
            )
        )(dists_input)
        dists_model = Model(inputs=dists_input, outputs=dists)

        join = Concatenate()([markers, dists])

        y = Dense(
            1,
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"output_l2"],
            )
        )(join)

        model = Model(inputs=[markers_model.input, dists_model.input], outputs=y)

        model.compile(
            optimizer=Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=["mae"]
        )
        return maf_pre, dist_pre, model

    def predict(self, model, X):

        maf_pre, dist_pre, model = model

        markers = maf_pre.transform(X)
        dists = dist_pre.transform(X)
        y = model([markers, dists]).numpy().reshape(-1)
        return y

    def fit(self, params, X, y, weights, individuals, **kwargs):
        # In lieu of a good way to reinitialise models, we do this.
        tf.keras.backend.clear_session()

        maf_pre, dist_pre, model = self.model(params)
        maf_trans = maf_pre.fit_transform(X, individuals=individuals)
        dist_trans = dist_pre.fit_transform(X, individuals=individuals)

        if weights is not None:
            weights = weights.reshape(-1, 1)

        model.fit(
            [maf_trans, dist_trans],
            y.reshape(-1, 1),
            epochs=params["epochs"],
            verbose=0,
            batch_size=100,
            shuffle=True,
            sample_weight=weights,
            **kwargs
        )
        return maf_pre, dist_pre, model 


class TFGatedModel(TestModel):

    use_weights = True

    def sample_params(self, trial):

        params = preprocessing_trial(
            trial,
            options=["maf", "maf_dist"]
        )
        params["reduce_size"] = trial.suggest_int("reduce_size", 50, 500)
        params["reduce_l1"] = trial.suggest_int("reduce_l1", 0, 200)
        params["reduce_l2"] = trial.suggest_int("reduce_l2", 0, 200)
        params["reduce_dropout"] = trial.suggest_float("reduce_dropout", 0, 1)
        params["nhidden"] = trial.suggest_int("nhidden", 0, 4)

        if params["nhidden"] > 0:
            params["layer_size"] = trial.suggest_int("layer_size", 5, 50)

        for i in range(params["nhidden"]):
            i += 1
            params[f"marker_l{i}_l2"] = trial.suggest_float(f"marker_l{i}_l2", 0, 200)
            params[f"marker_l{i}_linear_l2"] = trial.suggest_float(f"marker_l{i}_linear_l2", 0, 200)
            params[f"marker_l{i}_dropout"] = trial.suggest_float(f"marker_l{i}_dropout", 0, 1)

        params["output_l2"] = trial.suggest_float("output_l2", 0, 200)
        params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-2, 0.1)
        params["epochs"] = trial.suggest_int("epochs",  5, 50)
        return params

    def model(self, params):

        preprocessor = DropDuplicates(preprocessing_fns(params))
 
        if "dist" in params["preprocessor"]:
            in_shape = len(self.data.columns) - 2 + params["nsamples"]
        else:
            in_shape = len(self.data.columns) - 2
        # Subtract 2 because of response and invididual column
        markers_input = Input(shape=(in_shape, ), name="marker_input")

        markers = Dense(
            params["reduce_size"],
            name="reduce",
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(
                params["reduce_l1"],
                params["reduce_l2"]
            )
        )(markers_input)
        markers = Dropout(params["reduce_dropout"])(markers)

        for i in range(params["nhidden"]):
            i += 1
            markers = GatedResidualNetwork(
                params["layer_size"],
                params[f"marker_l{i}_dropout"],
                params[f"marker_l{i}_linear_l2"],
                params[f"marker_l{i}_l2"],
            )(markers)

        y = Dense(
            1,
            activation="linear",
            use_bias=True,
            kernel_regularizer=regularizers.l2(
                params[f"output_l2"],
            )
        )(markers)

        model = Model(inputs=markers_input, outputs=y)

        model.compile(
            optimizer=Adam(learning_rate=params["learning_rate"]),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=["mae"]
        )
        return preprocessor, model

    def predict(self, model, X):

        preprocessor, model = model

        markers = preprocessor.transform(X)
        y = model(markers).numpy().reshape(-1)
        return y

    def fit(self, params, X, y, weights, individuals, **kwargs):
        # In lieu of a good way to reinitialise models, we do this.
        tf.keras.backend.clear_session()

        preprocessor, model = self.model(params)
        X_trans = preprocessor.fit_transform(X, individuals=individuals)

        if weights is not None:
            weights = weights.reshape(-1, 1)

        model.fit(
            X_trans,
            y.reshape(-1, 1),
            epochs=params["epochs"],
            verbose=0,
            batch_size=100,
            shuffle=True,
            sample_weight=weights,
            **kwargs
        )
        return preprocessor, model 


class LocalGatedLinearUnit(tf.keras.layers.Layer):

    def __init__(self, size, l2norm):
        super(LocalGatedLinearUnit, self).__init__()
        self.linear = LocallyConnected1D(1, size, kernel_regularizer=regularizers.l2(l2norm))
        self.sigmoid = LocallyConnected1D(1, size, activation="sigmoid")
    
    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class LocalGatedResidualNetwork(tf.keras.layers.Layer):

    def __init__(self, size, dropout_rate, linear_l2norm, nl_l2norm):
        super(LocalGatedResidualNetwork, self).__init__()
        self.size = size
        self.elu_dense = LocallyConnected1D(1, size, activation="elu", kernel_regularizer=regularizers.l2(nl_l2norm))
        self.linear_dense = LocallyConnected1D(1, size, kernel_regularizer=regularizers.l2(linear_l2norm))
        self.dropout = Dropout(dropout_rate)
        self.gated_linear_unit = LocalGatedLinearUnit(size, linear_l2norm)
        self.layer_norm = tf.keras.layers.BatchNormalization()
        self.pad = tf.keras.layers.ZeroPadding1D(padding=(0, 1))
        return

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)

        inputs = self.pad(inputs)

        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class GatedLinearUnit(tf.keras.layers.Layer):

    def __init__(self, units, l2norm):
        super(GatedLinearUnit, self).__init__()
        self.linear = Dense(units, kernel_regularizer=regularizers.l2(l2norm))
        self.sigmoid = Dense(units, activation="sigmoid")
    
    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)


class GatedResidualNetwork(tf.keras.layers.Layer):

    def __init__(self, units, dropout_rate, linear_l2norm, nl_l2norm):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = Dense(units, activation="elu", kernel_regularizer=regularizers.l2(nl_l2norm))
        self.linear_dense = Dense(units, kernel_regularizer=regularizers.l2(nl_l2norm))
        self.dropout = Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units, linear_l2norm)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.project = Dense(units)
        return

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)

        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)

        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


MODELS = {
    "xgboost": XGBModel,
    "knn": KNNModel,
    "rf": RFModel,
    "ngboost": NGBModel,
    "svr": SVRModel,
    "extratrees": ExtraTreesModel,
    "elasticnet_dist": ElasticNetDistModel,
    "elasticnet": ElasticNetModel,
    "lassolars_dist": LassoLarsDistModel,
    "lassolars": LassoLarsModel,
    "tf_basic": TFModel,
    "tf_sum": TFSumModel,
    "tf_concat": TFConcatModel,
    "tf_gated": TFGatedModel,
}


def cli(prog, args):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="run a model",
    )

    parser.add_argument(
        "infile",
        type=str,
        help="the H5 file to test."
    )

    parser.add_argument(
        "-p", "--prefix",
        type=str,
        help="Prefix of the output files."
    )

    parser.add_argument(
        "-r", "--reponse",
        type=str,
        help="The response variable to optimise",
        choices=["resistance", "yield"],
        default="resistance"
    )

    parser.add_argument(
        "-c", "--chip",    
        type=int,
        help="Which SNP chip to use",
        default=0,
    )

    parser.add_argument(
        "-s", "--stat",
        type=str,
        help="The statistic to optimise for",
        default="mae",
        choices=["mae", "ndcg", "ndcgat50", "ndcgat90", "pearsons"]
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        help="The model to optimise",
        default=list(MODELS.keys())
    )

    parser.add_argument(
        "-n", "--ntrials",
        type=int,
        help="The number of trials to run.",
        default=200,
    )

    parser.add_argument(
        "-t", "--maxtime",
        type=int,
        help="The maximum time optimisation is allowed to take in seconds.",
        default=14400,
    )

    parser.add_argument(
        "-n", "--njobs",
        type=int,
        help="The number of parallel tasks to run.",
        default=-1,
    )
    return parser.parse_args(args)


def wrapper(
    infile,
    response,
    chip,
    model,
    stat,
    seed,
    prefix,
    ntrials,
    njobs,
    maxtime,
):
    if (response == "resistance"):
        response = "resistance_combo1"
        rank_reverse = True
    else:
        response = "yield1"
        rank_reverse = False

    training = prep_h5_training_dataset(infile, response, chip)

    direction = "minimize" if stat == "mae" else "maximize"
    stat = f"{stat}_test_means"

    study = optuna.create_study(direction=direction)

    model = MODELS[model](
        data=training,
        response=response,
        indiv="individual",
        stat=stat,
        rank_reverse=rank_reverse,
        seed=seed,
        model_name=prefix
    )

    study.optimize(model, n_trials=ntrials, n_jobs=njobs, timeout=maxtime)
    study_df = study.trials_dataframe()
    study_df["model_name"] = prefix
    study_df.to_csv(f"{prefix}_optimise.tsv", sep="\t")


    rank_trans = PercentileRankTransformer(
        [50, 90, 99],
        rank_reverse
    )
    rank_trans.fit(training[response], training["individual"])

    results_dfs = []
    preds_dfs = []
    best_model = model.best_model(study)
    for generation, df in get_h5_test_dataset(
        infile,
        response,
        chip
    ):
        stats, preds = eval_generations(
            df,
            model,
            best_model,
            response,
            rank_trans,
            rank_reverse
        )
        stats["generation"] = generation.decode()
        stats["model_name"] = prefix
        results_dfs.append(stats)

        preds["generation"] = generation.decode()
        preds["model_name"] = prefix
        preds_dfs.append(preds)
        break

    stats = pd.DataFrame(results_dfs)
    preds = pd.concat(preds_dfs)
    stats.to_csv(f"{prefix}_generations_stats.tsv", sep="\t")
    preds.to_csv(f"{prefix}_generations_preds.tsv", sep="\t")
    return study_df, stats, preds

def main():
    args = cli(prog=sys.argv[0], args=sys.argv[1:])
    wrapper(
        args.infile,
        args.response,
        args.chip,
        args.model,
        args.stat,
        args.seed,
        args.prefix,
        args.ntrials,
        args.njobs,
        args.maxtime
    )

    return


if __name__ == "__main__":
    main()
