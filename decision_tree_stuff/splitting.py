import abc
from dataclasses import dataclass
from typing import Any, Generator, Type

import polars as pl


@dataclass
class SplitParams:
    attribute: str
    threshold: float

    # @log_calls
    def split(
        self, samples: pl.DataFrame | pl.LazyFrame
    ) -> tuple[pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame]:
        return samples.filter(pl.col(self.attribute) <= self.threshold), samples.filter(
            pl.col(self.attribute) > self.threshold
        )

    def to_debug_string(self) -> str:
        return "{ " + f'"{self.attribute}" <= {self.threshold}' + " }"


class SplittingMethod(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def threshold_expr(cls, population_expr: pl.Expr) -> pl.Expr:
        ...

    @classmethod
    def compute_params(cls, samples: pl.DataFrame, splitting_key: str) -> SplitParams:
        return SplitParams(
            splitting_key,
            samples.select(cls.threshold_expr(pl.col(splitting_key))).to_series()[0],
        )

    @classmethod
    def split(cls, samples: pl.DataFrame, by: str) -> tuple[pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame]:
        return cls.compute_params(samples, by).split(samples)


class MidpointSplitter(SplittingMethod):
    @classmethod
    def threshold_expr(cls, expr: pl.Expr) -> pl.Expr:
        return (expr.min() + expr.max()) / 2.0


class MedianSplitter(SplittingMethod):
    @classmethod
    def threshold_expr(cls, expr: pl.Expr) -> pl.Expr:
        return expr.median()


class MeanSplitter(SplittingMethod):
    @classmethod
    def threshold_expr(cls, expr: pl.Expr) -> pl.Expr:
        return expr.mean()


class SplitMetric(abc.ABC):
    seek_minimum: bool

    @classmethod
    @abc.abstractmethod
    def eval_from_p1_expr(cls, p1_expr: pl.Expr) -> pl.Expr:
        ...


class EntropySplitMetric(SplitMetric):
    seek_minimum = True

    @classmethod
    def eval_from_p1_expr(cls, p1_expr: pl.Expr) -> pl.Expr:
        p0_expr = 1.0 - p1_expr
        return (-p0_expr * p0_expr.log(base=2) - (p1_expr * p1_expr.log(base=2))).fill_nan(0.0)


class GiniImpuritySplitMetric(SplitMetric):
    seek_minimum = True

    @classmethod
    def eval_from_p1_expr(cls, p1_expr: pl.Expr) -> pl.Expr:
        p0_expr = 1.0 - p1_expr
        return 1.0 - p0_expr.pow(2) - p1_expr.pow(2)


SPLIT_METRIC_LOOKUP: dict[str, Type[SplitMetric]] = {
    "entropy": EntropySplitMetric,
    "gini": GiniImpuritySplitMetric,
}

SPLIT_METHOD_LOOKUP: dict[str, Type[SplittingMethod]] = {
    "midpoint": MidpointSplitter,
    "median": MedianSplitter,
    "mean": MeanSplitter,
}


def compute_n_best_splits(
    samples: pl.DataFrame | pl.LazyFrame,
    metric: Type[SplitMetric] | str,
    method: Type[SplittingMethod] | str,
    n: int,
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(metric, str):
        metric = SPLIT_METRIC_LOOKUP[metric]

    if isinstance(method, str):
        method = SPLIT_METHOD_LOOKUP[method]

    splittable_attrs = list(set(samples.columns) - set(["class"]))
    assert len(splittable_attrs) > 0, "No splittable attrs"

    # fmt: off
    best_split_frame: pl.LazyFrame | pl.DataFrame = samples \
        .melt(id_vars='class', value_vars=splittable_attrs) \
        .with_columns(
            method.threshold_expr(pl.col('value'))
                .over('variable') \
                .alias('threshold')
        ) \
        .with_columns(
            (pl.col('value') <= pl.col('threshold')).alias('left')
        ) \
        .groupby('variable', 'threshold') \
        .agg(
            pl.col('class') \
                .filter(pl.col('left')) \
                .mean() \
                .fill_null(0.) \
                .alias('left_p1'),
            pl.col('class') \
                .filter(pl.col('left').is_not()) \
                .mean() \
                .fill_null(0.) \
                .alias('right_p1'),
        ) \
        .with_columns(
            (
                metric.eval_from_p1_expr(pl.col('left_p1')) + \
                metric.eval_from_p1_expr(pl.col('right_p1'))
            ).fill_nan(0.) \
            .alias('metric')
        ) \
        .sort('metric', descending=not metric.seek_minimum) \
        .limit(n)
    # fmt: on

    return best_split_frame


def next_best_split(
    samples: pl.DataFrame | pl.LazyFrame,
    metric: Type[SplitMetric] | str,
    method: Type[SplittingMethod] | str,
    n: int = 1,
) -> Generator[SplitParams, Any, Any]:
    eager = isinstance(samples, pl.DataFrame)

    all_splits = compute_n_best_splits(samples, metric, method, n)

    if not eager:
        assert isinstance(all_splits, pl.LazyFrame)
        all_splits = all_splits.collect()

    assert isinstance(all_splits, pl.DataFrame)
    best_split_dicts: list[dict[str, Any]] = all_splits.to_dicts()

    for best_split in best_split_dicts:
        best_split_params = SplitParams(best_split["variable"], best_split["threshold"])
        yield best_split_params
