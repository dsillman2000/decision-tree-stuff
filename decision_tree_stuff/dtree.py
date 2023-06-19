import abc
from typing import NamedTuple, Optional, Type

import polars as pl
from decision_tree_stuff.splitting import (
    EntropySplitMetric,
    MeanSplitter,
    SplitMetric,
    SplitParams,
    SplittingMethod,
    next_best_split,
)


def get_majority(classes: pl.Series) -> int:
    assert classes.len() > 0, "Cannot get majority for empty classes"
    return classes.mode()[0]

class TreeNode(abc.ABC):
    @abc.abstractmethod
    def classify(self, samples: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        ...

    @abc.abstractmethod
    def to_debug_string(self) -> str:
        ...


class LeafNode(TreeNode):
    def __init__(self, label: int):
        self._label: int = label

    def to_debug_string(self) -> str:
        return '"class" = ' + str(self.label)

    @classmethod
    def from_majority_class(cls, classes: pl.Series) -> "LeafNode":
        assert classes.name == "class", "Expected `classes` Series name to be 'class', but got %s" % classes.name
        return cls(get_majority(classes))

    @property
    def label(self) -> int:
        return self._label

    def classify(self, samples: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        eager = isinstance(samples, pl.DataFrame)
        num_samples = samples.height if eager else samples.select("class").collect().height

        return pl.Series("prediction", [], dtype=pl.UInt8).extend_constant(self.label, num_samples)


class DecisionNode(TreeNode):
    def __init__(self, attribute: str, threshold: float, left_label: int, right_label: int):
        self._attribute: str = attribute
        self._threshold: float = threshold
        self._left: TreeNode = LeafNode(left_label)
        self._right: TreeNode = LeafNode(right_label)

    @property
    def attribute(self) -> str:
        return self._attribute

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def left(self) -> TreeNode:
        return self._left

    @left.setter
    def left(self, left_node: TreeNode):
        self._left = left_node

    @property
    def right(self) -> TreeNode:
        return self._right

    @right.setter
    def right(self, right_node: TreeNode):
        self._right = right_node

    def to_params(self) -> SplitParams:
        return SplitParams(self.attribute, self.threshold)

    def classify(self, samples: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        eager = isinstance(samples, pl.DataFrame)
        idxed_samples = samples.with_row_count()

        left_samples, right_samples = self.to_params().split(idxed_samples)

        left_preds: pl.DataFrame | pl.LazyFrame = left_samples.with_columns(
            self.left.classify(left_samples.drop("row_nr"))
        )

        right_preds: pl.DataFrame | pl.LazyFrame = right_samples.with_columns(
            self.right.classify(right_samples.drop("row_nr"))
        )

        if eager:
            assert isinstance(left_preds, pl.DataFrame) and isinstance(right_preds, pl.DataFrame)
            return pl.concat([left_preds, right_preds]).sort("row_nr")["prediction"].cast(pl.UInt8).alias("prediction")

        assert isinstance(left_preds, pl.LazyFrame) and isinstance(right_preds, pl.LazyFrame)
        return pl.concat([left_preds, right_preds]).sort("row_nr").select("prediction").collect().to_series()

    def to_debug_string(self) -> str:
        return (
            "{ "
            + f'"{self.attribute}" <= {self.threshold}'
            + " } ?"
            + "\n  t: "
            + self.left.to_debug_string().replace("\n", "\n| ")
            + "\n  f: "
            + self.right.to_debug_string().replace("\n", "\n| ")
        )


class DecisionTreeParams(NamedTuple):
    splitting_method: Type[SplittingMethod] | str = MeanSplitter
    split_metric: Type[SplitMetric] | str = EntropySplitMetric
    min_split_samples: int = 0
    min_split_entropy: float = 0.0
    max_depth: int = -1


class DecisionTree:
    def __init__(
        self,
        params: DecisionTreeParams,
        __root: Optional[TreeNode] = None,
        __depth: int = 0,
    ):
        self._params: DecisionTreeParams = params
        self._root: Optional[TreeNode] = __root
        self._depth: int = max(__depth, 0)
        self._entropy: float = -float("inf")
        self._left_subtree: Optional[DecisionTree] = None
        self._right_subtree: Optional[DecisionTree] = None

    @property
    def learned_tree(self) -> Optional[TreeNode]:
        return self._root

    def to_debug_string(self) -> str:
        if self._root is None:
            return "None"

        if isinstance(self._root, DecisionNode):
            assert self._left_subtree is not None and self._right_subtree is not None
            return "(E={:.4f}) ".format(self._entropy) + (
                "{ "
                + f'"{self._root.attribute}" <= {self._root.threshold}'
                + " } ?"
                + "\n  t: "
                + self._left_subtree.to_debug_string().replace("\n ", "\n  |")
                + "\n  f: "
                + self._right_subtree.to_debug_string().replace("\n ", "\n   ")
            )

        if isinstance(self._root, LeafNode):
            return "(E={:.4f}) ".format(self._entropy) + '"class" = ' + str(self._root.label)

        return ""

    # @log_calls
    def fit(self, dataset: pl.DataFrame | pl.LazyFrame, pruning_rounds: int = 0):
        eager = isinstance(dataset, pl.DataFrame)
        eager_classes = dataset.select("class") if eager else dataset.select("class").collect()

        if self._root is None:
            self._root = LeafNode.from_majority_class(eager_classes.to_series())

        root_entropy: float = eager_classes.select(
            EntropySplitMetric.eval_from_p1_expr(pl.col("class").mean()).fill_nan(0.0)
        ).to_series()[0]
        self._entropy = root_entropy

        if root_entropy == 0.0:
            return

        root_samples: int = dataset.height if eager else dataset.select("class").collect().height

        if (
            root_entropy >= self._params.min_split_entropy
            and root_samples >= self._params.min_split_samples
            and self._depth != self._params.max_depth
        ):
            for best_split in next_best_split(
                dataset,
                self._params.split_metric,
                self._params.splitting_method,
                n=max(pruning_rounds, 1),
            ):
                left, right = best_split.split(dataset)
                if not eager:
                    assert isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)
                    left, right = left.collect(), right.collect()
                assert isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)

                if min(left.height, right.height) == 0:
                    return

                left_label = get_majority(left["class"])
                right_label = get_majority(right["class"])

                self._root = DecisionNode(best_split.attribute, best_split.threshold, left_label, right_label)
                self._left_subtree = DecisionTree(self._params, self._root.left, self._depth + 1)
                self._right_subtree = DecisionTree(self._params, self._root.right, self._depth + 1)

                self._left_subtree.fit(left, pruning_rounds=pruning_rounds)
                self._right_subtree.fit(right, pruning_rounds=pruning_rounds)

                if pruning_rounds > 0 and len(set(self.leaf_classes())) == 1:
                    del self._left_subtree
                    del self._right_subtree
                    self._root = LeafNode.from_majority_class(eager_classes.to_series())
                    continue

                break

            if isinstance(self._root, DecisionNode):
                self._root.left = self._left_subtree.learned_tree # type: ignore
                self._root.right = self._right_subtree.learned_tree  # type: ignore

    def leaf_classes(self) -> list[int]:
        if isinstance(self._root, LeafNode):
            return [self._root.label]
        assert self._left_subtree is not None and self._right_subtree is not None
        return self._left_subtree.leaf_classes() + self._right_subtree.leaf_classes()

    def classify(self, samples: pl.DataFrame | pl.LazyFrame) -> pl.Series:
        if self._root is not None:
            return self._root.classify(samples)
        eager = isinstance(samples, pl.DataFrame)
        num_samples = samples.height if eager else samples.select("class").collect().height
        return pl.Series(name="prediction", values=[], dtype=pl.UInt8).extend_constant(0, num_samples)

    def transform(self, dataset: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return dataset.with_columns(self.classify(dataset))
