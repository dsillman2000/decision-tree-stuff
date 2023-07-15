import abc
import io
import json
import operator
import os
from typing import Any, Dict, NamedTuple, Optional, Self, Type

import polars as pl

from decision_tree_stuff.splitting import (
    EntropySplitMetric,
    MeanSplitter,
    SplitMetric,
    SplitParams,
    SplittingMethod,
    find_best_split,
)


def get_majority(classes: pl.Series) -> int:
    assert classes.len() > 0, "Cannot get majority for empty classes"
    return classes.mode()[0]

class TreeNode(abc.ABC):
    @abc.abstractmethod
    def classify(self, samples: pl.DataFrame | pl.LazyFrame, idx_col: str = 'row_nr') -> pl.Series:
        ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, dict_repr: dict) -> Self:
        ...

    @abc.abstractmethod
    def dict(self) -> dict[str, Any]:
        ...


class LeafNode(TreeNode):
    def __init__(self, label: int, class_name: str='class'):
        self._label: int = label
        self._class_name: str = class_name

    @classmethod
    def from_majority_class(cls, classes: pl.Series) -> "LeafNode":
        # assert classes.name == "class", "Expected `classes` Series name to be 'class', but got %s" % classes.name
        return cls(get_majority(classes), class_name=classes.name)
    
    @classmethod
    def from_dict(cls, dict_repr: dict, class_name: str) -> "LeafNode":
        assert class_name in dict_repr.keys()
        return LeafNode(dict_repr[class_name], class_name)
    
    def dict(self) -> dict[str, Any]:
        return {self._class_name: self.label}

    @property
    def label(self) -> int:
        return self._label

    def classify(self, samples: pl.DataFrame | pl.LazyFrame, idx_col: str = 'row_nr') -> pl.Series:
        eager = isinstance(samples, pl.DataFrame)
        num_samples = samples.height if eager else samples.with_row_count().select('row_nr').collect().height

        return pl.Series("prediction", [], dtype=pl.UInt8).extend_constant(self.label, num_samples)


class DecisionNode(TreeNode):
    def __init__(self, attribute: str, threshold: float):
        self._attribute: str = attribute
        self._threshold: float = threshold
        self._left: Optional[TreeNode] = None
        self._right: Optional[TreeNode] = None

    @property
    def attribute(self) -> str:
        return self._attribute

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def left(self) -> Optional[TreeNode]:
        return self._left

    @left.setter
    def left(self, left_node: TreeNode):
        self._left = left_node

    @property
    def right(self) -> Optional[TreeNode]:
        return self._right

    @right.setter
    def right(self, right_node: TreeNode):
        self._right = right_node

    def to_params(self) -> SplitParams:
        return SplitParams(self.attribute, self.threshold)

    def classify(self, samples: pl.DataFrame | pl.LazyFrame, idx_col: str = 'row_nr') -> pl.Series:
        eager = isinstance(samples, pl.DataFrame)

        if idx_col not in samples.columns:
            idxed_samples = samples.with_row_count(idx_col)
        else:
            idxed_samples = samples

        left_samples, right_samples = self.to_params().split(idxed_samples)

        if self.left is not None and self.right is not None:
            left_preds: pl.DataFrame | pl.LazyFrame = left_samples.with_columns(
                self.left.classify(left_samples, idx_col=idx_col)
            )

            right_preds: pl.DataFrame | pl.LazyFrame = right_samples.with_columns(
                self.right.classify(right_samples, idx_col=idx_col)
            )
        else:
            raise Exception("Decision node missing one or more child.")
            # eager_classes_left, eager_classes_right = \
            #     left_samples.select(class_column).lazy().collect().to_series(), \
            #     right_samples.select(class_column).lazy().collect().to_series()
            # left_preds: pl.DataFrame | pl.LazyFrame = left_samples.with_columns(
            #     pl.lit(eager_classes_left.mode()[0]).alias("prediction")
            # )
            # right_preds: pl.DataFrame | pl.LazyFrame = right_samples.with_columns(
            #     pl.lit(eager_classes_right.mode()[0]).alias("prediction")
            # )

        if eager:
            assert isinstance(left_preds, pl.DataFrame) and isinstance(right_preds, pl.DataFrame)
            return pl.concat([left_preds, right_preds]).sort(idx_col)["prediction"].cast(pl.UInt8).alias("prediction")

        assert isinstance(left_preds, pl.LazyFrame) and isinstance(right_preds, pl.LazyFrame)
        return left_preds.merge_sorted(right_preds, idx_col).select("prediction").collect().to_series()
    
    def condition_str(self, lt: bool=True) -> str:
        _cmp_str: str = "<=" if lt else ">"
        return f"{self.attribute} {_cmp_str} {self.threshold}" 

    @classmethod
    def from_condition_str(cls, condition_str: str, lt: bool=True) -> "DecisionNode":
        attr, str_thresh = condition_str.split("<=" if lt else ">")
        return cls(attr.strip(), float(str_thresh.strip()))
    
    @classmethod
    def from_dict(cls, dict_repr: dict) -> "DecisionNode":
        lt_key: Optional[str] = next(filter(lambda k: "<=" in k, dict_repr.keys()), None)
        gt_key: Optional[str] = next(filter(lambda k: ">" in k, dict_repr.keys()), None)
        assert lt_key is not None
        assert gt_key is not None
        slf = cls.from_condition_str(lt_key)

        if any(['<=' in k for k in dict_repr[lt_key].keys()]):
            slf.left = DecisionNode.from_dict(dict_repr[lt_key])
        else:
            slf.left = LeafNode.from_dict(dict_repr[lt_key], list(dict_repr[lt_key].keys())[0])

        if any(['<=' in k for k in dict_repr[gt_key].keys()]):
            slf.right = DecisionNode.from_dict(dict_repr[gt_key])
        else:
            slf.right = LeafNode.from_dict(dict_repr[gt_key], list(dict_repr[gt_key].keys())[0])

        return slf
    
    def dict(self) -> dict[str, Any]:
        return {
            self.condition_str(lt=True): self.left.dict() if self.left is not None else None,
            self.condition_str(lt=False): self.right.dict() if self.right is not None else None
        }


class DecisionTreeParams(NamedTuple):
    feature_columns: list[str]
    class_column: str = 'class'
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

    def fit(self, dataset: pl.DataFrame | pl.LazyFrame, prune: bool = False):
        eager = isinstance(dataset, pl.DataFrame)
        class_name = self._params.class_column
        eager_classes = dataset.select(class_name) if eager else dataset.select(class_name).collect()

        if self._root is None:
            self._root = LeafNode.from_majority_class(eager_classes.to_series())

        root_entropy: float = eager_classes.select(
            EntropySplitMetric.eval_from_p1_expr(pl.col(class_name).mean()).fill_nan(0.0)
        ).to_series()[0]
        self._entropy = root_entropy

        if root_entropy == 0.0:
            return

        root_samples: int = dataset.height if eager else dataset.select(class_name).collect().height

        if (
            root_entropy >= self._params.min_split_entropy
            and root_samples >= self._params.min_split_samples
            and self._depth != self._params.max_depth
        ):
            best_split = find_best_split(dataset, class_name, self._params.split_metric, self._params.splitting_method)
            left, right = best_split.split(dataset)
            if not eager:
                assert isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)
                left, right = left.collect(), right.collect()
            assert isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)

            if min(left.height, right.height) == 0:
                return

            # left_label = get_majority(left[class_name])
            # right_label = get_majority(right[class_name])

            self._root = DecisionNode(best_split.attribute, best_split.threshold)
            self._left_subtree = DecisionTree(self._params, self._root.left, self._depth + 1)
            self._right_subtree = DecisionTree(self._params, self._root.right, self._depth + 1)

            self._left_subtree.fit(left, prune=prune)
            self._right_subtree.fit(right, prune=prune)

            if prune and len(set(self.leaf_classes())) == 1:
                del self._left_subtree
                del self._right_subtree
                self._root = LeafNode.from_majority_class(eager_classes.to_series())
                return

            assert isinstance(self._root, DecisionNode)
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
    
    @classmethod
    def from_dict(cls, dict_repr: dict[str, Any]) -> "DecisionTree":
        params = DecisionTreeParams(**dict_repr["params"])
        depth = dict_repr["depth"]
        if any(["<=" in k for k in dict_repr["nodes"].keys()]):
            root = DecisionNode.from_dict(dict_repr["nodes"])
        else:
            root = LeafNode.from_dict(dict_repr["nodes"], list(dict_repr["nodes"].keys())[0])
        return cls(params, root, depth)
    
    def dict(self) -> dict[str, Any]:
        params = dict(zip(self._params._fields, self._params))
        nodes = self._root.dict() if self._root is not None else None
        return {'params': params, 'depth': self._depth, 'nodes': nodes}
    
    def save_json(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write(self.json(indent=4))

    @classmethod
    def load_json(cls, filepath: str) -> "DecisionTree":
        with open(filepath, 'r') as f:
            return cls.from_dict(json.loads(f.read()))
        
    def json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.dict(), indent=indent)