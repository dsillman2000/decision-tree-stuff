import io
import json
import math
import tempfile

import polars as pl
import pytest
from decision_tree_stuff import DecisionTree, DecisionTreeParams
from decision_tree_stuff.dtree import DecisionNode, LeafNode
from decision_tree_stuff.splitting import (
    EntropySplitMetric,
    SplitParams,
    find_best_split,
)


@pytest.fixture
def smalldf() -> pl.DataFrame:
    return pl.DataFrame({
        'feature_1': [1., 2., 3., 4., 5.],
        'feature_2': [0., 0., 0., 0., 1.],
        'class': [1, 0, 0, 1, 0]
    })

def test_leaf_node_classify(smalldf):

    leaf = LeafNode(1, 'class')
    assert leaf.classify(smalldf).series_equal(pl.Series(name="prediction", values=[1, 1, 1, 1, 1]))

    leaf = LeafNode(0, 'class')
    assert leaf.classify(smalldf).series_equal(pl.Series(name="prediction", values=[0, 0, 0, 0, 0]))

def test_leaf_node_from_majority(smalldf):

    leaf = LeafNode.from_majority_class(smalldf["class"])
    assert leaf.label == 0

def test_decision_node_classify(smalldf):

    dec = DecisionNode('feature_2', 0.)
    dec.left = LeafNode(0, 'class')
    dec.right = LeafNode(1, 'class')
    assert dec.classify(smalldf).series_equal(pl.Series(name="prediction", values=[0, 0, 0, 0, 1]))

    dec = DecisionNode('feature_1', 2)
    dec.left = LeafNode(0, 'class')
    dec.right = LeafNode(1, 'class')
    assert dec.classify(smalldf).series_equal(pl.Series(name="prediction", values=[0, 0, 1, 1, 1]))

def test_entropy(smalldf: pl.DataFrame):

    assert math.isclose(smalldf.select(EntropySplitMetric.eval_from_p1_expr(pl.col('class').mean())).to_series()[0], 0.9709505944546686)

def test_split_params(smalldf):

    left, right = SplitParams("feature_1", 2.).split(smalldf)
    assert isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)

    assert left.frame_equal(pl.DataFrame({
        'feature_1': [1., 2.],
        'feature_2': [0., 0.],
        'class': [1, 0],
    }))
    assert right.frame_equal(pl.DataFrame({
        'feature_1': [3., 4., 5.],
        'feature_2': [0., 0., 1.],
        'class': [0, 1, 0],
    }))

    left, right = SplitParams("feature_1", 2.).split(smalldf.lazy())
    assert isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)

    assert left.collect().frame_equal(pl.DataFrame({
        'feature_1': [1., 2.],
        'feature_2': [0., 0.],
        'class': [1, 0],
    }))
    assert right.collect().frame_equal(pl.DataFrame({
        'feature_1': [3., 4., 5.],
        'feature_2': [0., 0., 1.],
        'class': [0, 1, 0],
    }))

@pytest.fixture
def heterodf() -> pl.DataFrame:
    """Classified by x_coord > 1."""
    return pl.DataFrame({
        'x_coord': [1., 2., 1., 1., 0., 2., 1.],
        'y_coord': [0., 1., 1., 1., 0., 0., 1.],
        'color': [0, 1, 0, 0, 0, 1, 0]
    })


def test_find_best_split(heterodf: pl.DataFrame):

    assert find_best_split(heterodf, 'color', 'entropy', 'midpoint') == SplitParams('x_coord', 1.)
    assert find_best_split(heterodf.lazy(), 'color', 'entropy', 'midpoint') == SplitParams('x_coord', 1.)

def test_decision_tree(heterodf: pl.DataFrame):

    dt = DecisionTree(DecisionTreeParams(['x_coord', 'y_coord'], 'color', 'midpoint', 'entropy'))
    dt.fit(heterodf)
    
    assert isinstance(dt.learned_tree, DecisionNode) and dt.learned_tree.to_params() == SplitParams('x_coord', 1.)
    assert isinstance(dt.learned_tree.left, LeafNode) and dt.learned_tree.left.label == 0
    assert isinstance(dt.learned_tree.right, LeafNode) and dt.learned_tree.right.label == 1
    
    assert dt.dict() == {
        'params': {
            'feature_columns': ['x_coord', 'y_coord'],
            'class_column': 'color',
            'max_depth': -1, 
            'min_split_entropy': 0.0, 
            'min_split_samples': 0,
            'split_metric': 'entropy', 
            'splitting_method': 'midpoint',
        },
        'depth': 0,
        'nodes': {'x_coord <= 1.0': {'color': 0}, 'x_coord > 1.0': {'color': 1}},
    }
    assert json.loads(dt.json(indent=4)) == dt.dict()
    
    dt = DecisionTree(DecisionTreeParams(['x_coord', 'y_coord'], 'color', 'midpoint', 'entropy'))
    dt.fit(heterodf.lazy())
    
    assert isinstance(dt.learned_tree, DecisionNode) and dt.learned_tree.to_params() == SplitParams('x_coord', 1.)
    assert isinstance(dt.learned_tree.left, LeafNode) and dt.learned_tree.left.label == 0
    assert isinstance(dt.learned_tree.right, LeafNode) and dt.learned_tree.right.label == 1
    
    assert dt.dict() == {
        'params': {
            'feature_columns': ['x_coord', 'y_coord'],
            'class_column': 'color',
            'max_depth': -1, 
            'min_split_entropy': 0.0, 
            'min_split_samples': 0,
            'split_metric': 'entropy', 
            'splitting_method': 'midpoint',
        },
        'depth': 0,
        'nodes': {'x_coord <= 1.0': {'color': 0}, 'x_coord > 1.0': {'color': 1}},
    }
    assert json.loads(dt.json(indent=4)) == dt.dict()

def test_save_and_load_json(heterodf: pl.DataFrame):

    dt = DecisionTree(DecisionTreeParams(['x_coord', 'y_coord'], 'color', 'midpoint', 'entropy'))
    dt.fit(heterodf)

    tmpf = tempfile.mktemp()
    dt.save_json(tmpf)
    dt2 = DecisionTree.load_json(tmpf)

    assert dt.dict() == dt2.dict()
