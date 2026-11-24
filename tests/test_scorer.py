"""Scorer contracts checked against hand calculations and missing-value cases."""
import numpy as np
import pandas as pd
import pytest
from compartment.metrics import evaluate


def test_spearman_and_mae_by_hand():
    frame = pd.DataFrame(dict(target=['a', 'a', 'b', 'b'],
                              gold_Avg=[4.8, 4.5, 1.0, 1.3],
                              pred_Avg=[0.91, 0.62, 0.40, 0.35]))
    out = evaluate(frame, 'pv')[0]['outputs']['Avg']
    assert out['spearman']['value'] == pytest.approx(0.8)
    assert out['mae'] == pytest.approx((3.89 + 3.88 + 0.60 + 0.95) / 4)
    frame['pred_Avg'] *= -1
    assert evaluate(frame, 'pv')[0]['outputs']['Avg']['spearman']['value'] == pytest.approx(-0.8)


def test_mae_uses_pairwise_finite_rows_for_each_nn_output():
    frame = pd.DataFrame(dict(target=['a'] * 4,
        gold_ModAvg=[1, 3, np.nan, 4], pred_ModAvg=[2, 1, 0, np.inf],
        gold_HeadAvg=[2, np.nan, 4, 5], pred_HeadAvg=[2, 0, 2, 4]))
    out = evaluate(frame, 'nn')[0]['outputs']
    assert out['ModAvg']['mae'] == pytest.approx(1.5)
    assert out['ModAvg']['rmse'] == pytest.approx(np.sqrt(2.5))
    assert out['ModAvg']['n_valid'] == 2
    assert out['HeadAvg']['mae'] == pytest.approx(1)
    assert out['HeadAvg']['n_valid'] == 3


def test_mae_is_none_without_valid_pairs_and_defined_for_constant_predictions():
    frame = pd.DataFrame(dict(target=['a', 'a'], gold_Avg=[1, 3], pred_Avg=[2, 2]))
    out = evaluate(frame, 'pv')[0]['outputs']['Avg']
    assert out['mae'] == 1
    assert out['spearman']['value'] is None
    frame['pred_Avg'] = np.nan
    out = evaluate(frame, 'pv')[0]['outputs']['Avg']
    assert out['mae'] is None
    assert out['rmse'] is None
    assert out['n_valid'] == 0
