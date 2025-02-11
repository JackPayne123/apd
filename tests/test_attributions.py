"""Tests for attributions.py methods."""

import torch

from spd.attributions import calc_activation_attributions


def test_calc_activation_attributions_obvious():
    component_acts = {"layer1": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    expected = torch.tensor([[1.0, 1.0]])

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_different_d_out():
    component_acts = {
        "layer1": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        "layer2": torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),
    }
    expected = torch.tensor(
        [[1.0**2 + 2**2 + 5**2 + 6**2 + 7**2, 3**2 + 4**2 + 8**2 + 9**2 + 10**2]]
    )

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_with_n_instances():
    # Batch=1, n_instances=2, C=2, d_out=2
    component_acts = {
        "layer1": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
        "layer2": torch.tensor([[[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]),
    }
    expected = torch.tensor(
        [
            [
                [1.0**2 + 2**2 + 9**2 + 10**2, 3**2 + 4**2 + 11**2 + 12**2],
                [5**2 + 6**2 + 13**2 + 14**2, 7**2 + 8**2 + 15**2 + 16**2],
            ]
        ]
    )

    result = calc_activation_attributions(component_acts)
    torch.testing.assert_close(result, expected)
