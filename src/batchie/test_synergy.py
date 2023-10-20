import numpy as np
import pytest
from batchie import synergy
from batchie.common import CONTROL_SENTINEL_VALUE


def test_create_single_treatment_effect_map():
    test_observations = np.array([1.0, 0.9, 0.1])

    test_treatment_ids = np.array(
        [
            [CONTROL_SENTINEL_VALUE, 1],
            [CONTROL_SENTINEL_VALUE, 2],
            [1, 2],
        ]
    )

    test_sample_ids = np.array([1, 1, 1])

    result = synergy.create_single_treatment_effect_map(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
    )

    assert result == {
        (1, 1): 1.0,
        (1, 2): 0.9,
        (1, CONTROL_SENTINEL_VALUE): 1.0,
    }


def test_create_single_treatment_effect_array():
    test_observations = np.array([0.9, 0.8, 0.1])

    test_treatment_ids = np.array(
        [
            [CONTROL_SENTINEL_VALUE, 1],
            [CONTROL_SENTINEL_VALUE, 2],
            [1, 2],
        ]
    )

    test_sample_ids = np.array([1, 1, 1])

    result = synergy.create_single_treatment_effect_array(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [1.0, 0.9],
                [1.0, 0.8],
                [0.9, 0.8],
            ],
        ),
    )


def test_synergy():
    test_observations = np.array([1.0, 1.0, 0.1])

    test_treatment_ids = np.array(
        [
            [CONTROL_SENTINEL_VALUE, 1],
            [CONTROL_SENTINEL_VALUE, 2],
            [1, 2],
        ]
    )

    test_sample_ids = np.array([1, 1, 1])

    (
        multi_treatment_sample_ids,
        multi_treatment_treatments,
        multi_treatment_synergy,
    ) = synergy.calculate_synergy(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
    )

    np.testing.assert_array_equal(multi_treatment_sample_ids, np.array([1]))
    np.testing.assert_array_equal(multi_treatment_treatments, np.array([[1, 2]]))
    np.testing.assert_array_equal(multi_treatment_synergy, np.array([0.9]))


def test_synergy_skips_in_lenient_mode():
    test_observations = np.array([1.0, 1.0, 0.1, 0.1])

    test_treatment_ids = np.array(
        [[CONTROL_SENTINEL_VALUE, 1], [CONTROL_SENTINEL_VALUE, 2], [1, 2], [3, 4]]
    )

    test_sample_ids = np.array([1, 1, 1, 1])

    (
        multi_treatment_sample_ids,
        multi_treatment_treatments,
        multi_treatment_synergy,
    ) = synergy.calculate_synergy(
        sample_ids=test_sample_ids,
        treatment_ids=test_treatment_ids,
        observation=test_observations,
        strict=False,
    )

    np.testing.assert_array_equal(multi_treatment_sample_ids, np.array([1]))
    np.testing.assert_array_equal(multi_treatment_treatments, np.array([[1, 2]]))
    np.testing.assert_array_equal(multi_treatment_synergy, np.array([0.9]))


def test_synergy_fails_in_strict_mode():
    test_observations = np.array([1.0, 1.0, 0.1, 0.1])

    test_treatment_ids = np.array(
        [[CONTROL_SENTINEL_VALUE, 1], [CONTROL_SENTINEL_VALUE, 2], [1, 2], [3, 4]]
    )

    test_sample_ids = np.array([1, 1, 1, 1])

    # assert raises value error
    with pytest.raises(ValueError):
        synergy.calculate_synergy(
            sample_ids=test_sample_ids,
            treatment_ids=test_treatment_ids,
            observation=test_observations,
            strict=True,
        )
