"""Shared numerical settings for the R05 one-dimensional theory benchmark."""

import numpy as np


SPACE_TEST_COUNT = 25
SPACE_TEST_MIN = 0.0
SPACE_TEST_MAX = 4.0
SPACE_TEST_WIDTH = 0.25
SPACE_TEST_CENTERS = np.linspace(
    SPACE_TEST_MIN,
    SPACE_TEST_MAX,
    SPACE_TEST_COUNT,
)
TIME_TEST_COUNT = 8
TIME_SUPPORT_FACTOR = 1.8


def apply_fixed_1d_tests(config, *, time_test_type="bump", horizon=None):
    """Apply the trajectory-independent R05 test family to a 1D config."""
    config["space_tests"].update(
        type="gaussian",
        layout="grid",
        L=SPACE_TEST_COUNT,
        centers=SPACE_TEST_CENTERS.tolist(),
        s=SPACE_TEST_WIDTH,
    )
    config["time_tests"].update(type=time_test_type, K=TIME_TEST_COUNT)
    if horizon is not None:
        config["time_tests"]["support"] = (
            TIME_SUPPORT_FACTOR * float(horizon) / (TIME_TEST_COUNT + 1)
        )
    return config
