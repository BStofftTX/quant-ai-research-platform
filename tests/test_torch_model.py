"""Tests for the optional PyTorch market classifier."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant_ai_research_platform.torch_model import TorchMarketClassifier


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.normal(size=(40, 3)),
        columns=["momentum", "volatility", "volume"],
    )
    target = pd.Series((features["momentum"] > 0).astype(int))
    return features, target


def test_probability_output(sample_data):
    features, target = sample_data
    model = TorchMarketClassifier(epochs=20)
    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (40, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_reproducible_predictions(sample_data):
    features, target = sample_data

    first = TorchMarketClassifier(seed=42, epochs=20)
    second = TorchMarketClassifier(seed=42, epochs=20)

    first.fit(features, target)
    second.fit(features, target)

    np.testing.assert_allclose(
        first.predict_proba(features),
        second.predict_proba(features),
    )


def test_cpu_predictions(sample_data):
    features, target = sample_data
    model = TorchMarketClassifier(device="cpu", epochs=20)
    model.fit(features, target)

    predictions = model.predict(features)

    assert predictions.shape == (40,)
    assert set(np.unique(predictions)).issubset({0, 1})
