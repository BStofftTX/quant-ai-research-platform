import pandas as pd
import pytest

from quant_ai_research_platform.modeling import (
    create_target,
    split_model_dataset,
)


def test_create_target_uses_next_day_return():
    data = pd.DataFrame({"Close": [100.0, 110.0, 105.0, 120.0]})

    target = create_target(data)

    assert target.tolist() == [1, 0, 1]


def test_split_model_dataset_preserves_time_order():
    dataset = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0, 1] * 5,
        }
    )

    train, test = split_model_dataset(dataset, train_fraction=0.8)

    assert len(train) == 8
    assert len(test) == 2
    assert train.index.max() < test.index.min()


def test_split_model_dataset_rejects_invalid_fraction():
    dataset = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0, 1] * 5,
        }
    )

    with pytest.raises(ValueError):
        split_model_dataset(dataset, train_fraction=1.0)
