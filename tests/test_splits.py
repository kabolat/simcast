import pandas as pd
import pytest

from simcast.data.windows import chronological_split, generate_origins


def _origins(count: int, stride_steps: int) -> pd.DatetimeIndex:
    start = pd.Timestamp("2024-01-01 23:45", tz="UTC")
    return pd.date_range(
        start,
        periods=count,
        freq=pd.Timedelta(minutes=15 * stride_steps),
        name="origin_timestamp",
    )


def test_split_uses_floor_counts_for_default_protocol() -> None:
    origins = _origins(347, stride_steps=96)

    split = chronological_split(origins, horizon_steps=96)

    assert len(split.train) == 222
    assert len(split.validation) == 55
    assert len(split.test) == 70
    assert split.purged_train == ()
    assert split.purged_validation == ()
    assert split.train[-1] < split.validation[0] < split.test[0]


def test_split_floors_each_stage_not_the_final_partition_sizes() -> None:
    split = chronological_split(
        _origins(7, stride_steps=4),
        horizon_steps=4,
        tune_fraction=0.80,
        validation_fraction_within_tune=0.20,
    )

    # floor(7 * .8) = 5 tune; floor(5 * .2) = 1 validation.
    assert len(split.train) == 4
    assert len(split.validation) == 1
    assert len(split.test) == 2


def test_stride_shorter_than_horizon_purges_earlier_boundaries() -> None:
    origins = _origins(12, stride_steps=2)

    split = chronological_split(
        origins,
        horizon_steps=5,
        tune_fraction=0.75,
        validation_fraction_within_tune=1 / 3,
    )

    # Raw sizes are train=6, validation=3, test=3. With a two-step stride and
    # five-step horizon, two earlier origins overlap at each boundary.
    assert split.train == tuple(origins[:4])
    assert split.purged_train == tuple(origins[4:6])
    assert split.validation == (origins[6],)
    assert split.purged_validation == tuple(origins[7:9])
    assert split.test == tuple(origins[9:])

    train_targets = {
        origin + lead * pd.Timedelta(minutes=15)
        for origin in split.train
        for lead in range(1, 6)
    }
    validation_targets = {
        origin + lead * pd.Timedelta(minutes=15)
        for origin in split.validation
        for lead in range(1, 6)
    }
    test_targets = {
        origin + lead * pd.Timedelta(minutes=15)
        for origin in split.test
        for lead in range(1, 6)
    }
    assert train_targets.isdisjoint(validation_targets)
    assert validation_targets.isdisjoint(test_targets)


def test_stride_equal_to_horizon_needs_no_purge() -> None:
    origins = generate_origins(
        "2024-01-01 23:45+00:00",
        "2024-01-12 23:45+00:00",
        origin_stride_steps=4,
        frequency_minutes=15,
        origin_time="23:45",
    )
    origins = origins[:12]

    split = chronological_split(
        origins,
        horizon_steps=4,
        tune_fraction=0.75,
        validation_fraction_within_tune=1 / 3,
    )

    assert len(split.train) == 6
    assert len(split.validation) == 3
    assert len(split.test) == 3
    assert split.purged_train == ()
    assert split.purged_validation == ()


def test_split_rejects_non_chronological_origins() -> None:
    origin = pd.Timestamp("2024-01-01 23:45", tz="UTC")
    with pytest.raises(ValueError, match="strictly increasing"):
        chronological_split([origin, origin], horizon_steps=4)
