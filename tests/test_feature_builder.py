import math

import pytest
import torch

from simcast.cli.run_powertech_experiments import FEATURE_SETS
from simcast.fm.feature_builder import FeatureBuilder, ScalarStandardizer, lead_to_patch_indices


def _quantiles(median: torch.Tensor) -> torch.Tensor:
    return torch.stack((median - 2.0, median, median + 2.0), dim=-1)


def test_leads_map_to_shared_output_patches() -> None:
    torch.testing.assert_close(lead_to_patch_indices(8, 3), torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]))


def test_deterministic_feature_order_and_patch_mapping() -> None:
    embeddings = torch.tensor([[[[10.0, 11.0], [20.0, 21.0]]]])
    median = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5)
    predictions = _quantiles(median)
    locations = torch.tensor([[52.0, 5.0]])
    builder = FeatureBuilder(output_patch_size=3, standardize_scalar_features=False)

    features = builder.transform(embeddings, predictions, [0.1, 0.5, 0.9], locations)

    assert features.shape == (1, 1, 5, 10)
    torch.testing.assert_close(
        features[0, 0, :, :2],
        torch.tensor([[10.0, 11.0], [10.0, 11.0], [10.0, 11.0], [20.0, 21.0], [20.0, 21.0]]),
    )
    torch.testing.assert_close(features[0, 0, :, 2:5], torch.tensor([[-0.5, 0.0, 0.5]]).expand(5, -1))
    torch.testing.assert_close(features[0, 0, :, 5], median.flatten())
    torch.testing.assert_close(features[0, 0, :, 6], torch.full((5,), math.log(4.0)))
    torch.testing.assert_close(features[0, 0, :, 7], torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5]))
    torch.testing.assert_close(features[0, 0, :, 8:], locations.expand(5, -1))


def test_scalar_statistics_are_fit_once_on_training_origins() -> None:
    embeddings = torch.zeros(2, 1, 1, 1)
    training_medians = torch.tensor([1.0, 3.0]).reshape(2, 1, 1)
    validation_medians = torch.tensor([101.0]).reshape(1, 1, 1)
    builder = FeatureBuilder(
        output_patch_size=1,
        use_forecast_embedding=False,
        use_quantile_shape=False,
        use_log_spread=False,
        use_within_patch_position=False,
        use_location=False,
    )

    builder.fit(embeddings, _quantiles(training_medians), [0.1, 0.5, 0.9])
    assert builder.scalar_standardizer.mean is not None
    original_mean = builder.scalar_standardizer.mean.clone()
    standardized_training = builder.transform(embeddings, _quantiles(training_medians), [0.1, 0.5, 0.9])
    standardized_validation = builder.transform(embeddings[:1], _quantiles(validation_medians), [0.1, 0.5, 0.9])

    torch.testing.assert_close(standardized_training.flatten(), torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(standardized_validation.flatten(), torch.tensor([99.0]))
    torch.testing.assert_close(builder.scalar_standardizer.mean, original_mean)


def test_standardization_never_changes_raw_chronos_embedding() -> None:
    torch.manual_seed(3)
    embeddings = torch.randn(3, 2, 2, 4)
    medians = torch.randn(3, 2, 4)
    predictions = _quantiles(medians)
    locations = torch.tensor([[52.0, 5.0], [53.0, 6.0]])
    builder = FeatureBuilder(output_patch_size=2)

    features = builder.fit_transform(embeddings, predictions, [0.1, 0.5, 0.9], locations)

    patch_indices = torch.tensor([0, 0, 1, 1])
    torch.testing.assert_close(features[..., :4], embeddings.index_select(2, patch_indices))
    scalar_features = features[..., 4:]
    torch.testing.assert_close(
        scalar_features.mean(dim=(0, 1, 2)), torch.zeros(scalar_features.shape[-1]), atol=1e-6, rtol=0
    )


def test_transform_requires_training_fit_when_standardization_is_enabled() -> None:
    builder = FeatureBuilder(output_patch_size=2, use_location=False)
    embeddings = torch.zeros(1, 1, 1, 2)
    predictions = _quantiles(torch.ones(1, 1, 2))

    with pytest.raises(RuntimeError, match="fit on training origins"):
        builder.transform(embeddings, predictions, [0.1, 0.5, 0.9])


def test_scalar_standardizer_handles_constant_features_without_nan() -> None:
    values = torch.tensor([[1.0, 2.0], [1.0, 4.0]])
    standardizer = ScalarStandardizer().fit(values)

    transformed = standardizer.transform(values)

    assert torch.isfinite(transformed).all()
    torch.testing.assert_close(transformed[:, 0], torch.zeros(2))


@pytest.mark.parametrize(
    ("feature_set", "expected_dimension"),
    [
        ("embedding_dynamic_only", 5),
        ("quantile_dynamic_only", 6),
        ("combined_dynamic", 10),
        ("full", 12),
    ],
)
def test_confirmatory_ablation_feature_dimensions(feature_set: str, expected_dimension: int) -> None:
    settings = FEATURE_SETS[feature_set]
    builder = FeatureBuilder(output_patch_size=2, standardize_scalar_features=False, **settings)
    embeddings = torch.zeros(1, 2, 1, 4)
    predictions = _quantiles(torch.ones(1, 2, 2))
    locations = torch.tensor([[52.0, 5.0], [53.0, 6.0]]) if settings["use_location"] else None

    features = builder.transform(embeddings, predictions, [0.1, 0.5, 0.9], locations)

    assert features.shape == (1, 2, 2, expected_dimension)
