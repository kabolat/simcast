import numpy as np
import pytest
import torch
from chronos.chronos2 import Chronos2Model, Chronos2Pipeline
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Output

from simcast.fm import Chronos2FeatureExtractor


def _extractor() -> Chronos2FeatureExtractor:
    torch.manual_seed(11)
    config = Chronos2CoreConfig(
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        chronos_config={
            "context_length": 32,
            "input_patch_size": 4,
            "input_patch_stride": 4,
            "output_patch_size": 4,
            "quantiles": [0.1, 0.5, 0.9],
            "max_output_patches": 3,
            "use_reg_token": True,
            "use_arcsinh": False,
        },
    )
    return Chronos2FeatureExtractor(Chronos2Pipeline(Chronos2Model(config)))


def _inputs_with_covariates(prediction_length: int) -> list[dict[str, object]]:
    return [
        {
            "target": np.linspace(offset, offset + 1, 13, dtype=np.float32),
            "past_covariates": {
                "known": np.linspace(0, 1, 13, dtype=np.float32),
                "past_only": np.linspace(2, 3, 13, dtype=np.float32),
            },
            "future_covariates": {
                "known": np.linspace(1, 2, prediction_length, dtype=np.float32),
            },
        }
        for offset in (0.0, 10.0)
    ]


def test_extractor_freezes_chronos_and_runs_under_no_grad() -> None:
    extractor = _extractor()
    grad_modes: list[bool] = []

    def capture_grad_mode(_module: torch.nn.Module, _args: tuple[object, ...], _kwargs: dict[str, object]) -> None:
        grad_modes.append(torch.is_grad_enabled())

    handle = extractor.model.register_forward_pre_hook(capture_grad_mode, with_kwargs=True)
    try:
        extractor.predict(["a", "b"], _inputs_with_covariates(5), 5)
    finally:
        handle.remove()

    assert not extractor.model.training
    assert all(not parameter.requires_grad for parameter in extractor.model.parameters())
    assert grad_modes == [False]


def test_native_outputs_have_model_derived_dimensions() -> None:
    extractor = _extractor()
    result = extractor.predict(["a", "b"], _inputs_with_covariates(5), 5)

    assert result.entity_ids == ["a", "b"]
    torch.testing.assert_close(result.quantile_levels, torch.tensor([0.1, 0.5, 0.9]))
    assert result.quantile_predictions.shape == (2, 5, 3)
    assert result.forecast_embeddings.shape == (2, 2, 8)
    assert result.output_patch_size == 4
    assert result.quantile_predictions.dtype == torch.float32
    assert result.forecast_embeddings.dtype == torch.float32
    assert result.quantile_predictions.device.type == "cpu"
    assert result.forecast_embeddings.device.type == "cpu"


def test_wrapper_quantiles_match_standard_pipeline_prediction() -> None:
    extractor = _extractor()
    inputs = _inputs_with_covariates(5)

    standard = extractor.pipeline.predict(
        inputs,
        prediction_length=5,
        batch_size=64,
        cross_learning=False,
        limit_prediction_length=True,
        unrolled_quantiles=[0.1, 0.5, 0.9],
    )
    result = extractor.predict(["a", "b"], inputs, 5, batch_size=64)
    expected = torch.cat(standard, dim=0).permute(0, 2, 1)

    torch.testing.assert_close(result.quantile_predictions, expected, rtol=0, atol=0)


def test_target_rows_are_selected_with_covariate_rows_present() -> None:
    extractor = _extractor()
    captured_group_ids: list[torch.Tensor] = []
    captured_outputs: list[Chronos2Output] = []

    def capture_groups(_module: torch.nn.Module, _args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        group_ids = kwargs["group_ids"]
        assert isinstance(group_ids, torch.Tensor)
        captured_group_ids.append(group_ids.detach().cpu())

    def capture_output(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
        output: Chronos2Output,
    ) -> None:
        captured_outputs.append(output)

    pre_handle = extractor.model.register_forward_pre_hook(capture_groups, with_kwargs=True)
    post_handle = extractor.model.register_forward_hook(capture_output, with_kwargs=True)
    try:
        result = extractor.predict(["entity-0", "entity-1"], _inputs_with_covariates(5), 5, batch_size=64)
    finally:
        pre_handle.remove()
        post_handle.remove()

    # Each entity has target, past-only, and known-future rows. Rows inside an
    # entity may mix, but group IDs are distinct across physical entities.
    torch.testing.assert_close(captured_group_ids[0], torch.tensor([0, 0, 0, 1, 1, 1]))
    raw_output = captured_outputs[0]
    assert raw_output.quantile_preds is not None
    assert raw_output.forecast_embeds is not None
    target_rows = torch.tensor([0, 3])
    expected_quantiles = raw_output.quantile_preds[target_rows, :, :5].permute(0, 2, 1).cpu()
    expected_embeddings = raw_output.forecast_embeds[target_rows].cpu()
    torch.testing.assert_close(result.quantile_predictions, expected_quantiles)
    torch.testing.assert_close(result.forecast_embeddings, expected_embeddings)


def test_requires_exactly_one_target_per_entity() -> None:
    extractor = _extractor()

    with pytest.raises(ValueError, match="exactly one target variate"):
        extractor.predict(["multi-target"], [torch.randn(2, 12)], 4)


def test_rejects_horizon_beyond_direct_capacity() -> None:
    extractor = _extractor()

    with pytest.raises(NotImplementedError, match="single direct forward pass"):
        extractor.predict(["entity"], [torch.randn(12)], 13)


def test_entity_ids_must_match_preprocessed_inputs() -> None:
    extractor = _extractor()

    with pytest.raises(ValueError, match="equal length"):
        extractor.predict(["only-one-id"], [torch.randn(12), torch.randn(12)], 4)
