from dataclasses import fields

import torch
from chronos.chronos2 import Chronos2Model
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Output


def _tiny_model() -> Chronos2Model:
    torch.manual_seed(7)
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
    return Chronos2Model(config).eval()


def test_output_contract_exposes_forecast_embeds_with_patch_shape() -> None:
    model = _tiny_model()
    context = torch.arange(39, dtype=torch.float32).reshape(3, 13)

    with torch.no_grad():
        output = model(context=context, group_ids=torch.arange(3), num_output_patches=2)

    assert "forecast_embeds" in {field.name for field in fields(Chronos2Output)}
    assert output.forecast_embeds is not None
    assert output.forecast_embeds.shape == (3, 2, model.config.d_model)


def test_exposed_features_are_the_quantile_head_input() -> None:
    model = _tiny_model()
    captured_head_input: list[torch.Tensor] = []
    captured_head_output: list[torch.Tensor] = []

    def capture_head(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured_head_input.append(args[0].detach().clone())
        captured_head_output.append(output.detach().clone())

    handle = model.output_patch_embedding.register_forward_hook(capture_head)
    try:
        with torch.no_grad():
            output = model(context=torch.randn(2, 15), num_output_patches=3)
    finally:
        handle.remove()

    assert output.forecast_embeds is not None
    torch.testing.assert_close(output.forecast_embeds, captured_head_input[0], rtol=0, atol=0)
    with torch.no_grad():
        replayed_head_output = model.output_patch_embedding(output.forecast_embeds)
    torch.testing.assert_close(replayed_head_output, captured_head_output[0], rtol=0, atol=0)


def test_quantile_predictions_match_the_unmodified_numerical_path() -> None:
    """The extra return value must not alter the legacy quantile computation."""

    model = _tiny_model()
    context = torch.tensor([[1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    num_output_patches = 2

    with torch.no_grad():
        output = model(context=context, group_ids=torch.arange(2), num_output_patches=num_output_patches)
        encoder_output, loc_scale, _, _ = model.encode(
            context=context,
            group_ids=torch.arange(2),
            num_output_patches=num_output_patches,
        )
        hidden_states = encoder_output.last_hidden_state
        assert hidden_states is not None
        forecast_embeds = hidden_states[:, -num_output_patches:]
        raw_head = model.output_patch_embedding(forecast_embeds)

        batch_size = context.shape[0]
        num_quantiles = len(model.chronos_config.quantiles)
        patch_size = model.chronos_config.output_patch_size
        normalized_quantiles = (
            raw_head.reshape(batch_size, num_output_patches, num_quantiles, patch_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_quantiles, num_output_patches * patch_size)
        )
        legacy_quantiles = model.instance_norm.inverse(
            normalized_quantiles.reshape(batch_size, -1), loc_scale
        ).reshape(batch_size, num_quantiles, num_output_patches * patch_size)

    assert output.quantile_preds is not None
    torch.testing.assert_close(output.quantile_preds, legacy_quantiles, rtol=0, atol=0)
