from pathlib import Path
from typing import Any

from simcast.cli.download_data import allow_patterns, download_data


def test_download_patterns_are_selective_and_versioned() -> None:
    patterns = allow_patterns("transformer")
    assert patterns == [
        "liander2024_targets.yaml",
        "load_measurements/transformer/*.parquet",
        "weather_measurements/transformer/*.parquet",
        "weather_forecasts_versioned/transformer/*.parquet",
    ]
    assert all("weather_forecasts/" not in pattern for pattern in patterns)


def test_download_uses_validated_config_revision_and_local_dir(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "dataset"
    call: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        call.update(kwargs)
        return str(kwargs["local_dir"])

    result = download_data(
        config,
        overrides=[f"data.local_dir={destination}"],
        snapshot_download_fn=fake_snapshot_download,
    )
    assert result == destination
    assert call["repo_type"] == "dataset"
    assert call["revision"] == "dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044"
    assert call["allow_patterns"] == allow_patterns("transformer")
