import json
from pathlib import Path

import simcast.cli.run_experiment as workflow
from simcast.config import SimcastConfig


def test_high_level_workflow_reuses_cache_and_runs_core_plus_bounded_kernel(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    trained: list[tuple[str, int, bool]] = []

    def fake_train(config, *, cache_dir, output_dir):
        assert cache_dir == cache
        path = Path(output_dir)
        path.mkdir()
        (path / "run_metadata.json").write_text(
            json.dumps({"group": {"name": config.data.entity_type, "entity_ids": ["a"], "entity_count": 1}}),
            encoding="utf-8",
        )
        trained.append((config.dependence.method, config.training.epochs, config.subset_training.enabled))
        return path

    def fake_evaluate(config, *, methods, method_runs, cache_dir, output_dir):
        del config
        assert list(method_runs) == list(methods)
        assert cache_dir == cache
        path = Path(output_dir)
        path.mkdir()
        return path

    monkeypatch.setattr(workflow, "train_from_config", fake_train)
    monkeypatch.setattr(workflow, "evaluate_from_config", fake_evaluate)
    monkeypatch.setattr(
        workflow,
        "build_cache_from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("existing cache must be reused")),
    )
    config = SimcastConfig.model_validate({"output": {"root_dir": tmp_path / "runs"}})
    result = workflow.run_experiment_from_config(
        config,
        cache_dir=cache,
        output_dir=tmp_path / "experiment",
        include_kernel_smoke=True,
    )

    assert result == (tmp_path / "experiment").resolve()
    assert [item[0] for item in trained] == [*workflow.CORE_METHODS, "conditional_kernel"]
    assert trained[-1] == ("conditional_kernel", 5, False)
    assert (result / "experiment_manifest.json").is_file()
    assert (result / "resolved_config.yaml").is_file()
