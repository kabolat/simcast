import numpy as np
import torch

from simcast.dependence import IndependentCopula, StaticGaussianCopula
from simcast.evaluation.aggregate import evaluate_aggregate_ensemble
from simcast.fm.pit import build_group_pit
from simcast.sampling.gaussian_copula import generate_scenarios


def test_cpu_data_to_pit_static_sampling_and_aggregate_metrics() -> None:
    rng = np.random.default_rng(27)
    entity_ids = ["transformer::a", "transformer::b", "transformer::c"]
    covariance = np.asarray([[1.0, 0.7, 0.2], [0.7, 1.0, 0.1], [0.2, 0.1, 1.0]])
    true_y = np.stack([rng.multivariate_normal(np.zeros(3), covariance, size=50) for _ in range(2)], axis=-1)
    truth = torch.tensor(true_y, dtype=torch.float32)  # [origin, entity, lead]
    levels = torch.tensor([0.1, 0.5, 0.9])
    marginal_values = torch.tensor([-1.2816, 0.0, 1.2816])
    predictions = marginal_values.expand(50, 3, 2, 3).clone()
    pit = build_group_pit(truth, predictions, levels)
    assert pit.valid_origin_lead.all()

    static = StaticGaussianCopula().fit(pit.z[:40], entity_ids)
    independent = IndependentCopula(entity_ids, n_leads=2)
    method_samples: dict[str, torch.Tensor] = {}
    for method_name, model in (("independent", independent), ("static_gaussian", static)):
        cases = []
        for origin in range(40, 50):
            by_lead = []
            for lead in (1, 2):
                generator = torch.Generator().manual_seed(1000 + origin * 2 + lead)
                scenario = generate_scenarios(
                    model.correlation_matrix(lead).to(torch.float32),
                    predictions[origin, :, lead - 1],
                    levels,
                    num_samples=512,
                    generator=generator,
                )
                by_lead.append(scenario.aggregate_samples)
            cases.append(torch.stack(by_lead))
        method_samples[method_name] = torch.stack(cases)

    true_aggregate = truth[40:].sum(dim=1)
    for samples in method_samples.values():
        report = evaluate_aggregate_ensemble(samples, true_aggregate, levels)
        assert report.quantile_predictions.shape == (10, 2, 3)
        assert np.isfinite(list(report.overall.values())).all()
        assert list(report.by_lead.index) == [1, 2]
