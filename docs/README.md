# Simcast research documentation

This directory is the scientific record for the Simcast proof of concept. It
explains what is estimated, what is held fixed, which information is available
at each forecast origin, how every dependence model is constructed, and how the
reported experiments can be reproduced.

Simcast asks one narrow question:

> If FM-derived entity-wise marginal quantile grids are fixed across methods,
> how much can spatial aggregate forecasts improve by replacing independent
> sampling with a learned same-lead cross-entity copula?

The distinction between *marginal forecasting* and *dependence modelling* is
fundamental. Chronos-2 supplies the native quantile grid; solar receives a
deterministic isotonic monotonicity repair. The resulting FM-derived grid is
then fixed across M0--M4. Dependence models change only which marginal outcomes
occur together. They do not fine-tune Chronos, alter the fixed grid, or model
dependence between different forecast leads.

## Suggested reading paths

For a scientific reader, read these in order:

1. [Research question, notation, and assumptions](01_research_problem.md)
2. [Data, forecast origins, and information sets](02_data_and_information_set.md)
3. [Frozen Chronos forecasts and PIT construction](03_chronos_and_pit.md)
4. [Dependence models M0--M4](04_dependence_models.md)
5. [Dependence fitting, scenario generation, and scores](05_training_sampling_scoring.md)
6. [Experimental protocol and results](08_experiments_and_results.md)
7. [Confirmatory experimental protocol](12_confirmatory_protocol.md)
8. [Method-notebook scientific companion](13_method_notebook_companion.md)

Within Chapters 1--6, the preferred order is definition and derivation, then a
worked physical or numerical example, then the configuration and
implementation guidance needed to reproduce that mathematical choice.
Chapters 7--10 translate the same objects into executable experiments and
scientific records; Chapters 11--13 delimit inference and provide independent
method monographs.

Historical tracked summaries may describe earlier exploratory configurations.
They are not silently relabeled as results of the current role-separated
composites; [Chapter 8](08_experiments_and_results.md) states the admissibility
criteria for new numerical claims.

For a researcher running or extending the code:

1. [Scientific workflow](06_scientific_workflow.md)
2. [Running and reproducing experiments](07_running_experiments.md)
3. [Interactive notebook guide](../notebooks/README.md)
4. [Configuration reference](09_configuration_reference.md)
5. [Artifact schemas and output files](10_artifact_reference.md)
6. [Validation, limitations, and extension guide](11_validation_and_extension.md)

## One-page conceptual map

```text
Liander targets + point-in-time weather + entity metadata
                           |
                           v
        aligned, leakage-safe origin windows [N,K,L/H]
                           |
                           v
            frozen and revision-pinned Chronos-2
             |                            |
             |                            +--> output-patch embeddings
             v
        native quantiles [N,K,H,Q]
             |
             +--> realized values --> discrete PIT --> Gaussian scores z
             |                                      (train/validation labels)
             |
             +--> fixed feature construction <------ embeddings + location
                           |
                           v
       full-group M0 independent / M1 static / M2--M4 conditional
                           |
                           v
          same-lead Gaussian-copula uniforms [M,K]
                           |
                           v
       nearest-native-quantile projection (identical marginals)
                           |
                           v
           entity scenarios --> spatial sum --> test scores
```

## Scope and status

This is a research proof of concept, not a production forecasting service. Its
strongest safeguards are scientific: exact upstream revision pins, chronological
splits, point-in-time availability, physically sealed test labels, fixed
marginals across methods, saved resolved configurations, and explicit model and
evaluation artifacts. Existing archived results remain historical exploratory
evidence. The current composite protocol declares all five methods, multi-seed
fitting for M2--M4, and origin-block uncertainty, but it still cannot establish
external validity from one year.

All notation used across the documentation is defined in
[01_research_problem.md](01_research_problem.md#indices-and-static-groups).
