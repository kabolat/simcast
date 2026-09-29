# Simcast research documentation

This directory has two connected reading paths. `scientific/` explains what is
estimated, what is held fixed, what information is available at each forecast
origin, and how each dependence model is constructed. `technical/` explains
how those declared objects are configured, computed, stored, and reproduced.

Simcast asks one narrow question:

> If FM-derived entity-wise marginal quantile grids are fixed across methods,
> how much can spatial aggregate forecasts improve by replacing independent
> sampling with a learned same-lead cross-entity copula?

The distinction between *marginal forecasting* and *dependence modelling* is
fundamental. Chronos-2 supplies the native quantile grid; all supplied Liander
bases inherit deterministic isotonic monotonicity repair from the shared base
configuration. The resulting FM-derived grid is then fixed across M0--M4. Dependence models change only which marginal outcomes
occur together. They do not fine-tune Chronos, alter the fixed grid, or model
dependence between different forecast leads.

## Scientific documentation

Read these in order to understand the study:

1. [Research question, notation, and assumptions](scientific/01_research_problem.md)
2. [Data, forecast origins, and information sets](scientific/02_data_and_information_set.md)
3. [Frozen Chronos forecasts and PIT construction](scientific/03_chronos_and_pit.md)
4. [Dependence models M0--M4](scientific/04_dependence_models.md)
5. [Dependence fitting, scenario generation, and scores](scientific/05_training_sampling_scoring.md)
6. [Scientific workflow](scientific/06_scientific_workflow.md)
7. [Experiments, reporting, and result interpretation](scientific/07_experiments_and_results.md)

Within Chapters 1--6, the preferred order is definition and derivation, then a
worked physical or numerical example, then the configuration and
implementation guidance needed to reproduce that mathematical choice.

## Technical documentation

Use these documents to operate or inspect the repository:

1. [Usage guide: installation, fitting, evaluation, reporting, and outputs](technical/usage_guide.md)
2. [Configuration reference: base, method, composite, evaluation, and report fields](technical/configuration_reference.md)
3. [Artifact reference: cache, fit, evaluation, and report schemas](technical/artifact_reference.md)

The tracks deliberately cross-reference one another. A scientific chapter
states the statistical object and links to the relevant configuration or
artifact reference. A technical document links back to the chapter that
defines the estimand or method it operationalizes.
[Chapter 7](scientific/07_experiments_and_results.md) states the admissibility
criteria for numerical claims and the evidence required for reporting them.

The [interactive notebook guide](../notebooks/README.md) provides a third,
didactic route: notebooks use both scientific definitions and technical
interfaces, but are organized for explanation rather than as the source of
record for either.

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
             +--> realized values --> configured finite-quantile PIT
             |                         --> Gaussian scores z
             |                             (train/validation labels)
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
       matching configured marginal projection (identical marginals)
                           |
                           v
           entity scenarios --> cross-entity statistic T --> test scores
                           |
                           v
          report: paired effects against a declared reference
```

## Scope and status

This is a research proof of concept, not a production forecasting service. Its
strongest safeguards are scientific: exact upstream revision pins, chronological
splits, point-in-time availability, physically sealed test labels, fixed
marginals across methods, saved resolved configurations, and explicit model and
evaluation artifacts. The composite protocol declares all five methods,
multi-seed fitting for M2--M4, and origin-block uncertainty, but it still
cannot establish external validity from one year.

All notation used across the documentation is defined in
[01_research_problem.md](scientific/01_research_problem.md#indices-and-static-groups),
with a consolidated cross-chapter
[notation quick-reference](scientific/01_research_problem.md#notation-quick-reference).
