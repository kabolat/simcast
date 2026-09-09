# Scientific workflow

## 1. Objects and transformations

The experiment is a sequence of statistical transformations:

$$
\mathcal D_g
\xrightarrow{\text{information rule}}
\{\mathcal I^{(i)},\mathbf y_g^{(i)}\}_{i=1}^N
\xrightarrow{\text{frozen Chronos-2}}
\{\widehat y_{k,\tau,q_j}^{(i)}\}
\xrightarrow{f^{\mathrm{PIT}}}
\{\mathbf z_{g,\tau}^{(i)}\}
\xrightarrow{\text{dependence fit}}
\{R_{g,\tau}^{(i)}\}
\xrightarrow{\text{copula sampling}}
\{\widetilde A_{g,\tau}^{(i,m)}\}
\xrightarrow{\text{scores}}
\widehat{\mathcal R}.
$$

The base configuration fixes every transformation through the pseudo-PIT. The
method configuration changes only the map to $R_{g,\tau}^{(i)}$. A composite
configuration defines repeated comparisons and uncertainty analysis.

## 2. Complete cases

For group $g$, every tensor has the fixed entity axis
$\mathcal E_g=[k_1,\ldots,k_{K_g}]$. Define entity validity
$V_{k,\tau}^{(i)}\in\{0,1\}$ and group validity

$$V_{g,\tau}^{(i)}=\prod_{k\in\mathcal E_g}V_{k,\tau}^{(i)}.$$

Only cases with $V_{g,\tau}^{(i)}=1$ enter dependence fitting or evaluation.
The dataset object enumerates these complete $(i,\tau)$ pairs; the collator
stacks them without selecting entities; and the trainer asserts an entity
dimension of exactly $K_g$. Evaluation checks that every supplied correlation
is $K_g\times K_g$.

Example: if one of 15 transformer observations is missing at lead 32 of origin
$i$, all 15 components of that one case are excluded. The other leads remain
eligible if complete.

Implementation: `DependenceDataset`, `DependenceCollator`,
`ConditionalTrainer.expected_num_entities`, and `evaluate_from_config` enforce
the rule independently.

## 3. Frozen-marginal record

Chronos is called separately for each physical entity. The record stores named
axes for origins, entities, leads, native quantiles, output patches, and hidden
coordinates. Test realizations and derived test pseudo-PIT values are stored in
a sealed Zarr group inaccessible through training mode.

The record is identified by the base fingerprint, not by a manually chosen
name. The fingerprint covers the complete ordered group, pinned data, temporal
protocol, information set, chronological split, Chronos revisions, and PIT
construction. This allows one expensive marginal computation to support all
five dependence methods without confusing scientifically different forecasts.

Example: changing M2 rank does not rebuild Chronos. Changing vintage weather to
oracle changes $\mathcal I^{(i)}$ and therefore requires a different cache.

Implementation: `build_cache_from_config` constructs the record;
`base_fingerprint` and `locate_compatible_cache` govern reuse.

## 4. Dependence fitting

M0 writes the identity specification. M1 estimates shrinkage correlation from
training pseudo-scores. M2--M4 minimize Gaussian-copula negative
pseudo-log-likelihood over all complete training cases and choose a checkpoint
by chronological validation pseudo-NLL. Test outcomes do not influence this
choice.

The public term is **dependence fitting** because M0 is specified rather than
trained and M1 has a closed-form estimator. Inside gradient-based code,
training and optimization retain their ordinary mathematical meanings.

M4 follows exactly the same partition and stopping semantics as M2 and M3. A
small preliminary budget is an explicit composite override applicable to any
of these three methods.

## 5. Singular transformation

`run_singular` combines one base with one method, obtains the compatible
marginal record, fits that method, and evaluates it. An optional explicit
reference is fitted and evaluated with common random numbers. The saved base
and method remain separate, preserving the logic of controlled comparison.

```bash
uv run python -m simcast.cli.run_singular \
  --base configs/bases/liander2024/transformer.yaml \
  --method configs/methods/m3_set_aware_low_rank.yaml \
  --reference-method configs/methods/m0_independent.yaml
```

## 6. Composite transformation

`run_composite` resolves only explicitly listed entries. Each cell is the tuple

$$c=(\text{base entry},\text{method entry},\text{seed or deterministic}).$$

The expansion manifest is an auditable enumeration of these tuples. Identical
deterministic fits are shared. Conditional fits remain seed-specific. Each
non-reference cell is evaluated with its declared reference on identical cases
and random draws.

```bash
uv run python -m simcast.cli.run_composite \
  --config configs/venues/<venue>/main.yaml
```

The paired origin-level loss for method $a$ against reference $b$ is

$$d_{a,b,g}^{(i)}=S_{a,g}^{(i)}-S_{b,g}^{(i)},$$

where valid leads are averaged within origin. A moving-block bootstrap samples
chronological blocks of $d^{(i)}$ to retain short-range temporal dependence.

## 7. Resume and immutability

Before a composite starts, the fully resolved base, method, override, and seed
definitions are hashed. Resume succeeds only if that digest equals the stored
manifest digest. A cell marked complete is skipped only when its recorded
outputs pass completion checks. A changed composite must use a new run
identifier.

Existing scientific runs and reports are never migrated in place. The generic
legacy reader parses historical evaluation manifests as unmodified JSON; it
does not make them executable under the current protocol.

## 8. Relation between CLI and notebooks

The CLI is the authoritative route for full reproduction. Notebooks use the
same role-specific loaders, fingerprint lookup, runtime adapter, dependence
fitter, and evaluator. They expose intermediate equations and arrays for
interpretation, while write-producing cells default to disabled.

Thus the two interfaces differ in exposition, not in statistical
transformation. A notebook calculation is comparable with a CLI result only
when its resolved base, method, seed, and cache fingerprint coincide.
