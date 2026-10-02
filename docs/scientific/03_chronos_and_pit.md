# Frozen marginal forecasts, quantile validity, and finite PIT observations

**Operational realization.** The `chronos.*` and `pit.*` arguments are listed
in the [configuration reference](../technical/configuration_reference.md), and
the cache fields that preserve the resulting fixed marginal record are listed
in the [artifact reference](../technical/artifact_reference.md).

This chapter explains how a frozen Chronos-2 forecast becomes the marginal
component of the joint probabilistic model. It proceeds from predictive random
variables to finite quantiles, quantile validity, historical rank observations,
and finally the covariates used by conditional dependence models.

The central separation is

$$
\underbrace{F_{k,\tau}^{(i)}}_{\text{entity marginal distribution}}
\quad+\quad
\underbrace{C_{g,\tau}^{(i)}}_{\text{cross-entity copula}}
\quad\Longrightarrow\quad
\underbrace{P(\{Y_{k,\tau}^{(i)}\}_{k\in\mathcal E_g}\mid\mathcal I^{(i)})}_{\text{joint forecast}}.
$$

Chronos-2 determines the finite representation of $F_{k,\tau}^{(i)}$. Methods
M0--M4 may change $C_{g,\tau}^{(i)}$, but they may not change any marginal
quantile value.

## 1. Frozen marginal forecasting

### Theory

For forecast instance $i$, entity $k$, and one-based lead $\tau$, let

$$
F_{k,\tau}^{(i)}(y)
=P(Y_{k,\tau}^{(i)}\le y\mid\mathcal I^{(i)}).
$$

$Y_{k,\tau}^{(i)}$ is the future random variable and $\mathcal I^{(i)}$ is the
information available at forecast origin $t^{(i)}$. Chronos-2 is used only to
approximate this marginal distribution. Its weights are frozen, it is placed in
evaluation mode, and no dependence-model loss is propagated into it.

The forecast for one physical entity is conditioned on that entity's own
history and admissible covariates. Distinct physical entities retain distinct
Chronos group identifiers. Consequently, the marginal forecaster does not
perform cross-entity learning; spatial interaction is introduced only by the
copula studied later.

### Example

Suppose the group contains three transformers. At origin
$t^{(i)}=\text{2024-06-01 23:45 UTC}$, Chronos receives seven days of transformer 1's
past load and the covariates available for its next day. It returns transformer
1's marginal quantiles. Transformers 2 and 3 are processed as distinct series.
M3 may later contextualize their *dependence features*, but it cannot revise
any of these three marginal forecasts.

### Configuration and implementation

| Scientific choice | Configuration | Implementation |
|---|---|---|
| frozen model identity | `chronos.model_id`, `chronos.model_revision` | `Chronos2FeatureExtractor` |
| source implementation | `chronos.source_revision` | `src/simcast/fm/chronos2_features.py` |
| numerical precision | `chronos.dtype`, `chronos.device` | `Chronos2FeatureExtractor.predict` |
| physical entities remain separate | always; no configuration | group-ID construction in `Chronos2FeatureExtractor` |
| lookback and horizon | `forecast.lookback_steps`, `forecast.horizon_steps` | cache construction in `build_cache_from_config` |

Each entity is a separate Chronos task. Cross-learning between entities would
change the scientific intervention and is not part of the study.

## 2. Finite native quantile representation

### Theory

Chronos does not return an analytic CDF. It returns $Q$ ordered probability
levels $0<q_1<\cdots<q_Q<1$ and corresponding predicted values

$$
\mathcal Q_{k,\tau}^{(i)}
=\{(\hat y_{k,\tau,q_j}^{(i)},q_j)\}_{j=1}^{Q}.
$$

Here $q_j$ is a probability, whereas $\hat y_{k,\tau,q_j}^{(i)}$ is a value in
the physical unit of the target. For example,
$\hat y_{k,\tau,0.9}^{(i)}=50$ means that the forecast assigns 90% probability
to an outcome no greater than 50, subject to the finite-grid approximation.

The current model supplies $Q=21$ levels. With horizon $H=96$ and output patch
size $S=16$, it also returns $P=H/S=6$ forecast-side representations
$e_{k,p}^{(i)}\in\mathbb R^{768}$. Leads 1--16 share representation $p=0$,
leads 17--32 share $p=1$, and so forth, with

$$
p(\tau)=\lfloor\frac{\tau-1}{16}\rfloor.
$$

### Example

For $H=96$, lead $\tau=35$ belongs to patch
$p(35)=\lfloor34/16\rfloor=2$. The quantile values remain lead-specific, but
the 768-dimensional Chronos representation is shared with leads 33--48. A
within-patch position covariate later distinguishes lead 35 from its neighbours.

### Configuration and implementation

The number and locations of native quantiles are properties of the pinned
Chronos model, not tunable Simcast probabilities. `forecast.horizon_steps`
controls $H$. `Chronos2FeatureExtractor.predict` returns
`quantile_predictions` and `forecast_embeddings`; `build_cache_from_config`
stores them as `quantile_prediction` and `forecast_embedding`.

## 3. Quantile crossing and monotone repair

### Theory

A valid quantile function is nondecreasing in probability. A crossing occurs
when a lower probability receives a larger predicted value than a higher
probability:

$$
\exists j\in\{1,\ldots,Q-1\}:\quad
\hat y_{k,\tau,q_j}^{(i)}>\hat y_{k,\tau,q_{j+1}}^{(i)}.
$$

Crossing makes the row inconsistent with any CDF. Two scientifically distinct
responses are supported:

1. `none`: declare the row invalid;
2. `isotonic`: replace it by the nearest nondecreasing sequence in squared
   Euclidean distance,

$$
\widetilde{\boldsymbol y}
=\arg\min_{x_1\le\cdots\le x_Q}
\sum_{j=1}^{Q}(x_j-\hat y_j)^2.
$$

This repair changes the marginal grid before dependence modelling. It is not a
copula operation. Once repaired, the same values are held fixed for every
dependence method.

### Numerical example

Consider levels $(0.1,0.5,0.9)$ and predictions $(20,35,32)$. The final two
values cross because $35>32$. Under `none`, this entity row is invalid. Under
isotonic least squares, the conflicting pair is pooled, yielding
$(20,33.5,33.5)$. The repaired row is nondecreasing, although two quantiles are
tied. Ties are valid and are treated deterministically.

The solar data provide an important empirical example: night-time forecasts
contain many raw crossings. All supplied bases apply isotonic repair before
both historical PIT construction and future scenario projection by default.
The `none` option remains available when crossed rows should instead be marked
invalid.

### Configuration and implementation

| Behaviour | Configuration | Function |
|---|---|---|
| reject crossed rows | `pit.monotone_repair: none` | `discretized_pit`, `interpolated_pit` |
| least-squares repair | `pit.monotone_repair: isotonic` | `repair_quantiles_isotonic` |
| crossing summaries | no additional argument | `quantile_crossings`, `crossing_diagnostics` |

All functions are in `src/simcast/fm/pit.py`. Raw crossing diagnostics are
computed before repair so the intervention remains visible.

## 4. Two finite-quantile PIT constructions

### Theory

If a complete CDF $F$ were known, the probability integral transform (PIT) of
an observation $y$ would be $u=F(y)$. Under a continuous calibrated predictive
distribution, $U$ would be uniform on $(0,1)$. Here only $Q$ quantile values are
known. A rule is therefore required to define the distribution between and
beyond those values. `pit.mode` makes that scientific choice explicit.

### Discretized midpoint mode

The $Q$ predicted quantiles partition the real line into $Q+1$ intervals:

$$
\begin{aligned}
I_0&=(-\infty,\hat y_{q_1}],\\
I_j&=(\hat y_{q_j},\hat y_{q_{j+1}}],\quad j=1,\ldots,Q-1,\\
I_Q&=(\hat y_{q_Q},\infty).
\end{aligned}
$$

The corresponding probability edges are

$$
a=(0,q_1,\ldots,q_Q,1).
$$

Because the observation identifies only a cell, not an exact CDF value, the
method assigns the midpoint of that cell's probability interval:

$$
c_j=\frac{a_j+a_{j+1}}2.
$$

The resulting $u=c_j$ is called a **pseudo-PIT**: it is a deterministic,
finite approximation to $F(y)$ rather than the exact continuous PIT. This is
the default `pit.mode: discretized` construction.

### Piecewise-linear interpolation mode

The alternative `pit.mode: linear_interpolation` defines a quantile function
directly from the same repaired native grid. Write
$x_j=\widetilde y_{q_j}$ for a valid nondecreasing row. Its inverse marginal map
is

$$
\widetilde Q(u)=
\begin{aligned}
x_1 &\quad && 0\le u\le q_1,\\
x_j+\dfrac{u-q_j}{q_{j+1}-q_j}(x_{j+1}-x_j)
&\quad && q_j<u<q_{j+1},\\
x_Q &\quad && q_Q\le u\le1.
\end{aligned}
$$

Thus the distribution is linear in probability between adjacent native
quantiles. It is constant outside their probability range: $x_1$ has boundary
mass $q_1$ and $x_Q$ has boundary mass $1-q_Q$. This conservative convention
does not invent lower or upper tails. If isotonic repair makes several adjacent
values equal, that flat segment creates an additional atom.

The corresponding forward PIT is the inverse of each strictly increasing
segment. At an atom, a realized value does not identify one unique probability,
so the deterministic transform uses the midpoint of the atom's probability
interval. Values below $x_1$ receive zero and values above $x_Q$ receive one;
`pit.eps` subsequently keeps $\Phi^{-1}$ finite.

### Numerical example

Let

$$
(q_1,q_2,q_3)=(0.1,0.5,0.9),\qquad
(\hat y_{q_1},\hat y_{q_2},\hat y_{q_3})=(20,30,50).
$$

There are four cells, not three:

| observation $y$ | value interval | probability interval | pseudo-PIT |
|---:|---|---|---:|
| $y\le20$ | $I_0$ | $(0,0.1)$ | $0.05$ |
| $20<y\le30$ | $I_1$ | $(0.1,0.5)$ | $0.30$ |
| $30<y\le50$ | $I_2$ | $(0.5,0.9)$ | $0.70$ |
| $y>50$ | $I_3$ | $(0.9,1)$ | $0.95$ |

If the realized value is $y=36$, discretized mode records the cell midpoint
$u=0.70$, then Gaussianizes it:

$$
z=\Phi^{-1}(0.70)\approx0.524.
$$

This value says the realization lies above the forecast median but below the
forecast 90% quantile. It does not claim that the exact CDF value is 0.70.

In linear-interpolation mode, the same realization gives

$$
u=0.5+\frac{36-30}{50-30}(0.9-0.5)=0.62.
$$

For an observation exactly equal to the lower endpoint 20, the interpolated
law has an atom spanning probability interval $[0,0.1]$, so the deterministic
mid-PIT is 0.05. An observation below 20 receives zero.

### Configuration and implementation

| Behaviour | Configuration | Function |
|---|---|---|
| finite-cell construction | `pit.mode: discretized` | `discretized_pit` |
| piecewise-linear construction | `pit.mode: linear_interpolation` | `interpolated_pit` |
| Gaussian clipping | `pit.eps` | `gaussianize_pit` |
| group-level construction | `pit.mode`, `pit.monotone_repair`, `pit.eps` | `build_group_pit` |

Equality with a predicted quantile enters the interval ending at that quantile
in discretized mode because `discretized_pit` uses a left-sided search. Linear
mode interpolates only between supported native quantiles and never
extrapolates a tail.

## 5. Complete spatial pseudo-observations

### Theory

First define entity-level validity:

$$
V_{k,\tau}^{(i)}=
\begin{cases}
1,&\text{if }y_{k,\tau}^{(i)}\text{ and every marginal quantile are finite, and the row is valid after repair},\\
0,&\text{otherwise.}
\end{cases}
$$

The complete-group indicator is

$$
V_{g,\tau}^{(i)}=\prod_{k\in\mathcal E_g}V_{k,\tau}^{(i)}.
$$

Thus $V_{g,\tau}^{(i)}=1$ exactly when every one of the $K_g$ entity rows is
valid. Only then is the spatial score vector defined:

$$
\mathbf z_{g,\tau}^{(i)}
=[z_{k,\tau}^{(i)}]_{k\in\mathcal E_g}
\in\mathbb R^{K_g}.
$$

### Example

For a five-wind-park group, suppose four entities have valid pseudo-PITs and the
fifth has a missing realization. The validity vector is $(1,1,1,1,0)$, hence
$V_{g,\tau}^{(i)}=0$. The whole five-dimensional vector is recorded as invalid.
The analysis never estimates a four-entity correlation for this case.

### Configuration and implementation

`protocol.ordered_entity_ids` and
`protocol.entity_count` declare the group. `build_group_pit` applies the
complete-vector rule and returns `valid_origin_lead`. `PITLibrary` stores all
invalid group scores as missing values so every dependence method receives the
same scientific sample.

## 6. Training-frequency sensitivity for discretized cells

### Theory

Nominal cell midpoints need not be uniformly occupied in the training period.
The sensitivity analysis estimates training cell frequencies
$\hat p_{k,\tau,c}$ and maps each cell to the midpoint of its empirical mass:

$$
	ilde{u}_{k,\tau,c}
=\sum_{r<c}\hat{p}_{k,\tau,r}+\frac{1}{2}\hat{p}_{k,\tau,c}.
$$

This map is estimated from training origins only and then frozen. It changes the
pseudo-scores used to estimate dependence; it does not change marginal
quantiles or scenario projection.

This transform is defined only for `pit.mode: discretized`, because it estimates
the probabilities of the $Q+1$ named cells. It is rejected with
`linear_interpolation`, whose non-atomic interior values do not belong to a
finite set of cells.

### Example

Suppose three finite cells have training frequencies $(0.2,0.5,0.3)$. Their
empirical midpoints are $(0.10,0.45,0.85)$. A validation observation assigned
to the second nominal cell receives $\widetilde u=0.45$ regardless of validation
frequencies. This avoids validation/test leakage.

### Configuration and implementation

`pit.dependence_transform: nominal_cells` is primary;
`pit.dependence_transform: training_frequency` selects the sensitivity.
`fit_training_frequency_midpoints`, `apply_training_frequency_midpoints`, and
`dependence_pit_scores` implement the train-only transformation.

## 7. Probability-space projection for scenarios

### Theory

Historical PIT construction maps an observed target $y$ to probability space.
Scenario projection is the opposite-direction operation: it maps a simulated
uniform probability $U$ through the configured inverse marginal map. The two
operations have different directions but use the same scientific definition
selected by `pit.mode`.

Under `discretized`, nearest-level boundaries are

$$
b_0=0,\qquad b_j=\frac{q_j+q_{j+1}}2, \qquad b_Q=1.
$$

Every $U\in(0,1)$ is assigned to one native level, and the scenario value is
exactly the corresponding $\hat y_{q_j}$. There is no interpolation between
values. Under `linear_interpolation`, scenario values follow
$\widetilde Q(U)$ from Section 4: they vary linearly between adjacent quantile
values, equal the first value for $U\le q_1$, and equal the last for
$U\ge q_Q$. Because every Gaussian-copula component has a uniform marginal,
all dependence methods preserve whichever finite-quantile marginal law the
base declares.

### Numerical example

Using levels $(0.1,0.5,0.9)$ gives boundaries $(0,0.3,0.7,1)$. Uniform draws in
$(0,0.3)$ select $\hat y_{0.1}=20$; draws in $(0.3,0.7)$ select
$\hat y_{0.5}=30$; draws in $(0.7,1)$ select $\hat y_{0.9}=50$. Therefore a draw
$U=0.72$ produces 50. Dependence methods change joint combinations such as
$(20,50,30)$ across entities, not the set of values available to one entity.

Under linear interpolation, the same $U=0.72$ lies between $q_2=0.5$ and
$q_3=0.9$, so

$$
\widetilde Q(0.72)
=30+\frac{0.72-0.5}{0.9-0.5}(50-30)=41.
$$

The dependence method still changes only the joint combination of marginal
draws, never the values of the native quantile knots.

### Configuration and implementation

`pit.mode` selects both historical PIT construction and scenario projection.
`project_uniforms_to_quantiles` in
`src/simcast/sampling/quantile_projection.py` performs it, and
`GaussianCopulaSampler` combines it with correlated uniforms. The configured
isotonic-repaired grid, when applicable, is the grid projected here.

This choice is unrelated to how a reported aggregate quantile is extracted
from the finite Monte Carlo ensemble after entity scenarios have been
combined: that always uses the nearest empirical order statistic.

## 8. Features for conditional dependence

### Theory

For conditional methods, define a feature vector measurable at the origin:

$$
v_{k,\tau}^{(i)}=
[e_{k,p(\tau)}^{(i)},\ r_{k,\tau,1:Q}^{(i)},\ m_{k,\tau}^{(i)},\
\log(|s_{k,\tau}^{(i)}|+\epsilon_s),\ w_\tau,\ \ell_k].
$$

Here $m$ is the median, $s=\hat y_{0.9}-\hat y_{0.1}$ is the 80% spread,
$r_j=(\hat y_{q_j}-m)/(|s|+\epsilon_s)$ is normalized quantile shape,
$w_\tau=((\tau-1)\bmod16)/15$ is within-patch position, and $\ell_k$ is the
optional latitude/longitude pair. Scalar means and variances are estimated on
training origins only and frozen for validation/test.

### Example

If $(\hat y_{0.1},\hat y_{0.5},\hat y_{0.9})=(20,30,50)$, then $m=30$ and
$s=30$. The three shape values are approximately $(-0.333,0,0.667)$. These
describe asymmetry and spread of the current marginal forecast while the
Chronos embedding describes its internal forecast context.

### Configuration and implementation

The conditional method's `features.use_*` arguments include or remove each
component; `features.shape_eps` controls $\epsilon_s$; and
`features.standardize_scalar_features` controls train-only standardization.
`FeatureBuilder` in `src/simcast/fm/feature_builder.py` defines the ordering,
fits training statistics, and applies the frozen transformation.

## 9. Scientific record retained in the cache

The cache is an immutable representation of the marginal experiment. It stores
the ordered entity group, forecast origins, split labels, observations,
quantile grids, Chronos representations, pseudo-PITs, Gaussianized scores,
validity indicators, and raw crossing diagnostics. Its metadata records the
dataset/model revisions and resolved scientific configuration.

### Example inspection

For a cached tensor shaped `[347, 15, 96, 21]`, the scientific interpretation
is 347 forecast instances, the same 15 ordered entities at every instance, 96
one-day leads, and 21 native quantile levels. It is not 347 independent samples:
the origins are chronological and uncertainty analysis resamples them in
blocks.

### Configuration and implementation

`base.output.cache_dir` selects the cache root. `base_fingerprint` derives the
scientific identity from entity order, revisions, forecast and information-set
rules, split, Chronos, and PIT construction; `locate_compatible_cache` verifies
that identity against metadata. `build_cache_from_config` creates the record,
while `save_pit_library` and `load_pit_library` write and read it. A directory
label alone never establishes compatibility.
