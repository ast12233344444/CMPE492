### EAP (for filtering)

$f(x):$ Model, maps input domain to some activation of model in discussion.

$M(x_1, x_2):$ some metric we'll optimize for.

let $u, v$ be the nodes in computational graph so that edge $u \rightarrow v$ exists.

**1) Clean to Corrupted Extrapolation (select biggest)**

$l_{clean} = activation(x_{clean}, u, f)$
$I_{clean} = input(x_{clean}, v, f)$

$l_{corrupted} = activation(x_{corr}, u, f)$
$I_{corrupted} = input(x_{corr}, v, f)$

$$Score \approx (l_{corrupted} - l_{clean}) \cdot \frac{\partial M(x_{clean}, x_{corr})}{\partial I_{clean}}$$

note that this doesn't estimate the difference when we patch corrupted activations with clean ones. yet, this gives more accurate filterings on adversarial samples so we use this one.

---

**2) Corrupted to Clean Extrapolation (select smallest)**

$$Score \approx (l_{clean} - l_{corrupted}) \cdot \frac{\partial M(x_{corr}, x_{clean})}{\partial I_{corrupted}}$$

this directly estimates the effect we

### Edge Activation patching

Note: For simplicity, here I assume all parallel attention heads have dedicated layernorms dedicated to them, and we make all steerings are made to input of layernorm.

Let $f$ be the model, $E$ set of all computational Edges, $N$ set of all computational nodes, we're given $E'$ (set of edges to patch) and

$X_{corrupted}$: starting point
$X_{clean}$: place where we get patches

$l_{u, corr}$: activation of node u on corr. input
$l_{u, clean}$: activation of node u on clean input

$I_{v, corr}$: input of node v on corrupted input  \
$I_{v, clean}$: input of node v on clean input     | both pre-layernorm.

for all nodes $v \in N$ we iterate in **topological order** and apply

$$I_{v, patched} = I_{v, corr} + \sum_{\substack{e \in E' \\ e.destination=v}} (l_{e.source, clean} - l_{e.source, corr})$$

then we get the final state of the model.

for investigating adversarial examples, we apply given metric to classifier outputs (logits)

for investigating normal transformations we apply given metric to classifier inputs (residual stream)

### Edge Activation patching

Note: For simplicity, here I assume all parallel attention heads have dedicated layernorms dedicated to them, and we make all steerings are made to input of layernorm.

Let $f$ be the model, $E$ set of all computational Edges, $N$ set of all computational nodes, we're given $E'$ (set of edges to patch) and

$X_{corrupted}$: starting point
$X_{clean}$: place where we get patches

$l_{u, corr}$: activation of node $u$ on corr. input
$l_{u, clean}$: activation of node $u$ on clean input

$I_{v, corr}$: input of node $v$ on corrupted input
$I_{v, clean}$: input of node $v$ on clean input
*(both pre-layernorm)*

for all nodes $v \in N$ we iterate in **topological order** and apply

$$I_{v, patched} = I_{v, corr} + \sum_{\substack{e \in E' \\ e.destination=v}} (l_{e.source, clean} - l_{e.source, corr})$$

then we get the final state of the model.

For investigating adversarial examples, we apply given metric to classifier outputs (logits).

For investigating normal transformations we apply given metric to classifier inputs (residual stream).

---

### Edge loss Maximization

Similar to previous definitions we can define

$f(x)$: some activation of model
$M(x_1)$: some loss function of $f(x)$
$l_u$: activation of some computational node $u$
$I_v$: input of some computational node $v$
$e$: some given computational edge from $u \rightarrow v$
$X_{clean}$: clean inputs
$X'$: our optimized input

> $\alpha = 0.01$
> $X' = X_{clean}$
> 
> while True:
> $$Score = l_u(X') \cdot \text{detach}\left( \frac{\partial M(X')}{\partial I_v(X')} \right)$$
> $$X' = X' + \alpha \cdot \frac{\partial Score}{\partial X'}$$

this way we maximize the error introduced by a single edge.