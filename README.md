# Thompson-Sampling

A from-scratch implementation of **Thompson Sampling** for the Bernoulli multi-armed bandit problem, benchmarked against **UCB** and a **greedy** baseline. Based on a 2018 research paper, with plots comparing how each strategy performs over time.

## Background

In the multi-armed bandit problem, an agent repeatedly chooses one of several "arms." Each arm returns a reward drawn from an unknown distribution — here, a Bernoulli distribution, so each pull returns a 0 or 1 with some hidden success probability. The agent's goal is to maximize total reward, which forces a trade-off: **explore** arms it's unsure about, or **exploit** the arm that currently looks best.

**Thompson Sampling** handles this trade-off in a Bayesian way. It maintains a Beta distribution over each arm's success probability (Beta is the natural conjugate prior for the Bernoulli likelihood). On each step it draws one sample from every arm's posterior and pulls the arm with the highest sample. Arms it's uncertain about have wide posteriors, so they still get picked often enough to be explored; as evidence accumulates, the posteriors sharpen and the agent converges on the best arm.

## What's included

This repo implements and compares the following strategies on Bernoulli bandits:

- **Thompson Sampling** — Beta-Bernoulli posterior sampling (the main algorithm).
- **UCB** — Upper Confidence Bound; picks arms by an optimism-under-uncertainty bonus.
- **Greedy** — always pulls the arm with the highest observed mean so far (baseline).

Each strategy is run over the same bandit setup, and the results are plotted to compare their performance.

<!-- TODO: if your "greedy" actually uses an exploration rate, rename it "ε-greedy" above and here. -->
<!-- TODO: if you plot cumulative regret specifically, say "cumulative regret over time" instead of "performance". -->

## Results

The plots compare the strategies over the course of the simulation.

<!-- TODO: drop your plot image(s) in the repo and reference them, e.g.: -->
<!-- ![Regret comparison](results/regret.png) -->

Briefly: Thompson Sampling and UCB both explore intelligently and pull ahead of the greedy baseline, which tends to lock onto a suboptimal arm early. <!-- adjust to match what your plots actually show -->

## Setup

```bash
git clone https://github.com/wglxd/Thompson-Sampling.git
cd Thompson-Sampling
pip install -r requirements.txt
```

Dependencies are just `numpy` and `matplotlib`.

<!-- TODO: if you don't have a requirements.txt, either create one with `pip freeze > requirements.txt` -->
<!-- or replace the line above with: pip install numpy matplotlib -->

## Usage

```bash
python main.py
```

<!-- TODO: replace main.py with your actual entry-point filename, and note any flags/config. -->

This runs the simulation across all strategies and generates the comparison plots.

## Reference

Based on **"A Tutorial on Thompson Sampling"** (Russo, Van Roy, Kappen, Osband, Wen), 2018.
Available at: https://arxiv.org/abs/1707.02038

<!-- TODO: confirm this is the paper you followed. If it's a different 2018 paper, swap the title/link. -->

## License

<!-- TODO: optional — add a license (MIT is a common, permissive default) if you want others to reuse this. -->