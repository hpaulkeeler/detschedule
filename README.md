# Determinantal Scheduling for Wireless Networks

This repository contains MATLAB and Python implementations of determinantal point process (DPP) based scheduling algorithms for wireless networks. The code accompanies two research papers and provides tools for analyzing coverage probability and optimizing fairness in wireless network transmissions.

## Overview

Determinantal point processes provide a mathematical framework for randomly scheduling wireless network transmissions with built-in repulsion—nearby transmitters are less likely to transmit simultaneously, reducing interference. This approach generalizes classical Aloha protocols while remaining analytically tractable.

## Repository Structure

```
├── DemoDetPoisson.m / .py      # Standalone demo: DPP-thinned Poisson process
├── detcov_matlab/              # Coverage probability analysis (MATLAB)
├── detcov_python/              # Coverage probability analysis (Python)
├── detfairness_matlab/         # Fairness optimization (MATLAB)
├── detfairness_python/         # Fairness optimization (Python)
├── detcov_article.pdf          # Coverage probability paper (2020)
├── detfairness_article.pdf     # Fairness optimization paper (2025)
└── LICENSE                     # Apache 2.0
```

## Projects

### 1. Coverage Probability (`detcov_*`)

Analyzes SINR (Signal-to-Interference-plus-Noise Ratio) coverage probabilities under determinantal scheduling.

**Key files:**
- `ProbCovPairsDet` — Coverage analysis for transmitter-receiver pairs (bi-pole model)
- `ProbCovTXRXDet` — Coverage analysis where nodes are either transmitters or receivers
- `TestPalm` — Validates Palm distribution calculations empirically
- `funSimSimpleDPP` — DPP sampling using eigendecomposition

**Paper:** B. Błaszczyszyn, A. Brochard, and H.P. Keeler, "Coverage probability in wireless networks with determinantal scheduling," 2020.

### 2. Fairness Optimization (`detfairness_*`)

Extends coverage analysis to optimize scheduling for fairness, comparing three algorithms:

1. **Fixed Aloha** — Independent transmission with fixed probability *p*
2. **Adaptive Aloha** — Independent transmission with per-pair probabilities *p_i*
3. **Determinantal** — Correlated transmissions using DPP kernel matrix *K*

**Key files:**
- `FairPairsCompareDetAloha` — Main comparison script for all three scheduling methods
- `ProbCovPairsDetExactVsSamp` — Compares exact vs simulated coverage probabilities
- `funFairMeanDet` — Calculates average fairness across network configurations
- `funPairsL` — Constructs L-ensemble kernel matrix
- `funS` — Generates similarity matrices (Gaussian, Cauchy, or Bessel kernels)

**Paper:** B. Błaszczyszyn and H.P. Keeler, "Adaptive determinantal scheduling with fairness in wireless networks," 2025.

## Quick Start

### Demo: Determinantally-Thinned Poisson Process

The simplest entry point is the standalone demo that requires no dependencies beyond standard libraries:

**Python:**
```bash
python DemoDetPoisson.py
```

**MATLAB:**
```matlab
DemoDetPoisson
```

This simulates a Poisson point process and applies determinantal thinning, visualizing both the original and thinned processes.

### Coverage Probability Analysis

**Python:**
```bash
cd detcov_python
python ProbCovPairsDet.py
```

**MATLAB:**
```matlab
cd detcov_matlab
ProbCovPairsDet
```

### Fairness Optimization

**Python:**
```bash
cd detfairness_python
python FairPairsCompareDetAloha.py
```

**MATLAB:**
```matlab
cd detfairness_matlab
FairPairsCompareDetAloha
```

## Requirements

### Python
- NumPy
- SciPy
- Matplotlib

### MATLAB
- Statistics and Machine Learning Toolbox (optional, for some functions)

## Key Concepts

### Determinantal Point Processes

DPPs are probability distributions over subsets that exhibit repulsion—points are less likely to appear together if they are "similar" according to a kernel matrix. In wireless scheduling, this means nearby transmitters are less likely to transmit simultaneously.

### L and K Kernels

- **L kernel**: Defines the quality and similarity of potential transmitters
- **K kernel**: Related to L by K = L(I + L)⁻¹, gives marginal inclusion probabilities

### SINR Coverage

Coverage probability is P(SINR > τ) where:
- SINR = Signal power / (Interference + Noise)
- τ is the decoding threshold

## References

1. B. Błaszczyszyn and H.P. Keeler, "Determinantal thinning of point processes with network learning applications," 2018.

2. B. Błaszczyszyn, A. Brochard, and H.P. Keeler, "Coverage probability in wireless networks with determinantal scheduling," 2020.

3. B. Błaszczyszyn and H.P. Keeler, "Adaptive determinantal scheduling with fairness in wireless networks," 2025.

4. A. Kulesza and B. Taskar, "Determinantal point processes for machine learning," Foundations and Trends in Machine Learning, 2012.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
