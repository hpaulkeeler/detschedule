# detfairness_python

Adaptive determinantal scheduling with fairness in wireless networks (Python implementation).

## Overview

This code studies determinantal scheduling when maximizing coverage probability (the tail distribution of SINR) or fairness (a function of coverage probability).

The code generates network configurations of n transmitter-receiver pairs (x_1,y_1),...,(x_n,y_n) using the bi-pole model. Transmitters are scattered uniformly, and each receiver is located at a random or fixed distance from its transmitter with uniform angular orientation.

## Scheduling Algorithms

The code compares three random scheduling algorithms:

1. **Fixed Aloha**: Each pair is independently active with fixed probability p. Active pairs form a binomial point process.

2. **Adaptive Aloha**: Each pair is independently active with probability p_i (varying per pair). Generalizes fixed Aloha.

3. **Determinantal**: Each pair is active with probability K_ii (the i-th diagonal of kernel matrix K, where K=L/(I+L)). Active pairs form a determinantal point process. Generalizes adaptive Aloha.

The optimization maximizes either total throughput (coverage probability) or total fairness (log of throughput, yielding proportional fairness).

## Files

| File | Description |
|------|-------------|
| `FairPairsCompareDetAloha.py` | Main script comparing all three scheduling methods |
| `ProbCovPairsDet.py` | Coverage probability analysis for pairs |
| `ProbCovPairsDetExactVsSamp.py` | Compares exact vs simulated coverage probabilities |
| `ProbCovPairsDetFunctionTest.py` | Unit tests for coverage calculation functions |
| `funFairMeanDet.py` | Calculates average fairness across configurations |
| `funProbCovPairsDetExact.py` | Exact coverage probability calculation |
| `funProbCovPairsDetSamp.py` | Simulated coverage probability calculation |
| `funPairsL.py` | Constructs L-ensemble kernel matrix |
| `funS.py` | Generates similarity matrices (Gaussian, Cauchy, Bessel kernels) |
| `funSimSimpleDPP.py` | Simulates/samples a DPP |
| `funLtoK.py` | Converts L kernel to K kernel |
| `funPalmK.py` | Calculates Palm distribution kernel |

## Requirements

- NumPy
- SciPy
- Matplotlib

## Quick Start

```bash
# Run the main comparison of scheduling algorithms
python FairPairsCompareDetAloha.py

# Or validate exact vs simulated calculations
python ProbCovPairsDetExactVsSamp.py
```

## References

1. B. Błaszczyszyn and H.P. Keeler, "Adaptive determinantal scheduling with fairness in wireless networks," 2025.

2. B. Błaszczyszyn, A. Brochard, and H.P. Keeler, "Coverage probability in wireless networks with determinantal scheduling," 2020.

3. F. Kelly, A. Maulloo, and D. Tan, "Rate control for communication networks: shadow prices, proportional fairness and stability," 1998.

See also: [detfairness_matlab](../detfairness_matlab) for the MATLAB implementation.
