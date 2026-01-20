# detcov_matlab

Coverage probability in determinantal networks (MATLAB implementation).

## Overview

This code implements algorithms for randomly scheduling wireless network transmissions using discrete determinantal point processes (DPPs). The approach assigns medium access to repulsive subsets of potential transmitters, naturally extending spatial Aloha which schedules transmissions independently.

Under a general path loss model and Rayleigh fading, the code provides elegant analysis of coverage probabilities and transmission attempts (local delay). This is enabled by the explicit determinantal form of the conditional (Palm) distribution and closed-form expressions for the Laplace functional of determinantal processes.

## Files

| File | Description |
|------|-------------|
| `ProbCovPairsDet.m` | Coverage analysis for transmitter-receiver pairs (bi-pole model) |
| `ProbCovTXRXDet.m` | Coverage analysis where nodes are either transmitters or receivers |
| `TestPalm.m` | Validates Palm distribution calculations empirically against simulations |
| `funProbCovPairsDet.m` | Function for coverage probability (pairs model) |
| `funProbCovTXRXDet.m` | Function for coverage probability (TX-or-RX model) |
| `funSimSimpleDPP.m` | Simulates/samples a DPP using eigendecomposition |
| `funLtoK.m` | Converts L kernel matrix to normalized K kernel |
| `funPalmK.m` | Calculates Palm distribution kernel for a single point |

## Quick Start

```matlab
% Run coverage analysis for transmitter-receiver pairs
ProbCovPairsDet

% Or run the Palm distribution validation
TestPalm
```

For a standalone introduction to DPP simulation, run `DemoDetPoisson.m` in the repository root.

## Reference

B. Błaszczyszyn, A. Brochard, and H.P. Keeler, "Coverage probability in wireless networks with determinantal scheduling," 2020.

See also: [detcov_python](../detcov_python) for the Python implementation.
