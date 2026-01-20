# detcov_python

Coverage probability in determinantal networks (Python implementation).

## Overview

This code implements algorithms for randomly scheduling wireless network transmissions using discrete determinantal point processes (DPPs). The approach assigns medium access to repulsive subsets of potential transmitters, naturally extending spatial Aloha which schedules transmissions independently.

Under a general path loss model and Rayleigh fading, the code provides elegant analysis of coverage probabilities and transmission attempts (local delay). This is enabled by the explicit determinantal form of the conditional (Palm) distribution and closed-form expressions for the Laplace functional of determinantal processes.

## Files

| File | Description |
|------|-------------|
| `ProbCovPairsDet.py` | Coverage analysis for transmitter-receiver pairs (bi-pole model) |
| `ProbCovTXRXDet.py` | Coverage analysis where nodes are either transmitters or receivers |
| `TestPalm.py` | Validates Palm distribution calculations empirically against simulations |
| `funProbCovPairsDet.py` | Function for coverage probability (pairs model) |
| `funProbCovTXRXDet.py` | Function for coverage probability (TX-or-RX model) |
| `funSimSimpleDPP.py` | Simulates/samples a DPP using eigendecomposition |
| `funLtoK.py` | Converts L kernel matrix to normalized K kernel |
| `funPalmK.py` | Calculates Palm distribution kernel for a single point |

## Requirements

- NumPy
- Matplotlib

## Quick Start

```bash
# Run coverage analysis for transmitter-receiver pairs
python ProbCovPairsDet.py

# Or run the Palm distribution validation
python TestPalm.py
```

For a standalone introduction to DPP simulation, run `DemoDetPoisson.py` in the repository root.

## Reference

B. Błaszczyszyn, A. Brochard, and H.P. Keeler, "Coverage probability in wireless networks with determinantal scheduling," 2020.

See also: [detcov_matlab](../detcov_matlab) for the MATLAB implementation.
