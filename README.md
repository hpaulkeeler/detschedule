#  Adaptive determinantal scheduling with fairness in wireless networks

The MATLAB here code was used to generate the numerical results in the paper[1] by Blaszczyszyn and Keeler. The code's purpose is studying determinantal scheduling (described below) when maximizing the coverage probability, which is defined as the tail distribution of the signal-to-interference-plus-noise ratio (SINR), or a function of the coverage probability, which is called fairness.

This code generates network configurations of n transmitter-receiver pairs, which is known as the bi-pole or bi-polar network model. The code then finds the optimal access/transmitting probability for three separate random scheduling algorithms:

1) Fixed Aloha: Each transmitter-receiver pair (x_i,y_i) is independently active (so the transmitter is transmitting) with fixed probability p. The active pairs form a binomial point process.

2) Adaptive Aloha: Each transmitter-receiver pair (x_i,y_i) is independently active with probability p_i, where p_i values can vary. This scheduling scheme generalizes the fixed Aloha scheme.

3) Determinantal: Each transmitter-receiver pair (x_i,y_i) is active with probability K_ii, where K_ii is the i-th diagonal of the determinantal kernel matrix K, which is related to the L matrix by K=L/(I+L), where I is an identity matrix. The active pairs form a determinantal point process. This scheduling scheme generalizes the adaptive Aloha scheme.

When finding the optimal acccess probabilities, the code either maximizes the total throughput (defined as the SINR coverage probability of a receiver) or the total fairness (defined as the log of the throughput).

The log fairness function gives proportional fairness, but other fairness functions can be used.

This code was originally written by H.P. Keeler for the paper[1] by Blaszczyszyn and Keeler, which studies determinantal scheduling with proportional fairness in wireless networks.

If you use this code in published research, please cite paper[1].

References:

[1] Blaszczyszyn and Keeler, "Adaptive determinantal scheduling with fairness in wireless networks", 2025.

[2] Blaszczyszyn, Brochard and Keeler, "Coverage probability in wireless networks with determinantal scheduling", 2020.

Author: H. Paul Keeler, 2025.
