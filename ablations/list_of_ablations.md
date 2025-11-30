# List of Ablations

- Method Comparison
    - fedavg
    - sign-sgd
    - 1bit-cs-fl

- Impact of Compression Ratio
    - [0.01, 0.05, 0.1, 0.2, 0.5]
    - Fixed: method = "1bit-cs-fl"
    - Metrics: Final Test Accuracy vs. Compression Ratio
    - Expected Outcome: There will be a trade-off. Extremely low ratios (high compression) might degrade accuracy, while higher ratios improve it but cost more bandwidth.

- Sparsity Threshold Sensitivity
    - [0.001, 0.005, 0.01, 0.05]
    - Fixed: method = "1bit-cs-fl"
    - Metrics: Test Accuracy vs. Epochs
    - Expected Outcome: A threshold that is too high will discard important gradient information, hurting convergence.

- Scalability (Number of Clients)
    - [3, 5, 10, 20]
    - Fixed: method = "1bit-cs-fl"
    - Metrics: Convergence time (or rounds to reach X% accuracy)
    - Expected Outcome: The method should remain stable as the number of clients increases.

- Communication Efficiency (Bits vs. Accuracy)
    - Plot: Test Accuracy vs. Total Bits Transmitted (Upload + Download)
    - Compare: fedavg vs 1bit-cs-fl
    - Expected Outcome: 1bit-cs-fl should reach the same accuracy much "earlier" on the x-axis (Bits Transmitted) than fedavg.