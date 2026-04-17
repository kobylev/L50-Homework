# PRD: Benchmarking and Competition Analysis

## 1. Objective
To empirically validate the superiority of the Conditional LSTM Filter over classical Digital Signal Processing (DSP) baselines in high-noise and phase-volatile environments.

## 2. Competitive Baselines
### 2.1 FIR Bandpass Filter (Finite Impulse Response)
- Standard Scipy implementation.
- Fixed bandwidth centered at target frequencies.

### 2.2 IIR Butterworth Filter (Infinite Impulse Response)
- High-order recursive filter for steep roll-off comparison.

## 3. Comparative Evaluation Scenarios
- **Scenario A (Stationary Noise):** Baseline performance in the current $\mathcal{U}$ noise model.
- **Scenario B (Phase Jitter):** Performance when phase noise varies dynamically over time.
- **Scenario C (Low-Context):** Efficiency when window size is restricted.

## 4. Competitive Success Metrics
- **$\Delta$ MSE:** The percentage reduction in Mean Squared Error compared to the best-performing classical filter.
- **Adaptability:** The time (samples) required for the model to establish a phase lock compared to filter settling time.
