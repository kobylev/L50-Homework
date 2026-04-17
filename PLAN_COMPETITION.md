# Plan: Baseline Competition Implementation

## Phase 1: DSP Baseline Development
- [ ] Create `code/baselines.py`.
- [ ] Implement `get_fir_filter(freq)` using `scipy.signal.firwin`.
- [ ] Implement `get_iir_filter(freq)` using `scipy.signal.butter`.

## Phase 2: Integrated Evaluation
- [ ] Modify `code/evaluate.py` to include a `compare_with_baselines` function.
- [ ] Run the same independent test set through the LSTM, the FIR, and the IIR filters.
- [ ] Aggregate MSE results into a single comparison table.

## Phase 3: Reporting
- [ ] Generate `docs/competition_comparison.png` bar chart showing MSE per method.
- [ ] Update `README.md` with the "Competition Results" section.
