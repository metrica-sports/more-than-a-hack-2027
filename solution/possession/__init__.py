"""Frame-level ball possession from numeric tracking data only (no video).

Pipeline: load ATD -> per-frame features -> per-frame log-likelihoods for H/A/D
-> minimum-duration HMM (Viterbi) per half -> submission CSV.
"""
