source_patch: real val data/processed/sm_bfo_com
sample_idx: 7  seed: 123
files_written: ['noise_failure_progression/noise_failure_GAN_ResNet-Initialized_sigma_progression.png', 'noise_failure_progression/noise_failure_Hybrid-NoTransformer_sigma_progression.png', 'noise_failure_progression/noise_failure_Hybrid-STEMSeg_sigma_progression.png', 'noise_failure_progression/noise_failure_Original_GAN_Scratch_sigma_progression.png', 'noise_failure_progression/noise_failure_sigma_0_all_models.png', 'noise_failure_progression/noise_failure_sigma_1_all_models.png', 'noise_failure_progression/noise_failure_sigma_1p5_all_models.png', 'noise_failure_progression/noise_failure_sigma_2_all_models.png', 'noise_failure_progression/noise_failure_sigma_2p5_all_models.png', 'noise_failure_cases_all_models_progression.png']
failure_noise_levels (shared ε, noisy = x + σ·ε): [0.0, 1.0, 1.5, 2.0, 2.5]
Separate figures directory: noise_failure_progression/
Regenerate: python -m gan_seg.export_paper_fig7_noise_examples --out-dir paper_figures --failure-noise-levels "0.0,1.0,1.5,2.0,2.5" --failure-progression-combined
