pair: Fe 2019_06_26_s1726_Fe1BCNf_rebinned_HAADF-2613_movie_40_int5_labeled (pair-index=0)
extracted: /home/danial/CV/project/external_stem_data/jacs_single_atom_TEM/extracted
mask_sigma: 2.5
zero_shot_ckpt: /home/danial/CV/project/gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt
few_shot_ckpt: /home/danial/CV/project/gan_seg/checkpoints_jacs_fewshot/n5_seed42/hybrid-nogan/gan_seg_best.pt (n_shot=5, seed=42)
Few-shot panel filename: jacs_hybrid_fewshot_k5.png
Regenerate: python -m gan_seg.export_paper_fig10_jacs_examples --out-dir paper_figures
