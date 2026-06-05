source_patch: real val patch data/processed/sm_bfo_com[0]
corruptions (reviewer_study.apply_corruption): Gaussian sigma=1.0, Poisson peak=10.0, Mixed severity=1.0 (compound chain)
display: shared grayscale window from 1–99% range across all four panels (+ padding)
Regenerate: python -m gan_seg.export_smbfo_corruption_panels --out-dir paper_figures
