
Chladni-Cosmology Toy — Quick Instructions
==========================================

1) Save this file somewhere (it's already in your workspace if you downloaded it):
   - chladni_cosmo_toy.py

2) Minimal run (creates PNG frames in ./figs and outputs in ./outputs):
   python3 chladni_cosmo_toy.py

3) Try a "mode shift" (frequency switch halfway through):
   python3 chladni_cosmo_toy.py --N 192 --steps 6000 --save-every 300 \
       --freq1 1.6 --freq2 2.2 --switch-step 3000 --gamma 0.02 --c 1.0 --A 0.8

4) Look at:
   - figs/pattern_*.png (standing-wave-like patterns)
   - outputs/dominant_spacing.csv (estimated spacing over time)
   - outputs/spacing_vs_time.png (plot of spacing evolution)

Tip: Larger N and more steps give prettier patterns but take longer.
