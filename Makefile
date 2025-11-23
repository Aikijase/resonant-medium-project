.RECIPEPREFIX := >
.PHONY: p11-demo p11-sweep p11-plot p12-demo p12-sweep p12-plot

# existing Phase-11 kept (if you already had them, these are no-ops)
p11-demo:
> python3 tools/phase11_prism_sim.py --topology prism6 --N 6 --alpha 1.0 --steps 15000 --burn_in 2000 --tail 400

p11-sweep:
> python3 tools/phase11_prism_sweep.py --topology prism6 --N 6 --alpha 0.5 1.0 1.5 2.0 --seeds 0 1 2 --dw_patterns alt grad block

p11-plot:
> python3 tools/phase11_plot_prism.py --csv outputs/phase11/data/prism_sweep.csv --prefix outputs/phase11/plots/prism

# Phase-12
p12-demo:
> python3 tools/phase12_multichord_sim.py --M 3 --n 6 --centers 2.7 2.8 2.9 --patterns block grad alt --intra ring --bridge_mode random_k --bridges 1 --alpha 1.0

p12-sweep:
> python3 tools/phase12_sweep.py --M 3 --n 6 --centers 2.7 2.8 2.9 --patterns block grad alt --intra ring --bridge_mode random_k --bridges 0 1 2 --alpha 0.5 1.0 1.5 --noise 0.0 0.02 --seeds 0 1 2

p12-plot:
> python3 tools/phase12_plot.py --csv outputs/phase12/data/multichord_sweep.csv --prefix outputs/phase12/plots/multichord
