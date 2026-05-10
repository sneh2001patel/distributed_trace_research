# Structural Fidelity Comparison

Scores are similarities where higher is better. KS similarity is `1 - KS statistic`. Graph MMD similarity is `exp(-MMD)` over graph-level statistics.

## TT

| Generator | Class | Overall | Structure | Graph MMD Sim | Node KS Sim | Edge KS Sim | Density KS Sim | Degree KS Sim | Semantic | Valid Svc-Op | Duration KS Sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_sampling | normal | 0.280 | 0.521 | 0.561 | 0.352 | 0.385 | 0.464 | 0.659 | 0.313 | 0.096 | 0.004 |
| random_sampling | abnormal | 0.327 | 0.628 | 0.645 | 0.554 | 0.629 | 0.616 | 0.689 | 0.332 | 0.096 | 0.027 |
| random_sampling | combined | 0.319 | 0.642 | 0.624 | 0.537 | 0.592 | 0.567 | 0.822 | 0.319 | 0.096 | 0.011 |
| flat_vae | normal | 0.611 | 0.685 | 0.864 | 0.976 | 0.727 | 0.757 | 0.388 | 0.419 | 0.422 | 0.671 |
| flat_vae | abnormal | 0.575 | 0.678 | 0.816 | 0.972 | 0.736 | 0.796 | 0.395 | 0.460 | 0.369 | 0.720 |
| flat_vae | combined | 0.609 | 0.704 | 0.859 | 0.975 | 0.785 | 0.778 | 0.417 | 0.432 | 0.409 | 0.740 |
| hierarchical_vae | normal | 0.528 | 0.542 | 0.735 | 0.713 | 0.492 | 0.476 | 0.815 | 0.434 | 0.628 | 0.813 |
| hierarchical_vae | abnormal | 0.517 | 0.520 | 0.709 | 0.991 | 0.576 | 0.386 | 0.317 | 0.483 | 0.687 | 0.758 |
| hierarchical_vae | combined | 0.533 | 0.576 | 0.728 | 0.857 | 0.557 | 0.492 | 0.723 | 0.458 | 0.659 | 0.784 |

## SN

| Generator | Class | Overall | Structure | Graph MMD Sim | Node KS Sim | Edge KS Sim | Density KS Sim | Degree KS Sim | Semantic | Valid Svc-Op | Duration KS Sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_sampling | normal | 0.265 | 0.382 | 0.542 | 0.141 | 0.135 | 0.240 | 0.522 | 0.398 | 0.083 | 0.003 |
| random_sampling | abnormal | 0.358 | 0.562 | 0.738 | 0.473 | 0.499 | 0.497 | 0.711 | 0.383 | 0.084 | 0.104 |
| random_sampling | combined | 0.333 | 0.550 | 0.692 | 0.469 | 0.480 | 0.491 | 0.639 | 0.405 | 0.083 | 0.012 |
| flat_vae | normal | 0.628 | 0.650 | 0.900 | 0.963 | 0.794 | 0.649 | 0.254 | 0.426 | 0.591 | 0.694 |
| flat_vae | abnormal | 0.546 | 0.696 | 0.730 | 0.986 | 0.753 | 0.787 | 0.558 | 0.569 | 0.429 | 0.611 |
| flat_vae | combined | 0.605 | 0.680 | 0.850 | 0.975 | 0.786 | 0.773 | 0.303 | 0.450 | 0.565 | 0.691 |
| hierarchical_vae | normal | 0.696 | 0.813 | 0.882 | 0.977 | 0.863 | 0.730 | 0.956 | 0.624 | 1.000 | 0.751 |
| hierarchical_vae | abnormal | 0.711 | 0.700 | 0.835 | 0.982 | 0.808 | 0.630 | 0.497 | 0.858 | 1.000 | 0.816 |
| hierarchical_vae | combined | 0.703 | 0.819 | 0.888 | 0.982 | 0.884 | 0.694 | 0.922 | 0.665 | 1.000 | 0.761 |
