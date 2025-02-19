# Modif_0

## Additions

- `utils_modif_0.py` with a modified `retrieve_examples` function.
- Needs `from .utils_modif_0 import *` at the top of `run.py`.

## Modification

Instead of selecting the heads maximizing :
$$
s_{l,m}(x_i, c) = \frac{\mathbf{h}_l^m(x_i^T) \cdot \mu_c^{l,m}}{\|\mathbf{h}_l^m(x_i^T)\| \|\mu_c^{l,m}\|}
$$
We select the heads maximizing
$$
\hat{s}_{l,m}(x_i, c) = s_{l,m}(x_i, c) - \frac{1}{|C| - 1}\displaystyle\sum_{\hat{c} \neq c} s_{l,m}(x_i, \hat{c})
$$
What this does is that in enforces head to not only have precise centroids for each class, but to also have distant (angle-wise) centroids.



Note that this only has to be done when selecting the heads, after then we can use the regular $s_{l,m}(x_i, c)$ to classify each test sample $x_i$.
