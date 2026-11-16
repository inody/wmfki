# Third-party notices

The MIT License in `LICENSE` applies only to WMFKI project material for which
Daisuke Inoue is the copyright holder. It does not replace the licenses or
other rights that apply to external software and tools.

## Python numerical and plotting dependencies

The numerical workflows use NumPy, Matplotlib, scikit-learn, and pandas. These
packages are installed from their upstream distributions according to
`pyproject.toml` and `uv.lock`; their source code is not relicensed by the WMFKI
project. Each package remains subject to its own upstream license and notices.

## Externally supplied TeX plotting tools and fonts

Exact reference-figure rendering uses Matplotlib's TeX text mode with an
external LaTeX installation, including the standard `fontenc` and `times`
packages and their associated fonts. The TeX engine, packages, and fonts are
not distributed in this release and are not relicensed under the project MIT
License. Their upstream terms apply.

No conference document class or manuscript template is required or distributed
for the numerical workflows.
