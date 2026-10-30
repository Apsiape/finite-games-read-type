Finite games that read the type of a von Neumann algebra
Seth Douglas and Nidhal Mghirbi

Version 1.0.0, October 2026. A preprint; it has not been peer reviewed.

We construct finite two-player games with binary scoring whose normal strategies distinguish operator-algebraic type. A 243-pair game has perfect play in type III and none when either ambient algebra is semifinite. For factors with separable predual, hosting it is equivalent to an invariant normal state on the flow of weights, or to an expected hyperfinite type III1 subfactor. Prepared-state hosting is therefore insensitive to the ratio invariant throughout the injective flow-finite class. A different quantifier recovers that invariant: every faithful normal state has value one for the flagged magic-square game exactly when 1/2 belongs to the Connes invariant; a nine-question conference-basis game gives the corresponding criterion for 1/3. Together these two response properties characterize type III1. One prepared state already wins both games in dimension 25. We prove that every positive rational is forced as a modular ratio by some finite game, and identify the exact rational orthogonal-pair single-bridge parameters. For the type-III-only game, a robust sorted spectral law holds for every normal semifinite strategy. It yields matching entropy, deficit, dimension and variance exponents, including an exponential inverse-square-root dimension law. No validated numerical constant is given; the sharp constant is conjectural. We give explicit finite templates, retain their boundary penalties, and distinguish sorted trace position from log density. The specified normal semifinite perturbative gravitational models cannot attain perfect play, whereas suitable type III quantum-field-theory hosts can. This conditional algebraic consequence gives no preparation or energy bound.

The paper source is main.tex. The standalone submission source is
arxiv/. Sequential finite checks, recorded outputs, game tables and LF
SHA-256 checksums are in anc/.

Build outside the source tree:
    python build.py --output-dir BUILD_DIRECTORY
Audit source and metadata:
    python check_artifacts.py
Verify the pristine ancillary files:
    python anc/verify_hashes.py

The runner requires Python, NumPy, SciPy, SymPy and psutil. It pins one CPU,
limits numerical libraries and the linear-programming solver to one thread,
and enforces a one-billion-byte working-set ceiling. Copy the complete anc/
directory before rerunning its run_checks.py: checks write beside themselves,
and timing/memory outputs change. Keep the pristine manifest for verification.

The strict builder runs pdflatex, BibTeX, and enough subsequent passes to
settle references. It rejects final warnings, undefined references, missing
glyphs and overfull or underfull boxes.

The universal-state theorem concerns every faithful normal state. Perfect scores for the two ratio games in one
prepared state do not certify type III1. Spectral prices are abstract;
no physical energy, preparation or numerical cosmological floor follows.
Finite checks do not prove operator-algebraic structure theorems.

Paper and documentation: CC BY 4.0. Scripts: MIT or Apache 2.0, at your option.
See RIGHTS.md and the license files.
