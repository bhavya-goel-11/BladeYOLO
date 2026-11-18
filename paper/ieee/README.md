# Memorised Blades — IEEE journal version

`main.tex` (IEEEtran, journal mode) with one file per section in `sections/`, TikZ architecture diagrams in `tikz/`,
and the bibliography in `references.bib`. Figures are read from `../figures/`.

Build (TeX Live with `texlive-latex-extra texlive-publishers texlive-science`):

    cd paper/ieee && latexmk -pdf main.tex

Regenerate the inputs first if results changed:

    .venv/bin/python tools/paper_figures.py      # fig1-6
    .venv/bin/python tools/paper_figures2.py     # fig7-28
    .venv/bin/python tools/paper_figures3.py     # fig29-33: G29 (RF-DETR-FP) and the BladeYOLO comparison
    .venv/bin/python tools/build_ieee_tables.py  # sections/gen_*.tex: per-class audit and run registries

`paper/paper.md` (pandoc draft of 8 Nov) is the earlier, pre-G29 version and is kept for traceability.
