"""Shared visual identity for code-generated figures (matches lecturenotes.sty)."""

PALETTE = {
    "lnblue": "#1F5FAD",
    "lnteal": "#0F7B6C",
    "lnorange": "#C8642A",
    "lnred": "#B3261E",
    "lnpurple": "#6B4FA0",
    "lngold": "#A77B0A",
    "lngray": "#5F6368",
    "lnbg": "#F5F7FA",
}

CYCLE = [PALETTE[k] for k in ("lnblue", "lnorange", "lnteal", "lnpurple", "lnred", "lngold", "lngray")]

# rcParams applied before user code runs. Serif + Computer Modern math makes the
# figures typographically consistent with the LaTeX body text.
RCPARAMS = {
    "figure.figsize": [6.0, 3.6],
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "legend.frameon": False,
    "image.cmap": "viridis",
    "axes.prop_cycle_colors": CYCLE,  # applied manually (cycler needs an import)
}

TEXTWIDTH_PT = 465.0  # A4 (595pt) minus 2 x 2.3cm margins: \linewidth in notes.tex.j2
STYLE_VERSION = "2"  # 2: TikZ text-overlap check (invalidates cached figure verdicts)
