#!/usr/bin/env bash
# User-space LaTeX (no root needed): TinyTeX + every package the house style, the TikZ
# renderer, Russian output and teacher-made Beamer decks need. Afterwards:
#   lecturenotes doctor --refresh
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ ! -x "$TINYTEX_DIR/.TinyTeX/bin/x86_64-linux/tlmgr" ]; then
  # relocatable TinyTeX bundle (no install script piped into a shell)
  TT_VER=${TT_VER:-v2026.11}  # pinned release; the full bundle already contains most packages
  curl -fL -o "$TMPDIR/TinyTeX.tar.xz" \
    "https://github.com/rstudio/tinytex-releases/releases/download/$TT_VER/TinyTeX-linux-x86_64-$TT_VER.tar.xz"
  tar -xJf "$TMPDIR/TinyTeX.tar.xz" -C "$TINYTEX_DIR"
  rm -f "$TMPDIR/TinyTeX.tar.xz"
fi
test -x "$TINYTEX_DIR/.TinyTeX/bin/x86_64-linux/tlmgr" || { echo "TinyTeX install failed"; exit 1; }
tlmgr option repository ctan >/dev/null || true
tlmgr update --self || true
tlmgr install \
  amsmath amsfonts mathtools tools xcolor graphics graphics-def booktabs listings geometry hyperref url \
  pgf pgfplots tcolorbox tikzfill pdfcol environ trimspaces etoolbox xkeyval standalone \
  siunitx translations enumitem titlesec fancyhdr caption microtype lm iftex kvoptions ltxcmds \
  babel babel-english babel-russian cyrillic lh cm-super hyphen-russian \
  latexmk beamer translator multirow bm epstopdf-pkg mlmodern pdftexcmds infwarerr kvsetkeys \
  kvdefinekeys pdfescape hycolor letltxmacro auxhook refcount gettitlestring intcalc bitset \
  bigintcalc rerunfilecheck uniquecounter atbegshi atveryend
echo "TeX ready (PATH is set by scripts/env.sh)"
