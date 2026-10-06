"""Decision Index 0.2.1 harness for Track D (milestone m0-a).

Point estimates come from the official kit (apolinario/decision-index, pinned in
`harness.config`). This package adds slicing, paired cluster-bootstrap CIs,
debiased/smooth calibration error per length bin, and minimum-detectable-effect
planning on top of it.
"""

__version__ = "0.1.0"

import locale as _locale

# The kit reads its packaged subset lists with `read_text()` (locale encoding). On Windows
# (cp1252) two BRIGHT run ids containing U+2019 are garbled and silently dropped from the
# 0.2.1 scope. Refuse to run unless Python is in UTF-8 mode: `python -X utf8 -I -m ...`.
if _locale.getpreferredencoding(False).lower().replace("-", "") != "utf8":
    raise RuntimeError("harness needs UTF-8 mode: run `python -X utf8 -I ...` (the kit decodes its data files with the locale encoding)")
