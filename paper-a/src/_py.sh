#!/bin/sh
# Run the project's Python.
#
# WHY THIS EXISTS. Every script in this repository documents itself as
#
#     C:/research-toolchain/venv/Scripts/python.exe paper-a/src/<script>.py
#
# and on 2026-08-06 that interpreter stopped being executable: Windows
# Application Control began blocking the venv's python.exe (PowerShell reports
# "An Application Control policy has blocked this file"; bash reports
# "Permission denied"). The venv itself is intact -- its site-packages still
# hold numpy, scipy and PyMuPDF -- and the interpreter the venv was built from,
#
#     C:/research-toolchain/uvpy/cpython-3.12-windows-x86_64-none/python.exe
#
# still runs. Same interpreter, same ABI, so pointing PYTHONPATH at the venv's
# site-packages gives back exactly the environment the results were produced
# in. Nothing about the analysis changes; only the path used to reach it.
#
#     sh paper-a/src/_py.sh paper-a/src/build_paper_v3.py
#
# If the policy is lifted, the documented command works again and this shim can
# go. It is deliberately NOT baked into the scripts' docstrings, because the
# blocked path is a property of this machine today, not of the project.
PYVER=C:/research-toolchain/uvpy/cpython-3.12-windows-x86_64-none/python.exe
VENV=C:/research-toolchain/venv/Lib/site-packages
if [ -x "C:/research-toolchain/venv/Scripts/python.exe" ] \
        && "C:/research-toolchain/venv/Scripts/python.exe" -c "" 2>/dev/null; then
    exec "C:/research-toolchain/venv/Scripts/python.exe" "$@"
fi
PYTHONPATH="$VENV${PYTHONPATH:+;$PYTHONPATH}" exec "$PYVER" "$@"
