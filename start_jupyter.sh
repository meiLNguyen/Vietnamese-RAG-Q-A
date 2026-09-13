#!/usr/bin/env bash
# Launch Jupyter Lab with this project's venv, free of leaked PYTHONPATH.
cd "$(dirname "$0")"
unset PYTHONPATH
exec .venv/Scripts/jupyter-lab.exe
