# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#!/usr/bin/env bash
set -euo pipefail

# Move to this script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment for Isaac Sim/Isaac Lab
export OMNI_KIT_ACCEPT_EULA=1
export OMNI_KIT_FAST_SHUTDOWN=1

# Allow user override; otherwise try a few common locations
if [[ -z "${ISAACLAB_APPS_DIR:-}" ]]; then
  CANDIDATES=(
    "$SCRIPT_DIR/../IsaacLab/apps"
    "$SCRIPT_DIR/../../IsaacLab/apps"
    "$HOME/IsaacLab/apps"
  )
  for cand in "${CANDIDATES[@]}"; do
    if [[ -d "$cand" ]]; then
      export ISAACLAB_APPS_DIR="$cand"
      break
    fi
  done
  if [[ -z "${ISAACLAB_APPS_DIR:-}" ]]; then
    echo "[WARN] ISAACLAB_APPS_DIR is not set and could not be auto-detected. If AppLauncher needs it, set it before running:"
    echo "       export ISAACLAB_APPS_DIR=/path/to/IsaacLab/apps"
  fi
fi

# Pick Python executable
if [[ -n "${ISAACSIM_PYTHON_EXE:-}" ]]; then
  PYEXE="$ISAACSIM_PYTHON_EXE"
elif command -v python3 >/dev/null 2>&1; then
  PYEXE="python3"
elif command -v python >/dev/null 2>&1; then
  PYEXE="python"
else
  echo "[ERROR] Neither python3 nor python found, and ISAACSIM_PYTHON_EXE not set."
  exit 1
fi

# Args: default like .bat when none provided
if [[ "$#" -eq 0 ]]; then
  ARGS=(--robot go2 --robot_amount 1 --app python)
else
  ARGS=("$@")
fi

echo "[INFO] Python: $PYEXE"
echo "[INFO] Args: ${ARGS[*]}"

# Run
"$PYEXE" main.py "${ARGS[@]}"
exit $?