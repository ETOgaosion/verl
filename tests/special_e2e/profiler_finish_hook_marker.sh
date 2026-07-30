#!/usr/bin/env bash
# Finish-hook command for the profiler e2e tests: drops a marker file into the profiler
# save_path so the calling script can assert that workers really executed the hook.
set -euo pipefail

: "${VERL_PROFILE_SAVE_PATH:?must be exported by the profiler finish hook}"

mkdir -p "$VERL_PROFILE_SAVE_PATH"
MARKER="$VERL_PROFILE_SAVE_PATH/finish_hook_ran_${VERL_PROFILE_ROLE:-unknown}_rank${VERL_PROFILE_RANK:-unknown}"
touch "$MARKER"
echo "profiler finish hook marker: $MARKER"
