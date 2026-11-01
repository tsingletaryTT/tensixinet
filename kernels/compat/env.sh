# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Source this to run tensixinet with the TT-Lang kernels on a tt-metal newer than the TT-Lang in the shared venv:
#
#   source kernels/compat/env.sh && python3 -m tensixinet --live --kernels chip
#
# It puts tt-orchard's matched TT-Lang and tt-metal venv first on PATH (TTLANG_PAIR_VENV names another), builds the compatibility runtime
# root once (kernels/compat/make_runtime_root.sh, in ~/.cache/tensixinet/metalroot) and points TT_METAL_RUNTIME_ROOT at it, and adds
# soundfile (the pair venv has none) from ~/.cache/tensixinet/pairsite, installed there with pip the first time. Only the shell that
# sources this is changed. Opening a chip still needs a gozer lease; `--kernels chip` takes one itself.
_compat_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_compat_pair="${TTLANG_PAIR_VENV:-$HOME/.cache/tt-orchard/ttlang-pair/venv}"
if [ ! -f "$_compat_pair/orchard-pair-exports.sh" ]; then
  echo "no TT-Lang pair venv at $_compat_pair (set TTLANG_PAIR_VENV)" >&2
  return 1 2>/dev/null || exit 1
fi
source "$_compat_pair/orchard-pair-exports.sh"
_compat_root="${TENSIXINET_RUNTIME_ROOT:-$HOME/.cache/tensixinet/metalroot}"
[ -d "$_compat_root/tt_metal" ] || "$_compat_here/make_runtime_root.sh" "$_compat_root" >&2
export TT_METAL_RUNTIME_ROOT="$_compat_root"
_compat_site="${TENSIXINET_PAIR_SITE:-$HOME/.cache/tensixinet/pairsite}"
[ -f "$_compat_site/soundfile.py" ] || python3 -m pip install --quiet --target "$_compat_site" soundfile >&2
export PYTHONPATH="$_compat_site${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$_compat_pair/bin:$PATH"
unset _compat_here _compat_pair _compat_root _compat_site
