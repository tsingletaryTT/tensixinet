#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Build a private tt-metal runtime root that lets TT-Lang's generated kernels compile against a newer tt-metal.
#
#   kernels/compat/make_runtime_root.sh OUT_DIR       then:  TT_METAL_RUNTIME_ROOT=OUT_DIR python3 ...
#
# TT-Lang 1.1.x writes C++ that calls compute functions and includes a header which tt-metal has since moved or removed:
#   * tools/profiler/kernel_profiler.hpp moved to api/debug/kernel_profiler.hpp (tt-metal 27cad27903e).
#   * binary_op_init_common, add_tiles_init, sub_tiles_init and mul_tiles_init were removed as "deprecated shims" (tt-metal 65e51a17e96).
#     Their replacements are compute_kernel_hw_startup and add_init, sub_init and mul_init (binary_tiles_init underneath), which take the
#     same circular buffers.
# Without them, any kernel that multiplies two circular-buffer tiles (ttavdog, ttmix, ttcoco) fails to compile and the other four compile.
# The output is a tree of symlinks to the real tt-metal with those two headers replaced, so nothing shared is changed and only a process
# started with TT_METAL_RUNTIME_ROOT=OUT_DIR sees it. A TT-Lang that emits the new names makes this unnecessary.
set -euo pipefail
SRC="${TT_METAL_HOME:-$HOME/tt-metal}"
OUT="${1:?usage: make_runtime_root.sh OUT_DIR}"
[ -d "$SRC/tt_metal/hw/inc/api/compute" ] || { echo "no tt-metal at $SRC (set TT_METAL_HOME)" >&2; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"

# farm FROM TO: TO becomes a real directory whose entries are symlinks to FROM's entries
farm() {
  mkdir -p "$2"
  local e
  for e in "$1"/* "$1"/.[!.]*; do
    [ -e "$e" ] || continue
    ln -s "$e" "$2/$(basename "$e")"
  done
}
# open REL: make OUT/REL (and every directory above it) a real directory, so a file inside it can be replaced
open_dir() {
  local rel="$1" part path=""
  [ -z "$(ls -A "$OUT")" ] && farm "$SRC" "$OUT"
  for part in ${rel//\// }; do
    path="${path:+$path/}$part"
    if [ -L "$OUT/$path" ]; then
      local target
      target="$(readlink "$OUT/$path")"
      rm "$OUT/$path"
      farm "$target" "$OUT/$path"
    fi
  done
  mkdir -p "$OUT/$rel"  # a directory the real tree no longer has (tools/profiler) is made here
}

open_dir tools/profiler
rm -f "$OUT/tools/profiler/kernel_profiler.hpp"
cat > "$OUT/tools/profiler/kernel_profiler.hpp" <<'H'
// Compatibility: TT-Lang still includes the header at its old path, which tt-metal moved.
#pragma once
#include "api/debug/kernel_profiler.hpp"
H

COMPUTE=tt_metal/hw/inc/api/compute
open_dir "$COMPUTE"
rm -f "$OUT/$COMPUTE/eltwise_binary.h"
cat > "$OUT/$COMPUTE/eltwise_binary.h" <<H
// Compatibility: the real header, plus the names tt-metal removed and TT-Lang still emits.
#pragma once
#include "$SRC/$COMPUTE/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
namespace ckernel {
ALWI void binary_op_init_common(std::uint32_t icb0, std::uint32_t icb1, std::uint32_t ocb) { compute_kernel_hw_startup(icb0, icb1, ocb); }
ALWI void add_tiles_init(std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(icb0, icb1, acc_to_dest);
}
ALWI void sub_tiles_init(std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWSUB>(icb0, icb1, acc_to_dest);
}
ALWI void mul_tiles_init(std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(icb0, icb1, acc_to_dest);
}
}  // namespace ckernel
H
echo "runtime root written to $OUT (use it with TT_METAL_RUNTIME_ROOT=$OUT)"
