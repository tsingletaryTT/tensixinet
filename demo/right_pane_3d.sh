#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# Right pane of the enya3d demo: tt-toplike, opened on its Die 3D view. The view has no command-line flag, only the `3` key, so this
# starts tt-toplike and then sends that key to its own tmux pane. Read-only (hybrid backend: hwmon plus tt-smi), so it needs no lease.
( sleep 5; tmux send-keys -t "$TMUX_PANE" 3 ) &
exec tt-toplike --backend hybrid
