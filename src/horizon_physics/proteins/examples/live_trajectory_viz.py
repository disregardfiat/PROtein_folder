#!/usr/bin/env python3
"""
Live 3D visualizer for the minimizer trajectory debug log (JSONL).

Usage:
  # Terminal 1: run minimizer with a trajectory log
  python -c "
  from horizon_physics.proteins import minimize_full_chain
  minimize_full_chain('MKFLNDR', simulate_ribosome_tunnel=True, trajectory_log_path='/tmp/traj.jsonl')
  "

  # Terminal 2: run visualizer (start before or after the minimizer)
  python -m horizon_physics.proteins.examples.live_trajectory_viz /tmp/traj.jsonl

Or with run_tunnel_and_grade (if you add trajectory_log_path to the call), pass that path here.

The script tails the JSONL file (one {"t": step, "positions": [[x,y,z], ...]} per line)
and updates a matplotlib 3D scatter of Cα positions; bonds are drawn as lines between
consecutive residues. Close the window or Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time


def _tail_frames(path: str, frame_queue: queue.Queue) -> None:
    """Background thread: tail path and push (step, positions) into frame_queue."""
    while not os.path.isfile(path):
        time.sleep(0.2)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                t = data.get("t", 0)
                positions = data.get("positions", [])
                frame_queue.put((t, positions))
            except (json.JSONDecodeError, TypeError):
                continue
        # After EOF, keep checking for new lines (tail -f style)
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        frame_queue.put((data.get("t", 0), data.get("positions", [])))
                    except (json.JSONDecodeError, TypeError):
                        pass
            else:
                time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live 3D visualization of minimizer trajectory log (JSONL)."
    )
    parser.add_argument(
        "trajectory_log",
        nargs="?",
        default="/tmp/protein_trajectory.jsonl",
        help="Path to JSONL trajectory file (default: /tmp/protein_trajectory.jsonl)",
    )
    parser.add_argument(
        "--bonds",
        action="store_true",
        default=True,
        help="Draw bonds between consecutive Cα (default: True)",
    )
    parser.add_argument(
        "--no-bonds",
        action="store_false",
        dest="bonds",
        help="Disable bond lines",
    )
    parser.add_argument(
        "--update-ms",
        type=float,
        default=100,
        help="Min ms between plot updates (default: 100)",
    )
    args = parser.parse_args()
    path = os.path.abspath(args.trajectory_log)

    try:
        import numpy as np
    except ImportError as e:
        print("Live visualizer requires numpy:", e, file=sys.stderr)
        return 1
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        print(
            "Live visualizer requires matplotlib with 3D support. Error:",
            e,
            file=sys.stderr,
        )
        print(
            "Try: pip install --upgrade 'matplotlib>=3.5'  (or use a venv with a single matplotlib install)",
            file=sys.stderr,
        )
        return 1

    frame_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(target=_tail_frames, args=(path, frame_queue), daemon=True)
    thread.start()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], c="C0", s=8, alpha=0.9)
    line_bonds = ax.plot([], [], [], "C1-", linewidth=0.8, alpha=0.7)[0]
    title = ax.set_title("Waiting for trajectory...")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    # Fixed aspect so structure doesn't look stretched
    ax.set_box_aspect([1, 1, 1])

    last_step = -1
    last_update = 0.0
    positions: list = []

    def update_plot() -> bool:
        nonlocal last_step, last_update, positions
        try:
            step, pos_list = frame_queue.get_nowait()
        except queue.Empty:
            return True
        if not pos_list:
            return True
        positions = np.asarray(pos_list, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        n = positions.shape[0]
        scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        if args.bonds and n >= 2:
            line_bonds.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
        else:
            line_bonds.set_data_3d([], [], [])
        title.set_text(f"Step {step}  |  {n} residues")
        if n > 0:
            margin = max(5.0, np.ptp(positions))
            mid = positions.mean(axis=0)
            ax.set_xlim(mid[0] - margin, mid[0] + margin)
            ax.set_ylim(mid[1] - margin, mid[1] + margin)
            ax.set_zlim(mid[2] - margin, mid[2] + margin)
        last_step = step
        last_update = time.time()
        return True

    def on_timer(_event=None) -> None:
        now = time.time()
        if (now - last_update) * 1000 >= args.update_ms:
            update_plot()
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(50, int(args.update_ms)))
    timer.add_callback(on_timer)
    timer.start()

    # Process any frames already in the file
    while True:
        try:
            step, pos_list = frame_queue.get(timeout=0.05)
            if pos_list:
                positions = np.asarray(pos_list, dtype=float)
                if positions.ndim == 1:
                    positions = positions.reshape(-1, 3)
                n = positions.shape[0]
                scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                if args.bonds and n >= 2:
                    line_bonds.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
                else:
                    line_bonds.set_data_3d([], [], [])
                title.set_text(f"Step {step}  |  {n} residues")
                if n > 0:
                    margin = max(5.0, np.ptp(positions))
                    mid = positions.mean(axis=0)
                    ax.set_xlim(mid[0] - margin, mid[0] + margin)
                    ax.set_ylim(mid[1] - margin, mid[1] + margin)
                    ax.set_zlim(mid[2] - margin, mid[2] + margin)
                last_step = step
                last_update = time.time()
        except queue.Empty:
            break

    print(f"Tailing {path} (start minimizer with trajectory_log_path='{path}' if not already). Close window to exit.")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
