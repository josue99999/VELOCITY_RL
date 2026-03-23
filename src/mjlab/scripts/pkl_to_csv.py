"""Convert video2robot pkl motion files to CSV format.

Output CSV format (one row per frame):
  x, y, z, qx, qy, qz, qw, dof_0, ..., dof_N

Quaternion is xyzw (w last) — the same format expected by csv_to_npz.py.
"""

import pickle
from pathlib import Path

import numpy as np
import tyro

import mjlab


def main(
  input_file: str,
  output_file: str,
) -> None:
  """Convert a video2robot pkl motion file to CSV.

  Args:
    input_file: Path to the input .pkl file (output of video2robot).
    output_file: Path to the output .csv file.
  """
  with open(input_file, "rb") as f:
    data = pickle.load(f)

  root_pos: np.ndarray = data["root_pos"]  # (T, 3)
  root_rot: np.ndarray = data["root_rot"]  # (T, 4) xyzw
  dof_pos: np.ndarray = data["dof_pos"]  # (T, N_dof)

  motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
  np.savetxt(output_file, motion, delimiter=",", fmt="%.8f")

  fps = float(data.get("fps", 30.0))
  robot_type = data.get("robot_type", "unknown")
  n_frames, n_cols = motion.shape
  n_dof = n_cols - 7

  print(f"Robot  : {robot_type}")
  print(f"Frames : {n_frames}  FPS: {fps}")
  print(f"DOF    : {n_dof}")
  print(f"Saved  : {Path(output_file).resolve()}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
