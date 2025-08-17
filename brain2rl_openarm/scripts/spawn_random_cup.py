#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import tempfile

SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial><mass>0.05</mass></inertial>
      <collision name="collision">
        <geometry><cylinder><radius>0.03</radius><length>0.10</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.03</radius><length>0.10</length></cylinder></geometry>
        <material><ambient>0 0.3 0.8 1</ambient></material>
      </visual>
    </link>
  </model>
</sdf>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--x', type=float, default=None)
    ap.add_argument('--y', type=float, default=None)
    ap.add_argument('--z', type=float, default=0.8)
    args = ap.parse_args()

    name = args.name or f"cup_{random.randint(1000,9999)}"
    x = args.x if args.x is not None else random.uniform(-0.15, 0.15)
    y = args.y if args.y is not None else random.uniform(-0.15, 0.15)
    z = args.z

    sdf = SDF_TEMPLATE.format(name=name)
    with tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False) as f:
        f.write(sdf)
        sdf_path = f.name

    cmd = [
        "ros2", "run", "gazebo_ros", "spawn_entity.py",
        "-entity", name,
        "-file", sdf_path,
        "-x", str(x), "-y", str(y), "-z", str(z)
    ]
    print("[spawn_random_cup] Executing:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(sdf_path)
        except OSError:
            pass

if __name__ == "__main__":
    main()
