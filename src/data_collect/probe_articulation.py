"""打印一个关节物体 USD 的关节 / dof / 包围盒信息(用来给 heat_X_microwave 的专家补几何)。

用法::

    python src/data_collect/probe_articulation.py \
        --usd Assets/Object/RoboDojo/Articulation/SimReady_microwave/00000/object.usd
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main():
    stage = Usd.Stage.Open(args_cli.usd)
    root = stage.GetDefaultPrim()
    print(f"[probe] {args_cli.usd}")
    print(f"[probe] default prim {root.GetPath()}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
    if not rng.IsEmpty():
        lo, hi = rng.GetMin(), rng.GetMax()
        print(f"[probe] world bbox min {tuple(round(v, 4) for v in lo)} max {tuple(round(v, 4) for v in hi)}")
        print(f"[probe] extents {tuple(round(hi[i] - lo[i], 4) for i in range(3))}")
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            print(
                f"[probe] revolute {prim.GetName()} axis={joint.GetAxisAttr().Get()} "
                f"body0={[str(t) for t in joint.GetBody0Rel().GetTargets()]} "
                f"body1={[str(t) for t in joint.GetBody1Rel().GetTargets()]} "
                f"lower={prim.GetAttribute('physics:lowerLimit').Get()} "
                f"upper={prim.GetAttribute('physics:upperLimit').Get()}"
            )
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            print(f"[probe] prismatic {prim.GetName()}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            sub = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if sub.IsEmpty():
                continue
            lo, hi = sub.GetMin(), sub.GetMax()
            print(
                f"[probe] body {prim.GetPath()} "
                f"min {tuple(round(v, 3) for v in lo)} max {tuple(round(v, 3) for v in hi)}"
            )
    simulation_app.close()


if __name__ == "__main__":
    main()
