import numpy as np
import transforms3d as t3d

from env.reward_manager.reward_manager import RewardManager


def _np(v):
    return v.detach().cpu().numpy().reshape(-1) if hasattr(v, "detach") else np.asarray(v, dtype=float).reshape(-1)


class DropTableCommon:
    """drop_X_table group: lift the object well above the table, let go so it falls back onto the table, then return.
    Each drop_<object>_table task only sets ``OBJ`` and ``INSTRUCTION``."""

    OBJ = None
    INSTRUCTION = None
    LIFT = 0.15  # the object's centre must rise this far above where it started
    REST_TOL = 0.03  # landed: bbox centre at most half the longest bbox side + this above the table top

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        # Task-specific check, looked up by name like the built-in Func_Parser checks
        self.reward_manager.func_parser.is_resting_on_table = self._resting_on_table
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _resting_on_table(self, args):
        """1.0 when the object lies on the table top (not held up, not fallen to the floor), in any orientation."""
        lm = self.scene_manager.layout_manager
        env_idx = args["env_idx"]
        pos, rot = lm.get_instance_pose(env_idx, label=args["label"])
        bbox = lm.get_instance_bbox_vertices(inst_name=lm.get_instance_name(env_idx, args["label"]), env_idx=env_idx)
        if pos is None or bbox is None:
            return 0.0
        bbox = np.asarray(bbox, dtype=float).reshape(-1, 3)
        center = _np(pos)[:3] + t3d.quaternions.quat2mat(_np(rot)) @ bbox.mean(axis=0)
        table_z = float(lm.table_info[env_idx].get("height", 0.765))
        return float(table_z <= center[2] <= table_z + np.ptp(bbox, axis=0).max() / 2 + self.REST_TOL)

    def run_reward(self):
        # Two stages in order: first the object is lifted, later it rests on the table again with the gripper open
        # and the arm back (at the same step, so an object still held in the gripper does not count)
        rm = self.reward_manager
        rm.check([rm.is_lift(label=self.OBJ, z_threshold=self.LIFT)])
        rm.check(
            [
                ("is_resting_on_table", {"label": self.OBJ}),
                rm.is_all_gripper_open(),
                rm.all_robot_back_to_origin(),
            ]
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
