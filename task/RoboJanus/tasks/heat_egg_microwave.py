from pxr import Usd, UsdPhysics

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager
from utils.transformer import *


def door_joint(obj):
    """Name of the revolute joint whose child link is the door (ArtVIP links are named ``E_door_<n>``)."""
    # obj.prim is the articulation root link (ext_anchor); search the whole object
    for prim in Usd.PrimRange(obj.stage.GetPrimAtPath(obj.usd_prim_path)):
        if not prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetName() not in obj.dof_names:
            continue
        body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if body1 and "door" in body1[0].name.lower():
            return prim.GetName()
    raise ValueError(f"{obj.prim_path} has no door joint")


class HeatEggMicrowaveCommon:
    """Put an object into the open microwave. The other heat_<object>_microwave tasks reuse this class and only
    override ``LABEL`` (the object's YAML label) and ``INSTRUCTION``."""

    LABEL = "egg"
    INSTRUCTION = "Put the egg in the microwave and heat it."

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.open_door()
        self.reward_manager.reset()

    def open_door(self):
        """The episode starts with the microwave door fully open. The opened state is written into the saved joint
        positions, so the ``apply_saved_poses`` that the collect / eval envs run after ``reset`` keeps it open."""
        lm = self.scene_manager.layout_manager
        for env_idx in range(self.num_envs):
            mw = lm.get_scene_object(env_idx, lm.get_instance_name(env_idx, "microwave"))
            idx = mw.dof_names.index(door_joint(mw))
            q = mw.initial_joint_positions.clone()
            q[idx] = float(mw.lower_joint_positions[idx])  # ArtVIP doors open toward the lower limit (-90 deg)
            mw.initial_joint_positions = q
            mw.set_current_joint_positions(q)

    def run_reward(self):
        # The object rests inside the microwave's box (not on top of it, not in front of it) while the gripper is
        # released, the arms are back at their start poses and the microwave has not been shoved around. All four must
        # hold at the same step: the check latches, and an object that is still in the gripper, or is dragged out while
        # the gripper backs off, must not count.
        self.reward_manager.check(
            [
                self.reward_manager.is_A_root_point_in_B_bbox(label_A=self.LABEL, label_B="microwave"),
                self.reward_manager.is_all_gripper_open(),
                self.reward_manager.all_robot_back_to_origin(),
                self.reward_manager.is_not_moved(label="microwave", dis_threshold=0.02),
            ]
        )

    def gen_instruction(self, env_idx):
        templates = [self.INSTRUCTION]
        return templates


class heat_egg_microwave(HeatEggMicrowaveCommon, TaskEnv):
    pass
