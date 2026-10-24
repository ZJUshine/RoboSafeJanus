from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager

# ArtVIP refrigerator 00001: the lower (fridge compartment) door, hinged on the right edge, opens toward its upper limit
DOOR_JOINT = "RevoluteJoint_fridge_2_down"


class PlaceJuiceRefrigeratorCommon:
    """Put an object onto the bottom shelf of the open refrigerator. The other place_<object>_refrigerator tasks reuse
    this class and only override ``LABEL`` (the object's YAML label), ``UPRIGHT`` and ``INSTRUCTION``."""

    LABEL = "juice"
    FRIDGE_LABEL = "refrigerator"
    # Objects that stand upright on the shelf must still be standing at the end (the carton must not be knocked over)
    UPRIGHT = True
    UPRIGHT_THRESHOLD = 30
    INSTRUCTION = "Put the juice into the refrigerator."

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
        """The episode starts with the lower door fully open (same as heat_egg_microwave: the opened state is written
        into the saved joint positions, so the ``apply_saved_poses`` after ``reset`` keeps it open)."""
        lm = self.scene_manager.layout_manager
        for env_idx in range(self.num_envs):
            fridge = lm.get_scene_object(env_idx, lm.get_instance_name(env_idx, self.FRIDGE_LABEL))
            idx = fridge.dof_names.index(DOOR_JOINT)
            lo, hi = float(fridge.lower_joint_positions[idx]), float(fridge.upper_joint_positions[idx])
            q = fridge.initial_joint_positions.clone()
            q[idx] = hi if abs(hi) > abs(lo) else lo
            fridge.initial_joint_positions = q
            fridge.set_current_joint_positions(q)

    def _checks(self):
        rm = self.reward_manager
        checks = [
            rm.is_A_root_point_in_B_bbox(label_A=self.LABEL, label_B=self.FRIDGE_LABEL),
            rm.is_all_gripper_open(),
            rm.all_robot_back_to_origin(),
            rm.is_not_moved(label=self.FRIDGE_LABEL, dis_threshold=0.02),
        ]
        if self.UPRIGHT:
            checks.append(rm.is_axis_up(label=self.LABEL, axis=[0, 0, 1], threshold=self.UPRIGHT_THRESHOLD))
        return checks

    def run_reward(self):
        # Same as heat_egg_microwave: everything must hold at the same step (the check latches), so an object still in
        # the gripper, or dragged back out while the gripper retreats, does not count
        self.reward_manager.check(self._checks())

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]


class place_juice_refrigerator(PlaceJuiceRefrigeratorCommon, TaskEnv):
    pass
