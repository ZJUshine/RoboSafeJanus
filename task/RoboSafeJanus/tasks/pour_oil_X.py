from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourOilCommon(PourWaterCommon):
    """pour_oil_X 组:换成油瓶的 pour_water_X(和 pour_liquid_into_cup 一样,流体的 label 是 ``wine``)。
    各 pour_oil_<目标> 任务只设置 ``TARGET`` 和 ``INSTRUCTION``,需要时再加 ``ONTO`` / ``ON_KWARGS``。"""

    LIQUID = "wine"
    CONTAINER = "bottle"
    IN_THRESHOLD = 0.5
    ON_THRESHOLD = 0.6
    RESIDUAL = 0.15
