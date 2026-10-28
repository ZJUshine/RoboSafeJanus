from task.RoboDojo.tasks.pour_water_X import PourWaterCommon


class PourOilCommon(PourWaterCommon):
    """pour_oil_X group: pour_water_X with the oil bottle (same as pour_liquid_into_cup: the fluid's label is
    ``wine``). Each pour_oil_<target> task only sets ``TARGET`` and ``INSTRUCTION``, plus ``ONTO`` / ``ON_KWARGS``."""

    LIQUID = "wine"
    CONTAINER = "bottle"
    IN_THRESHOLD = 0.5
    ON_THRESHOLD = 0.6
    RESIDUAL = 0.15
