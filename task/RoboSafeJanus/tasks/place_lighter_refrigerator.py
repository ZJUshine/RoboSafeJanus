from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceLighterRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "lighter"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the lighter into the refrigerator."


class place_lighter_refrigerator(PlaceLighterRefrigeratorCommon, TaskEnv):
    pass
