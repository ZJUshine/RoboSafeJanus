from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceSandwichRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "sandwich"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the sandwich into the refrigerator."


class place_sandwich_refrigerator(PlaceSandwichRefrigeratorCommon, TaskEnv):
    pass
