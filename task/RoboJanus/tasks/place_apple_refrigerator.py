from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceAppleRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "apple"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the apple into the refrigerator."


class place_apple_refrigerator(PlaceAppleRefrigeratorCommon, TaskEnv):
    pass
