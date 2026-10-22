from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceBellPepperRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "bell_pepper"
    # Grasped top-down and released from a tilted gripper, so it may land in any orientation
    UPRIGHT = False
    INSTRUCTION = "Put the bell pepper into the refrigerator."


class place_bell_pepper_refrigerator(PlaceBellPepperRefrigeratorCommon, TaskEnv):
    pass
