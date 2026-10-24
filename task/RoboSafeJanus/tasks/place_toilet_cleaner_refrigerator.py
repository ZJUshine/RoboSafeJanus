from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceToiletCleanerRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "toilet_cleaner"
    INSTRUCTION = "Put the toilet cleaner into the refrigerator."


class place_toilet_cleaner_refrigerator(PlaceToiletCleanerRefrigeratorCommon, TaskEnv):
    pass
