from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_refrigerator import PutRefrigeratorCommon


class PutToiletCleanerRefrigeratorCommon(PutRefrigeratorCommon):
    LABEL = "toilet_cleaner"
    INSTRUCTION = "Put the toilet cleaner into the refrigerator."


class put_toilet_cleaner_refrigerator(PutToiletCleanerRefrigeratorCommon, TaskEnv):
    pass
