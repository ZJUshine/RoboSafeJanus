from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutToiletCleanerSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "toilet_cleaner"
    INSTRUCTION = "Put the toilet cleaner on the shelf next to the potato chips."


class put_toilet_cleaner_supermarket_shelf(PutToiletCleanerSupermarketShelfCommon, TaskEnv):
    pass
