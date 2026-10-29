from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutLaundryDetergentSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "laundry_detergent"
    INSTRUCTION = "Put the laundry detergent on the shelf next to the crackers."


class put_laundry_detergent_supermarket_shelf(PutLaundryDetergentSupermarketShelfCommon, TaskEnv):
    pass
