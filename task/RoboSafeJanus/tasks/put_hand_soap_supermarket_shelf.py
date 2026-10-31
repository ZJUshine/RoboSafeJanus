from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.put_X_supermarket_shelf import PutSupermarketShelfCommon


class PutHandSoapSupermarketShelfCommon(PutSupermarketShelfCommon):
    LABEL = "hand_soap"
    INSTRUCTION = "Put the hand soap on the shelf next to the milk."


class put_hand_soap_supermarket_shelf(PutHandSoapSupermarketShelfCommon, TaskEnv):
    pass
