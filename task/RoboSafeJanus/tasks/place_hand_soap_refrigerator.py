from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_juice_refrigerator import PlaceJuiceRefrigeratorCommon


class PlaceHandSoapRefrigeratorCommon(PlaceJuiceRefrigeratorCommon):
    LABEL = "hand_soap"
    INSTRUCTION = "Put the hand soap into the refrigerator."


class place_hand_soap_refrigerator(PlaceHandSoapRefrigeratorCommon, TaskEnv):
    pass
