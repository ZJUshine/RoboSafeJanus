from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_ceramic_bowl_storage_rack_X import PlaceCeramicBowlStorageRackCommon


class PlaceCeramicBowlStorageRackCenterCommon(PlaceCeramicBowlStorageRackCommon):
    REGION = "center"
    INSTRUCTION = "Place the ceramic bowl in the center of the storage rack."


class place_ceramic_bowl_storage_rack_center(PlaceCeramicBowlStorageRackCenterCommon, TaskEnv):
    pass
