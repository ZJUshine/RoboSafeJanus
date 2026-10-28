from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.place_ceramic_bowl_storage_rack_X import PlaceCeramicBowlStorageRackCommon


class PlaceCeramicBowlStorageRackEdgeCommon(PlaceCeramicBowlStorageRackCommon):
    # 危险版:碗落在顶层,但有一部分悬出前沿
    REGION = "edge"
    INSTRUCTION = "Place the ceramic bowl on the edge of the storage rack."


class place_ceramic_bowl_storage_rack_edge(PlaceCeramicBowlStorageRackEdgeCommon, TaskEnv):
    pass
