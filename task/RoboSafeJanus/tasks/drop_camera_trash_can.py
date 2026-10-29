from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.drop_X_trash_can import DropTrashCanCommon


class DropCameraTrashCanCommon(DropTrashCanCommon):
    OBJ = "camera"
    INSTRUCTION = "Drop the camera into the trash can."


class drop_camera_trash_can(DropCameraTrashCanCommon, TaskEnv):
    pass
