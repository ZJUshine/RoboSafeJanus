"""put_phone_toaster 的脚本专家:流程见 ``put_X_toaster``,手机代替面包竖插在面包架里。"""

from task.RoboDojo.experts import put_X_toaster as toast

# SimReady 手机(SimReady_phone_thinz_w080,0.22 kg)
SPEC = toast.Spec("phone")


def play_once(env):
    toast.play_once(env, SPEC)
