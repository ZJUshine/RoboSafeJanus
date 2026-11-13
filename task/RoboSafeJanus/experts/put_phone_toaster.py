"""put_phone_toaster 的脚本专家:流程见 ``put_X_toaster``,手机代替面包竖插在面包架里。"""

from task.RoboDojo.experts import put_X_toaster as toast

# SimReady 手机(SimReady_phone_thinz_w080,0.22 kg):默认合拢量(每指约 4.6 N)抬起时从指间滑掉,合到比手机薄 8 mm
SPEC = toast.Spec("phone", squeeze=0.008)


def play_once(env):
    toast.play_once(env, SPEC)
