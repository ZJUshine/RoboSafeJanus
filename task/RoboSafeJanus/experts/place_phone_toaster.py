"""place_phone_toaster 的脚本专家:同 ``place_bread_toaster``,手机代替面包竖插在面包架里。"""

from task.RoboDojo.experts import place_bread_toaster as toast

# 手机的碰撞体(1.41 cm)比包围盒(外观,1.15 cm)厚:合到包围盒厚度就已经压进 2.6 mm,再多压时手指陷进碰撞体,
# 抬起时手指沿手机滑上去;手中手机的碰撞球也按碰撞体的半厚度(7.1 mm)取
PHONE = toast.Spec("phone", squeeze=0.0, held_r=0.0072)


def play_once(env):
    toast.play_once(env, PHONE)
