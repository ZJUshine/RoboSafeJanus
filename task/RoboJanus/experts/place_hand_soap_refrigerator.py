"""place_hand_soap_refrigerator 的脚本专家:同 ``place_juice_refrigerator``,物体换成洗手液瓶,夹得紧一点(合到底)。"""

from task.RoboDojo.experts import place_juice_refrigerator as fridge

SPEC = fridge.Spec(
    "hand_soap", full_close=True, top_margin=0.05, carry_slow=3.0, obstacle_shrink=fridge.BOTTLE_SHRINK, max_yaw=30.0
)  # 瓶颈 / 泵头在顶上 5 cm;搬得快时瓶子在指间晃到 10 度,放下就倒;
# 斜着(25 度以上)夹的瓶子松手后靠在一根手指上,退出时被带倒


def play_once(env):
    fridge.play_once(env, SPEC)
