import os
import time
import csv

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM0", id="my_leader"))
follower = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM1", id="my_follower"))

leader.connect()
follower.connect()

rows = []
t_start = time.perf_counter()

try:
    while True:
        t_leader = time.perf_counter() - t_start
        leader_action = leader.get_action()          # {"shoulder_pan.pos": val, ...}

        follower.send_action(leader_action)           # normal teleop command
        
        t_follower = time.perf_counter() - t_start
        follower_obs = follower.get_observation()     # {"shoulder_pan.pos": val, ...}

        row = {"t_leader": t_leader, "t_follower": t_follower}
        for name in leader_action:
            row[f"leader__{name}"] = leader_action[name]
            row[f"follower__{name}"] = follower_obs.get(name)
        rows.append(row)

except KeyboardInterrupt:
    pass
finally:
    leader.disconnect()
    follower.disconnect()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "latency_data.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)