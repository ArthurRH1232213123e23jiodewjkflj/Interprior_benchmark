# 这些脚本移到 tasks/build_task/ 了

`author_reach_traj.py` / `goal_traj_to_pointflow.py` / `build_reach_viewer_robot.py` /
`inject_robot_into_author.py` / `patch_author_reach.py` 现在只有一份,在
`bench_cube_val/tasks/build_task/`,和 `build_authored_suite.py`(npz→suite)放在一起 ——
整条流水线一个目录。

原因:`goal_traj_to_pointflow.py` 曾在 `cube_reach/pipeline/` 和 `tasks/goal_traj/`
各存一份**逐位相同**的副本。同一个脚本多份拷贝是这个项目已经付过两次代价的失败模式
(见 log/Structure.md 的 Rule 0)。

本目录保留的资产:`../flows/` `../keyframes/` `../authored/` `../viewers/`。
