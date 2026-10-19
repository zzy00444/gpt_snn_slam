# GPT SNN SLAM

这是一个可训练、可流式推理的端到端全 SNN 事件相机 SLAM 基线。输入事件体素 `[B,T,C,H,W]`，一次前向同时输出：

- 相邻帧 6-DoF twist、积分后的世界坐标位姿 `[B,T,4,4]`；
- 每帧度量深度 `[B,T,1,H,W]`；
- 可反向传播的局部占据地图 `[B,T,1,M,M]`。

网络内部的特征提取、位姿头和深度头全部使用 LIF 神经元与 surrogate gradient。输出的位姿、深度是从膜电位/脉冲率得到的显式模拟量；地图更新通过 `grid_sample` 保持位姿和事件特征的梯度链路。

## 安装

```bash
pip install -e .
```

唯一的运行依赖是 PyTorch 2.0 或更高版本，不依赖 SpikingJelly、CUDA 或 ROS。

## 最小运行

```python
import torch
from snn_slam import AllSNN_SLAM, events_to_voxel_grid

# raw_events: [N,4] = (x, y, timestamp, polarity)
voxels = events_to_voxel_grid(raw_events, bins=8, height=180, width=240)
model = AllSNN_SLAM(in_channels=2, map_size=96)
result = model(voxels.unsqueeze(0))       # [B=1,T=8,C=2,H,W]
print(result.poses.shape, result.depth.shape, result.occupancy.shape)
```

对于在线处理，可以保留膜电位、位姿和地图状态：

```python
model.reset_state()
for frame in event_stream:                # [B,C,H,W]
    result = model.step(frame)
current_pose = result.poses[:, -1]
```

## 训练

有位姿/深度/地图标签时直接传入 `slam_loss`；缺少某一类标签时传 `None`，平滑项和脉冲稀疏项仍然有效：

```python
from snn_slam import slam_loss

output = model(events)
loss = slam_loss(output, target_poses, target_depth, target_occupancy)
loss.total.backward()
optimizer.step()
```

仓库自带一个无数据集依赖的闭环 smoke test：

```bash
python -m snn_slam --steps 4 --height 32 --width 32
```

真实数据接入时，数据加载器只需要把每个时间窗转换成事件体素。`events_to_voxel_grid` 接受 `(x,y,t,p)` 事件，自动处理空窗口、边界事件、正负极性和时间归一化。

## 目录

```text
snn_slam/
  encoding.py   事件体素编码
  snn.py        surrogate spike、LIF、脉冲卷积/线性层
  se3.py        可微 SO(3)/SE(3) 指数映射与位姿积分
  models.py     全 SNN 前端、位姿/深度头、可微局部地图
  losses.py     位姿、深度、地图、平滑与脉冲稀疏损失
  training.py   train_step、评估和合成 smoke 数据
```

这是一个研究基线：地图更新会用预测深度沿相机射线反投影到局部 XY 栅格，并用预测 SE(3) 位姿变换；默认使用归一化焦距 `(1, 1)`。有真实标定参数时，将其传给 `DifferentiableOccupancyMap(focal_length_normalized=(fx, fy))` 即可。
