# GPT SNN SLAM 项目报告

## 1. 项目摘要

本项目实现了一条从事件相机数据到位姿、深度和局部地图的端到端全 SNN 基线。网络的学习特征提取、位姿头和深度头均由 LIF 神经元、脉冲卷积/线性突触和 surrogate gradient 组成；SE(3) 积分、深度反投影和地图融合是与机器人几何接口相连的可微读出部分。

当前仓库适合作为研究原型和后续数据集适配的骨架。它已经具备可导的前向图、完整的流式状态管理、训练损失、合成 smoke 数据和 Git 提交历史，但尚未声称在某个公开事件 SLAM 数据集上达到可发表的精度。真实数据训练、标定参数、回环检测和系统级评测属于下一阶段工作。

当前主线状态：

| 项目 | 状态 |
| --- | --- |
| 端到端事件体素到 SLAM 输出 | 已完成 |
| 全 SNN 隐藏计算与 surrogate gradient | 已完成 |
| SE(3) 位姿积分 | 已完成 |
| 可微深度射线反投影地图 | 已完成基线 |
| 流式推理与膜电位状态 | 已完成 |
| 真实事件数据集适配 | 待实现 |
| TUM RGB-D 连续帧伪事件适配 | 已完成开发验证 |
| 回环检测与全局图优化 | 待实现 |
| 公共数据集指标与性能报告 | 已加入基础指标接口 |

## 2. 系统目标和边界

系统目标是让事件相机时间窗直接产生连续位姿和空间结构预测，并保留脉冲稀疏性和时间状态。输入是一组按时间分箱的事件体素；输出包括相邻时间步的 6-DoF twist、从初始坐标系积分得到的齐次位姿、度量深度和局部 XY 占据图。

当前版本聚焦局部、前端 SLAM。它不包含 ROS 节点、传感器驱动、回环闭合、全局位姿图优化、IMU 融合或完整的多相机外参管理。这些边界使核心网络可以先在纯 PyTorch 和合成数据上稳定验证，再逐项接入系统组件。

## 3. 数据流和模型结构

```mermaid
flowchart LR
    E[事件 (x,y,t,p)] --> V[时间体素编码]
    V --> B[SpikingBackbone\nConv + GroupNorm + LIF]
    B --> P[Pose head\nSpikingLinear + LIF]
    B --> D[Depth head\nSpikingConv2d + LIF]
    P --> X[6D twist decoder]
    X --> S[SE(3) exponential map\npose integration]
    D --> Z[metric depth decoder]
    S --> M[Depth ray projection\nDifferentiable map write]
    Z --> M
    S --> O[poses / twists]
    Z --> O2[depth]
    M --> O3[occupancy]
```

### 3.1 事件编码

`events_to_voxel_grid` 接收 `[N,4]` 的 `(x, y, timestamp, polarity)` 事件，将时间划分为 `bins` 个区间，输出 `[T,2,H,W]` 的正负极性体素。函数支持空窗口、边界裁剪、显式时间范围和按时间窗归一化。批处理时在外层增加 batch 维即可得到模型需要的 `[B,T,C,H,W]`。

### 3.2 全 SNN 前端

空间特征提取器由三层 `SpikingConv2d` 组成，步幅为 `2, 2, 1`。每一层包含卷积、GroupNorm 和 LIF；GroupNorm 避免小 batch 和在线单帧推理时 BatchNorm 统计量不稳定。位姿分支使用两个 `SpikingLinear`，深度分支使用两个 `SpikingConv2d`。隐藏层没有 ReLU 或普通 MLP 激活。

LIF 更新采用显式状态：

```text
v_t = beta * v_(t-1) + I_t
s_t = 1[v_t >= threshold]
v_next = v_t - threshold * s_t
```

前向是硬脉冲，反向使用 fast-sigmoid surrogate derivative，因此可以通过标准 PyTorch 优化器训练。所有状态都由调用方或模型的 streaming 状态字典持有，避免跨 batch 意外保留计算图。

### 3.3 位姿表示和积分

位姿头输出 `[tx, ty, tz, rx, ry, rz]`，前三项为平移 twist，后三项为 axis-angle 旋转。`se3_exp` 使用 Rodrigues 公式和稳定的零角度 Taylor 展开，`integrate_twists` 按时间累积齐次变换：

```text
T_t = T_(t-1) @ exp_se3(xi_t)
```

位姿损失由平移 L1 和在单位姿态处梯度有限的旋转误差组成。旋转损失避免直接在 `acos(1)` 处反向传播导致的无穷梯度。

### 3.4 深度和局部地图

深度头的脉冲输出通过 `[depth_min, depth_max]` 范围的线性读出得到度量深度。地图写入时将深度图下采样到最多 `24 x 24` 个点，根据归一化焦距和当前 SE(3) 位姿进行相机射线反投影，再用四邻域双线性权重写入局部 XY 栅格。地图更新为带衰减的可微融合：

```text
map_t = clamp(decay * map_(t-1) + (1 - decay) * projected_write, 0, 1)
```

默认归一化焦距为 `(1,1)`，真实数据应根据相机标定传入 `focal_length_normalized=(fx, fy)`。地图写入仍然保留从深度和位姿到地图损失的梯度链路。

### 3.5 TUM RGB-D 开发适配

`TUMRGBDWindowDataset` 读取 TUM Freiburg 目录中的 `rgb.txt`、`depth.txt`、`groundtruth.txt` 和 RGB/深度 PNG。对于每个相邻 RGB 帧，适配器计算对数亮度差并生成正、负两个伪事件通道；深度按 TUM 的 `5000` 缩放因子转换为米，ground truth 位姿转换为相对于窗口首帧的齐次矩阵。默认只保留同时有深度和位姿的帧窗口。

这条路径的目的是真实图像运动、深度监督、位姿损失和地图写入的开发闭环。它不模拟事件相机的真实异步时间戳、事件噪声、阈值漂移和运动模糊，正式结果仍需在事件数据集上复测。

## 4. 代码组织

| 文件 | 作用 | 维护要点 |
| --- | --- | --- |
| `snn_slam/snn.py` | surrogate spike、LIF、脉冲卷积和线性层 | 修改神经元动力学时同步更新零点梯度测试 |
| `snn_slam/encoding.py` | 原始事件到体素张量 | 保持 `(x,y,t,p)` 顺序和 `[T,C,H,W]` 约定 |
| `snn_slam/se3.py` | SO(3)/SE(3) 指数映射、积分和位姿损失 | 所有小角度公式必须检查有限梯度 |
| `snn_slam/models.py` | 主模型、流式状态和地图 | 新增输出时同步更新 `SLAMOutput` 和 README |
| `snn_slam/losses.py` | 监督/弱监督损失 | 支持缺失标签，避免把无标签项变成常数错误 |
| `snn_slam/training.py` | `train_step`、评估、合成 batch | 数据集接入后保留 `TrainBatch` 字段契约 |
| `snn_slam/datasets.py` | TUM RGB-D 到伪事件/监督窗口 | 不把 TUM 特定逻辑扩散到模型代码 |
| `snn_slam/metrics.py` | ATE、RPE、深度 AbsRel/RMSE | 指标输入保持 `[B,T,...]` 或 `[T,...]` |
| `snn_slam/__main__.py` | 无数据集 smoke CLI | 作为安装后的最小回归入口 |
| `tests/test_smoke.py` | 形状、反向和边界回归 | 每次改动先运行这些快速测试 |

## 5. 公共接口和张量契约

```python
from snn_slam import AllSNN_SLAM, events_to_voxel_grid, slam_loss

voxels = events_to_voxel_grid(events, bins=8, height=180, width=240)
model = AllSNN_SLAM(
    in_channels=2,
    map_size=96,
    focal_length_normalized=(1.0, 1.0),
)
output = model(voxels.unsqueeze(0))
```

`SLAMOutput` 字段如下：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `twists` | `[B,T,6]` | 每个时间步的平移/旋转 twist |
| `poses` | `[B,T,4,4]` | 从序列起点积分得到的世界位姿 |
| `depth` | `[B,T,1,H,W]` | 度量深度读出 |
| `occupancy` | `[B,T,1,M,M]` | 局部占据置信度 |
| `spikes` | `[B,T,C',H',W']` | 最后一级空间脉冲，供稀疏率/能耗分析 |

流式调用会保存膜电位、当前位姿和地图：

```python
model.reset_state()
for frame in stream:             # [B,C,H,W]
    output = model.step(frame)
model.detach_state()             # 截断 BPTT 时使用
```

切换 batch 大小、设备或数据类型时模型会自动丢弃不兼容的流式状态；开始新序列时仍建议显式调用 `reset_state()`。

## 6. 训练和评估计划

当前损失由五部分组成：有标签时的位姿几何损失、深度 L1、地图 BCE，以及无论是否有标签都可使用的时间平滑和脉冲稀疏项。推荐的训练阶段如下：

1. 用合成移动边缘运行 smoke 训练，确认损失可反传和状态可重置。
2. 接入事件数据集，把每个样本转换为固定时间窗体素；先冻结地图损失，只训练位姿和深度。
3. 加入相机内参和真实深度/占据标签，逐步提高地图损失权重。
4. 使用截断 BPTT 训练长序列，定期 `detach_state()`，避免计算图跨越整个序列。
5. 在验证集报告 ATE、RPE、深度 AbsRel/RMSE、地图 IoU 和平均脉冲率。

训练示例：

```python
from torch.optim import AdamW
from snn_slam import AllSNN_SLAM
from snn_slam.training import TrainBatch, train_step

model = AllSNN_SLAM()
optimizer = AdamW(model.parameters(), lr=2e-4)
metrics = train_step(model, TrainBatch(events, poses, depth, occupancy), optimizer)
print(metrics.as_dict())
```

## 7. 验证记录

当前已验证的回归项：

- 空事件和边界事件体素编码；
- `[B,T,C,H,W]` 前向输出形状；
- 深度监督损失和端到端反向传播；
- 单帧 streaming `step` 以及 batch 大小切换时的状态重置；
- `so3_exp`/`se3_exp` 在零旋转处的有限梯度；
- 合成 smoke CLI 的单步训练。

本机验证命令：

```bash
python -m compileall -q snn_slam tests
python -c "from tests.test_smoke import *; test_event_encoder_handles_empty_and_boundaries(); test_all_snn_forward_and_gradient(); test_streaming_step_matches_interface(); test_se3_exp_is_finite_at_identity(); print('all smoke tests passed')"
python -m snn_slam --steps 4 --height 32 --width 32
python scripts/run_tum_sequence.py --root F:/BaiduNetdiskDownload/rgbd_dataset_freiburg1_desk --window 8 --height 64 --width 96 --max-frames 64 --eval-windows 4 --train-steps 2
```

如果环境安装了 pytest，也可以直接运行 `pytest -q`。当前仓库只把 PyTorch 列为运行依赖，没有强制引入 pytest。

### 7.1 本地 TUM 实际数据 smoke 记录

已在 `F:/BaiduNetdiskDownload/rgbd_dataset_freiburg1_desk` 上完成一次真实图片序列闭环验证：窗口长度 3、缩放到 `24x32`、保留 8 个有效 RGB-D/ground-truth 帧、模型 `feature_channels=32`、地图大小 32，并执行 1 次 AdamW 更新。一次运行得到：

| 指标 | 数值 |
| --- | ---: |
| train total | 4.93385 |
| train pose | 0.07558 |
| train depth | 4.85803 |
| ATE RMSE | 0.094499 m |
| RPE translation | 0.074266 m |
| depth AbsRel | 4.321810 |
| depth RMSE | 7.527359 m |

这些数字只证明数据读取、伪事件编码、监督损失、反向传播和指标计算可以串通；训练步数极少，不能作为模型精度结论。正式实验需要固定窗口划分、训练/验证序列、随机种子和多轮优化。

## 8. 已知限制和风险

- 没有内置公开数据集下载、解码和标定文件解析；数据集适配应放在独立模块，不要把数据集逻辑写进 `models.py`。
- 当前地图是局部 XY 栅格，不包含高度层、射线遮挡建模、动态物体剔除或回环闭合。
- 位姿和深度读出是显式模拟量，不能直接等同于神经形态硬件上的整数脉冲解码；部署到硬件时需要单独定义量化和时间窗口。
- 默认参数适合 smoke test，不代表真实传感器的最佳阈值、衰减率、地图范围或深度范围。
- 真实数据上的性能、能耗和精度尚未测量，任何论文或产品结论都必须在固定数据划分和可复现实验脚本上重新验证。

## 9. 后续路线图

### P0：数据闭环

- 将 TUM 适配器抽象为统一事件数据集接口，并增加 `DataLoader` 示例；
- 支持相机内参、主点、时间戳单位和事件极性约定；
- 增加地图 IoU、轨迹对齐和可复现实验汇总脚本。

### P1：SLAM 前端增强

- 采用完整相机模型和遮挡/可见性处理；
- 增加 IMU 融合分支；
- 设计关键帧缓存和局部地图裁剪策略；
- 在长序列上验证截断 BPTT 与状态恢复。

### P2：全局一致性和部署

- 脉冲描述子和回环候选检索；
- 可微或可插拔的位姿图优化；
- ONNX/硬件后端前的权重量化、时间步裁剪和稀疏率 profiling；
- 在固定硬件、固定数据集和固定随机种子下发布性能表。

## 10. 维护约定

- 每个行为变更都应增加至少一个形状、梯度或边界回归测试。
- 改动输入/输出契约时，同时更新 `README.md`、本报告和 `SLAMOutput` 文档字符串。
- 先运行快速 smoke，再运行数据集级训练；不要用未验证的长序列训练替代单元回归。
- 提交信息使用动词开头并说明范围，例如 `Add ...`、`Fix ...`、`Document ...`。
- 推送前确认 `git status` 干净、测试通过，并在提交说明中记录运行环境和主要指标。

