# 智能搬运机器人树莓派上位机

树莓派负责视觉调试、任务码校验、九点路网规划和逐段调度；F4 下位机负责当前段运动、编码器反馈和完成/失败上报。当前命令行可执行手动路径和机械臂调试，**尚未把按钮、二维码、自动导航与取放机构串成完整自动任务**。

## 安装与部署

树莓派上的运行目录是 `/home/pi/文档/pi`。本机修改不会自动同步到树莓派，部署后应在该目录核对文件，再运行命令。先编译、烧录与当前协议匹配的 F4 固件。

```bash
cd /home/pi/文档/pi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

仓库已有 `config.json`；按现场设备修改 `serial.port`、`object_camera.index` 和 `qr_camera.index`。`config.example.json` 用作配置参考。视觉调试需要 OpenCV 窗口时，请使用树莓派桌面、VNC 或显示器。

## 九点路网与坐标

```text
3 ──950── 2 ──950── 1
│1101.1   │1101.1   │1100
6 ──950── 5 ──900── 4
│855.9    │850      │851.5
9 ──850── 8 ──850── 7
```

图中数字单位为毫米。沿用当前车体方向：`1→2` 为 X 正方向（前进），`1→4` 为 Y 正方向（左移）。F4 的坐标以本次 `PATH_RESET_ORIGIN` 位置为原点；每个路径目标是相对该原点的绝对毫米坐标。

`config.json` 的 `node_route.edges[].distance_mm` 保存各边实测长度。`node_coordinates` 只表示逻辑邻接；`node_mm_coordinates` 目前用于判断边的 X/Y 正负方向，不直接作为逐段位移。执行时按经过的边长累计目标，同向连续边合成一段；未填长度的边不参与最短路，手动路线若包含未测量边也禁止下发。各边测量值不完全满足环路闭合，绕环执行后的累计坐标可能产生偏差，需现场复测。

```bash
python3 -B main.py --plan-path 1-2-3-6-9
python3 -B main.py --path-status
python3 -B main.py --execute-path '0,500' --path-id 1
python3 -B main.py --execute-node-route 1-2-3-6-9 --path-id 1 --path-rpm 40
python3 -B main.py --path-stop
```

`--plan-path` 只检查输入的节点序列，不访问硬件。`--execute-path` 仅接受一个绝对毫米目标，用于单段联调；`--execute-node-route` 先校验整条路线，再逐段发送。首段重置 F4 局部原点，后续段保留同一原点；每段必须收到匹配的 `DONE` 才推进。`ERROR`、请求失败、串口失联或终止帧缺失时停止推进。`--keep-path-origin` 只用于明确需要沿用既有原点的单段联调。

路径控制顺序为 `PATH_CLEAR → PATH_RESET_ORIGIN（首段）→ PATH_UPLOAD → PATH_START → DONE/ERROR`。Pi 运行中周期性查询状态；查询只是确认链路和状态，不代替到位事件。F4 执行一段时，即使连续收不到有效 Pi 帧，也会继续到完成或运动超时。`PATH_STOP` 在四轮编码器连续约 200 ms 静止后才回 ACK；1 秒内无法确认则回 NACK。`PATH_CLEAR` 在运动中回 BUSY，应先停止。

## 视觉与任务码

```bash
python3 -B main.py --dry-run --task 123+123+321+231
python3 -B main.py --vision-debug --vision-mode combined
python3 -B main.py --vision-debug --vision-mode objects
python3 -B main.py --vision-debug --vision-mode rings
python3 -B main.py --vision-image /path/to/photo.jpg --vision-mode combined
```

`--dry-run` 只校验任务码，不启动串口或相机。任务码允许红 `1`、黄 `2`、蓝 `3`、绿 `4`；两批各三种不同颜色，第二批颜色需在第一批有同色物料。`--vision-debug` 只打开物料相机，二维码相机配置独立；物料相机优先 V4L2/MJPG。按 `q`/Esc 退出、`s` 保存标注图、`p` 打印检测细节。`--vision-camera-index N` 只覆盖本次物料相机编号。离线照片模式不打开相机、串口或机构；单张图不具备连续帧稳定性。

视觉标定从 `object_camera.object_detection.roi`、遮挡区域、HSV 和参考点开始。黑白环单独用灰度/圆弧特征检测；现场确定三个环心后填写 `ring_detection.position_centers_px`，验证空环、遮挡和偏心图，再设置 `ring_detection.calibrated=true`。黄色 HSV、相机内参、透视映射及三个机构姿态的像素到毫米矩阵仍需现场标定。`vision_calibration.poses` 的零矩阵不可直接标记为已标定。

`PlacementAlignmentController` 的设计流程为：稳定目标 → 横移/伸缩并等待反馈 → 重新识别 → 对准后下降 → 松爪一次 → 撤离。第一层用环心，第二层用稳定的第一层同色物料参考点。`TimedMechanism` 只有固定时长，不提供真实动作完成反馈；完整机构协议和 `FeedbackMechanism` 适配尚未接入，因此当前不应把视觉识别结果直接用于自动松爪。

## 机械臂手动调试

5 号是旋转轴，6 号是前后轴，7 号是丝杆升降；三轴与底盘共用 F4 的 Emm_V5 总线。`--arm-jog` 固定 5 rpm、加速度档 10，每次 1–32 脉冲，默认 8。`--arm-move` 只支持 6/7 号，必须显式指定方向和脉冲数；6 号上限 1600 脉冲、60 rpm，7 号上限 6400 脉冲、180 rpm。默认速度分别为 30/120 rpm，加速度档 250。

```bash
python3 -B main.py --arm-status 7
python3 -B main.py --arm-jog 7 --arm-direction - --arm-pulses 8
python3 -B main.py --arm-move 7 --arm-direction - --arm-pulses 320 --arm-rpm 20 --arm-acceleration 100
python3 -B main.py --arm-stop 7
```

这些命令是独立示例，先确认当前剩余行程，再逐条执行。现场已确认：5 号 `+`/``-`` 对应观察视角的左/右转；6 号 `+` 向前，`-` 未单独验证；7 号 `+` 下降、`-` 上升。三轴均无实体限位开关，也没有自动回零。6 号齿条当前剩余行程不能由脉冲上限推断；发现方向不符、接近端点或卡滞时，直接切断电机电源。`ARM_DONE` 只表示编码器达到固件当前判据，不证明机构位置或视觉目标已满足。

## 协议、代码与验证

Pi/F4 使用 `uart3`（F4 PB10/PB11，115200 8N1）的二进制帧：

```text
AA 55 | version=01 | command | sequence | length:u16LE | payload | CRC16-Modbus:u16LE
```

路径命令为 `0x10..0x15`，机械臂命令为 `ARM_JOG=0x20`、`ARM_STOP=0x21`、`ARM_STATUS_REQ=0x22`、`ARM_MOVE=0x23`；ACK/NACK 为 `0x90/0x91`，路径 `DONE/ERROR/STATUS` 为 `0x93/0x94/0x95`，机械臂 `STATUS/DONE` 为 `0xA0/0xA1`。实现分别在 `logistics_robot/path_protocol.py`、`path_client.py`、`route_executor.py`、`vision.py`、`ring_detection.py` 和 `placement_alignment.py`。

```bash
python3 -m unittest discover -s tests -v
```

本地测试验证解析、状态机、规划与部分视觉逻辑；正式使用还需分别完成 F4 编译烧录、Pi 文件同步核对、真实串口抓包、相机画面标定和低速实车验证。
