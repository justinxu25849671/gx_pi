# 智能搬运机器人树莓派上位机

本工程仅保留与 `F4_slavedevice` 已贯通的 USART3 单段绝对毫米目标二进制协议。
树莓派负责节点图规划和逐段调度；STM32 只负责当前段的运动、航向保持、超时/急停和完成或失败反馈。

```text
/Users/justin/Documents/program/company/gongxun/
├── pi/                # 本工程
└── F4_slavedevice/    # STM32F407VET6 + RT-Thread 下位机
```

上位机实现位于 `logistics_robot/path_protocol.py`、`path_client.py` 和
`serial_link.py`；下位机实现位于 `applications/modules/rpi_link.c` 与
`applications/app/path_control.c`。串口使用 F4 `uart3`（PB10/PB11），115200 8N1，
帧格式为：

```text
AA 55 | version | command | sequence | length | payload | CRC16-Modbus
```

旧 `gx_car` 的换行文本协议及其台架工具已移除；上位机不会再发送
`HEARTBEAT`、`ENABLE`、`VEL`、`STOP` 或 `ESTOP` 文本帧。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
```

配置 `config.json` 中的串口设备和九点路网坐标。`node_coordinates` 仅是用于邻接和
转弯判断的逻辑格点，绝不能下发给车。执行节点路线前，必须为每个会用到的节点填写
独立的 `node_mm_coordinates` 实测毫米坐标。路径协议使用以重置原点为基准的绝对毫米
坐标：X 前正、Y 左正。

## 使用

任务码只允许红 `1`、黄 `2`、蓝 `3`、绿 `4`；两批各三种不同颜色，且第二批颜色
必须在第一批存在同色承接物料。任务码和路网预览不访问硬件：

```bash
python3 main.py --dry-run --task 123+123+321+231
python3 main.py --plan-path 1-2-3-6-9
```

接入 F4 后，先查询状态，再执行路径联调：

```bash
python3 main.py --path-status
python3 main.py --execute-path '0,500' --path-id 1
python3 main.py --execute-node-route 1-2-3-6-9 --path-id 1 --path-rpm 40
python3 main.py --path-stop
```

每个节点段都严格等待：

```text
PATH_CLEAR → PATH_RESET_ORIGIN（仅首段）→ PATH_UPLOAD(1 point) → PATH_START
→ STM32 DONE / ERROR → 下一段（或停止）
```

`--execute-path` 仅用于单段联调。`--execute-node-route` 会在上一段 `DONE` 后才提交
下一段，遇到 `ERROR`、断链或请求失败不会推进。`--keep-path-origin` 可跳过单段联调的
原点重置。运行中按 Ctrl+C 会发送 `PATH_STOP` 并等待 ACK。ACK 超时时会先查询
`PATH_STATUS`，不会盲目重发 `START`。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

自动化测试只验证上位机编解码与请求状态机；不代表串口、STM32、电机或实车路径
已经验证。

## 物料识别相机调试

物料相机可在不连接 STM32 的情况下独立调试物料和放置环：

```bash
python3 main.py --vision-debug --vision-mode combined
python3 main.py --vision-image samples/held_object.jpg --vision-mode combined
```

详细的启动、按键、现场记录和故障排查见
[物料识别相机组调试手册](docs/object_camera_debug_manual.md)。

松爪前横移/伸缩、动作完成反馈、第一/二层目标区别和一次性放置约束见
[放置对准接入说明](docs/placement_alignment_manual.md)。当前配置的环检测与三个视觉
姿态均保持 `calibrated: false`，且仓库没有真实机构动作协议；完成现场标定和反馈适配
前，自动放置会被安全拒绝。
