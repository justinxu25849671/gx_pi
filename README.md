# 智能搬运机器人树莓派上位机

本工程仅保留与 `F4_slavedevice` 已贯通的 USART3 九点绝对航点二进制协议。

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

配置 `config.json` 中的串口设备和九点路网坐标。路径协议使用以重置原点为基准的
绝对毫米坐标：X 前正、Y 左正。

## 使用

任务码和路网预览不访问硬件：

```bash
python3 main.py --dry-run --task 156+123+516+231
python3 main.py --plan-path 1-2-3-6-9
```

接入 F4 后，先查询状态，再执行路径联调：

```bash
python3 main.py --path-status
python3 main.py --execute-path '0,500;500,500;500,1000' --path-id 1
python3 main.py --path-stop
```

默认执行路径前严格等待：

```text
PATH_CLEAR → PATH_RESET_ORIGIN → PATH_UPLOAD → PATH_START
```

`--keep-path-origin` 可跳过原点重置。运行中按 Ctrl+C 会发送 `PATH_STOP` 并等待
ACK。ACK 超时时会先查询 `PATH_STATUS`，不会盲目重发 `START`。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

自动化测试只验证上位机编解码与请求状态机；不代表串口、STM32、电机或实车路径
已经验证。
