"""智能搬运机器人树莓派上位机启动入口。

@File    : main.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 只负责启动应用；业务逻辑位于 ``logistics_robot`` 包内。
"""

from logistics_robot.app import main


if __name__ == "__main__":
    # 保持入口极简：便于被测试工具导入，也避免导入模块时启动机器人。
    raise SystemExit(main())
