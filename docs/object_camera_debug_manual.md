# 物料与放置环视觉调试手册

本调试入口只打开 `object_camera`。它不会创建二维码相机、`SerialPathLink`、路径对象
或机构对象。二维码相机仍由 `qr_camera` 独立配置。

## 启动

树莓派工程路径为 `/home/pi/文档/pi`。1080p/30 帧 USB 相机固定使用 V4L2 + MJPEG：

```bash
cd /home/pi/文档/pi
python3 -B main.py --vision-debug --vision-mode combined
```

也可只看物料或环：

```bash
python3 -B main.py --vision-debug --vision-mode objects
python3 -B main.py --vision-debug --vision-mode rings
```

`--vision-camera-index N` 只覆盖本次物料相机编号，不改配置。`q`/Esc 退出，`s` 保存
标注图，`p` 打印轮廓、参考点、稳定帧、质量以及环心质量。OpenCV 窗口需要桌面、VNC
或外接显示器，纯 SSH 通常不能显示。

## 离线照片

离线模式不打开任何相机、串口或机构：

```bash
python3 -B main.py --vision-image samples/held_red.jpg --vision-mode combined
python3 -B main.py --vision-image samples/empty_rings.jpg --vision-output logs/empty_rings_result.jpg
```

同一张照片只有一帧，所以 `stable=false` 是正常结果；稳定性必须用实时连续帧验证。

## 四颜色物料

`hsv_colors` 只允许 `red/yellow/blue/green`。黑白环不进入颜色分类。调试顺序：

1. 固定升降高度、夹爪旋转角、曝光和白平衡。
2. 在 `object_detection.roi` 中只保留工作台目标区域。
3. 将夹爪、线缆等固定遮挡写入 `occlusion_rects`，坐标可用 0..1 比例。
4. 用现场照片调整 HSV，再调整面积、宽高比、实心度和填充率。
5. 用 `reference_point.mode` 和 `offset_px` 标定抓取特征点；画面白色斜十字才是
   输出参考点，颜色轮廓重心只用于辅助显示。
6. 同色多目标时，程序结合上帧位置、观察参考点、质量和面积选择连续目标；达到
   `stable_frames` 且跳动不超过阈值后才标记稳定。

黄色阈值必须在取得实际黄色物料样本后重新标定。目前数值只是起点。

## 黑白环

环检测使用灰度边缘、可见圆弧、同心层数、拟合误差和环位绑定，不使用物料 HSV。
已录入物料底径 50 mm 和环外径 53、58、65、75、85、95 mm，但仍需按实际线宽、
拍摄高度和透视关系调整像素半径范围。

在实物图上确定三个环心后填写 `position_centers_px`，例如：

```json
"position_centers_px": {"1": [500, 520], "2": [960, 520], "3": [1420, 520]}
```

完成空环、持物遮挡和偏心照片验证后，才把 `ring_detection.calibrated` 改为 `true`。
在此之前候选会显示，但 `valid` 永远为假，自动放置不能启动。

## 证据边界

保存图像和终端输出只证明本地/树莓派视觉结果；不证明机构已对准、STM32 已动作、
松爪成功或真实码垛成功。配置或源码从本机改动后，需同步到 `/home/pi/文档/pi`，再在
树莓派上核对文件内容并重新运行。
