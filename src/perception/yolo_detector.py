"""YOLO 目标检测。

支持三种检测器:
  - YoloDetector:  进程内加载 ultralytics 模型 (.pt)
  - ExeDetector:   调用外部 EXE 程序，通过临时文件 + stdout JSON 通信
  - MockDetector:  占位检测器，返回空结果

EXE 接口约定:
  输入:  EXE 接收 --image <png路径> --conf <置信度> 参数
  输出:  stdout 输出 JSON 数组，每个元素:
         {"class":"monster","conf":0.92,"x":100,"y":200,"w":80,"h":60}
"""
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Callable, Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """单个检测结果。坐标基于传入的 frame。"""
    cls_name: str
    confidence: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class Detector:
    """检测器基类。子类实现 detect()。"""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class MockDetector(Detector):
    """占位检测器，返回空结果。"""

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []


class YoloDetector(Detector):
    """基于 ultralytics YOLO 的检测器。

    参考: 模型训练好后，设置 config.detection.model_path 指向 .pt 文件即可自动启用。
    """

    def __init__(self, model_path: str, conf: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self._path = model_path  # 供上层判断是否需要重建检测器

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(
                    cls_name=self.model.names[int(box.cls)],
                    confidence=float(box.conf),
                    x=int(x1), y=int(y1),
                    w=int(x2 - x1), h=int(y2 - y1),
                ))
        return dets


class ExeDetector(Detector):
    """通过外部 EXE 程序调用 YOLO 检测。

    每帧流程:
      1. 把 frame 写入临时 PNG 文件
      2. subprocess 调用 EXE:  {exe} --image {png} --conf {conf}
      3. 读取 stdout 的 JSON 数组
      4. 解析为 [Detection, ...] 返回
      5. 清理临时文件

    EXE 输出格式（stdout，JSON 数组）:
      [
        {"class":"monster","conf":0.92,"x":100,"y":200,"w":80,"h":60},
        {"class":"rope",   "conf":0.88,"x":300,"y":150,"w":20,"h":200},
        {"class":"floor",  "conf":0.95,"x":0,  "y":400,"w":800,"h":30}
      ]

    Args:
        exe_path:   EXE 文件路径
        conf:       置信度阈值
        extra_args: 额外命令行参数列表，如 ["--device", "cuda"]
        timeout:    单次检测超时秒数，默认 10
        on_log:     日志回调
    """

    def __init__(self, exe_path: str, conf: float = 0.5,
                 extra_args: Optional[List[str]] = None,
                 timeout: float = 10.0,
                 on_log: Optional[Callable[[str], None]] = None):
        self._path = exe_path  # 供上层判断是否需要重建检测器
        self.conf = conf
        self.extra_args = extra_args or []
        self.timeout = timeout
        self._log = on_log or (lambda m: None)
        self._log(f"[EXE检测] 已注册: {exe_path}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # 1. 写临时 PNG
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, frame)

            # 2. 调用 EXE
            cmd = [self._path, "--image", tmp_path, "--conf", str(self.conf)]
            cmd.extend(self.extra_args)

            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if proc.returncode != 0:
                stderr = proc.stderr.strip()[:200]
                self._log(f"[EXE检测] 返回码 {proc.returncode}: {stderr}")
                return []

            # 3. 解析 JSON
            data = json.loads(proc.stdout)
            dets = []
            for item in data:
                dets.append(Detection(
                    cls_name=str(item.get("class", "unknown")),
                    confidence=float(item.get("conf", 0)),
                    x=int(item["x"]),
                    y=int(item["y"]),
                    w=int(item["w"]),
                    h=int(item["h"]),
                ))
            return dets

        except subprocess.TimeoutExpired:
            self._log(f"[EXE检测] 超时 ({self.timeout}s)")
            return []
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._log(f"[EXE检测] 解析失败: {e}")
            return []
        except FileNotFoundError:
            self._log(f"[EXE检测] 文件未找到: {self._path}")
            return []
        except Exception as e:
            self._log(f"[EXE检测] 异常: {e}")
            return []
        finally:
            # 5. 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def create_detector(model_path: str, conf: float = 0.5,
                    on_log: Optional[Callable[[str], None]] = None) -> Detector:
    """根据配置创建检测器。

    - 路径以 .exe 结尾 → ExeDetector（调用外部 EXE）
    - 路径以 .pt/.onnx 结尾 → YoloDetector（进程内 ultralytics）
    - 路径不存在或为空 → MockDetector（占位）
    """
    log = on_log or (lambda m: None)
    if not model_path:
        log("[提示] 未设置模型路径，使用 Mock 检测器")
        return MockDetector()

    # ---- EXE 模式 ----
    if model_path.lower().endswith(".exe"):
        if not os.path.exists(model_path):
            log(f"[警告] EXE 文件不存在: {model_path}，使用 Mock 检测器")
            return MockDetector()
        try:
            det = ExeDetector(model_path, conf, on_log=log)
            return det
        except Exception as e:
            log(f"[警告] 创建 ExeDetector 失败: {e}，回退到 Mock 检测器")
            return MockDetector()

    # ---- .pt / .onnx 模式 ----
    if not os.path.exists(model_path):
        log(f"[警告] 模型文件不存在: {model_path}，使用 Mock 检测器")
        return MockDetector()

    try:
        det = YoloDetector(model_path, conf)
        log(f"[检测] 已加载 YOLO 模型: {model_path}")
        return det
    except ImportError:
        log("[警告] 未安装 ultralytics，回退到 Mock 检测器。安装: pip install ultralytics")
    except Exception as e:
        log(f"[警告] 加载模型失败: {e}，回退到 Mock 检测器")

    return MockDetector()