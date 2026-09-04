"""YOLO 目标检测。

支持四种检测器:
  - OnnxDetector: 进程内加载 ONNX 模型 (.onnx)，用 onnxruntime 推理（推荐，体积小）
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
        YOLO = __import__("ultralytics", fromlist=["YOLO"]).YOLO
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


class OnnxDetector(Detector):
    """基于 onnxruntime 的 YOLO 检测器（无需 torch/ultralytics，体积 ~200MB vs ~700MB）。

    预处理:  letterbox resize → RGB → normalize → CHW → batch
    后处理:  sigmoid scores → NMS → scale back to original frame

    模型输出格式 (YOLOv8, end2end=False):
      shape (1, 8, 8400)
      8 = 4 (cx, cy, w, h in 640×640 pixel coords) + 4 (class logits: floor, monster, rope, player)
    """

    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.45):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.conf = conf
        self.iou = iou
        self._path = model_path

        meta = self.session.get_modelmeta()
        self._names = {0: "floor", 1: "monster", 2: "rope", 3: "player"}
        if meta.custom_metadata_map:
            import json
            try:
                self._names = json.loads(meta.custom_metadata_map.get("names", "{}"))
                self._names = {int(k): v for k, v in self._names.items()}
            except (json.JSONDecodeError, ValueError):
                pass

        self._input_name = self.session.get_inputs()[0].name
        self._img_size = self.session.get_inputs()[0].shape[2]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        img, (scale, pad_left, pad_top) = self._preprocess(frame)
        outputs = self.session.run(None, {self._input_name: img})
        return self._postprocess(outputs[0], frame, scale, pad_left, pad_top)

    def _preprocess(self, frame: np.ndarray):
        """letterbox resize + normalize + CHW + batch。"""
        h, w = frame.shape[:2]
        scale = min(self._img_size / h, self._img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        dw = self._img_size - new_w
        dh = self._img_size - new_h
        pad_left, pad_top = dw // 2, dh // 2
        pad_right, pad_bottom = dw - pad_left, dh - pad_top
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img, (scale, pad_left, pad_top)

    def _postprocess(self, output, frame, scale, pad_left, pad_top):
        """解析 YOLOv8 ONNX 输出，NMS 后返回 Detection 列表。"""
        pred = output[0].T  # (7, 8400) → (8400, 7)
        boxes = pred[:, :4]   # (8400, 4)  cx, cy, w, h
        scores = pred[:, 4:]  # (8400, 3)  class scores

        # ultralytics 导出的 YOLOv8 ONNX 其 Detect 头已内置 Sigmoid，
        # 此时输出就是概率 (0,1)。只有 raw logits（最大值 > 1）才需要
        # 再套一次 sigmoid，否则会二次压缩置信度（0.94 -> 0.72），
        # 并让大量背景噪音抬升到阈值之上，导致框数爆炸。
        if scores.max() > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores))

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > self.conf
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        h, w = frame.shape[:2]
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        bboxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
        indices = cv2.dnn.NMSBoxes(
            bboxes, max_scores.tolist(), self.conf, self.iou,
        )
        if len(indices) == 0:
            return []

        dets = []
        for i in indices.flatten():
            cls_id = int(class_ids[i])
            dets.append(Detection(
                cls_name=self._names.get(cls_id, f"class_{cls_id}"),
                confidence=float(max_scores[i]),
                x=int(x1[i]), y=int(y1[i]),
                w=int(x2[i] - x1[i]), h=int(y2[i] - y1[i]),
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

    - 路径以 .onnx 结尾 → OnnxDetector（推荐，进程内 onnxruntime，无需 torch）
    - 路径以 .pt 结尾   → YoloDetector（进程内 ultralytics）
    - 路径以 .exe 结尾  → ExeDetector（调用外部 EXE）
    - 路径不存在或为空   → MockDetector（占位）
    """
    log = on_log or (lambda m: None)
    if not model_path:
        log("[提示] 未设置模型路径，使用 Mock 检测器")
        return MockDetector()

    if not os.path.exists(model_path):
        log(f"[警告] 模型文件不存在: {model_path}，使用 Mock 检测器")
        return MockDetector()

    path_lower = model_path.lower()

    # ---- ONNX 模式（推荐） ----
    if path_lower.endswith(".onnx"):
        try:
            det = OnnxDetector(model_path, conf)
            log(f"[检测] 已加载 ONNX 模型: {model_path}")
            return det
        except ImportError:
            log("[警告] 未安装 onnxruntime，回退到 Mock 检测器")
        except Exception as e:
            log(f"[警告] 加载 ONNX 模型失败: {e}，回退到 Mock 检测器")
        return MockDetector()

    # ---- EXE 模式 ----
    if path_lower.endswith(".exe"):
        try:
            det = ExeDetector(model_path, conf, on_log=log)
            return det
        except Exception as e:
            log(f"[警告] 创建 ExeDetector 失败: {e}，回退到 Mock 检测器")
            return MockDetector()

    # ---- .pt 模式（ultralytics） ----
    try:
        det = YoloDetector(model_path, conf)
        log(f"[检测] 已加载 YOLO 模型: {model_path}")
        return det
    except ImportError:
        log("[警告] 未安装 ultralytics，回退到 Mock 检测器。安装: pip install ultralytics")
    except Exception as e:
        log(f"[警告] 加载模型失败: {e}，回退到 Mock 检测器")

    return MockDetector()