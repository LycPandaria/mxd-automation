# 决策引擎测试

## 运行

在仓库根目录：

```bash
PYTHONPATH=. python -m unittest discover -s tests -t . -v
```

只需要 Python 标准库 + 项目本身的 `src`（不装 opencv / onnxruntime 也能跑：
用例只构造 `Context`，不跑 YOLO、不抓屏、不碰键鼠）。

## 目录

| 文件 | 作用 |
|---|---|
| `fake_executor.py` | 假执行器。按本地 `ActionExecutor` 签名实现（`key_up(key, force=)`），记录按下的键 |
| `test_smoke.py` | 冒烟：加血、射程内出手、无怪探索、空 Context、同平台判定 |
| `test_range_limit.py` | 活动范围：线外怪过滤、分界线/分界比例、过线走回、无黄点时原地、站桩只过滤 |
| `test_ghost.py` | 残影：出手次数 + 静止判定 + 位置记忆（8→16→…→300s）、跳过与不重锁、活怪 2 倍 |

## 与分享包（share-baseline 分支）测试的关系

分享包带 17 个单测，其中 `tests/test_decision_flow.py`、`tests/test_range_limit.py`
是决策层的"按帧回放"骨架（接管 `ctxmod.time`、构造 `Context`、断言 executor 收到的键），
质量很高。随着移植推进，依赖的接口逐项补齐：

- 已移植：`Context.frame_height/frame_width`、`_px()`、`_range_step`（活动范围）、
  `_last_cast_at`、残影 `_in_ghost_zone`
- 尚未移植：`_monster_nearby` / `_jump_by_distance`（跳跃攻击 + 追击耐心）、
  `_track_hits` / `_check_drift`（击退感知的卡键检测）、`_stand_*` 回位

分享包里还没搬过来的用例仍留在 `share-baseline` 分支，**随功能一起移植**——
每移植一个功能，就把对应用例搬到本目录并改成本地接口（例如 `test_range_limit.py`
已按本地"转向生效自检"重写）。

在 `share-baseline` 分支上查看/运行分享包原始用例：

```bash
git checkout share-baseline
PYTHONPATH=. python tests/test_decision_flow.py
git checkout merge/auto-farm
```

## 分支约定

| 分支 | 内容 |
|---|---|
| `main` | 合并前的原始主干 |
| `merge/auto-farm` | 合并工作分支：本地工作区快照 + 逐步移植的功能 + 本测试骨架 |
| `share-baseline` | 分享包源码覆盖快照，仅作 `git diff merge/auto-farm share-baseline` 对照，不运行、不合并 |
