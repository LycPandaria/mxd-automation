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

## 与分享包（share-baseline 分支）测试的关系

分享包带 17 个单测，其中 `tests/test_decision_flow.py`、`tests/test_range_limit.py`
是决策层的"按帧回放"骨架（接管 `ctxmod.time`、构造 `Context`、断言 executor 收到的键），
质量很高，但依赖分享包独有的接口和配置：

- `Context.frame_height/frame_width`（分辨率缩放）
- `_px()` / `_monster_nearby` / `_jump_by_distance`
- `_track_hits` / `_check_drift`（击退感知的卡键检测）
- `_range_step`（小地图活动范围）
- `_last_cast_at`（刚出手后的追击耐心）

这些在本地主干上还不存在，整批搬进来会全红、失去回归意义。
所以第 0 步只在本地主干上建"能跑起来且全绿"的最小骨架；
分享包的用例留在 `share-baseline` 分支，**随功能一起移植**——
每移植一个功能，就把对应用例搬到本目录并改成本地接口。

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
