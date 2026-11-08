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
| `test_chase_retreat.py` | 追击：刚出手先等怪过来、卡住两次放弃；近身：最近威胁判后撤、撤不动跳过怪；定位丢失不再左右走动 |
| `test_channel_detect.py` | 换频道界面识别：菜单/频道窗口在"等比缩放"和"固定像素"两种 UI 下都认得出 |
| `test_target_side.py` | 选怪优先级：同侧射程内 > 反侧最近 > 同侧射程外；反侧近身怪不参与威胁/击退 |
| `test_quit.py` | 定时结束游戏：时间计算、登录界面识别（含 768 固定像素 UI）、退出流程/确认框/中途停止 |
| `test_turn_hold.py` | 按住转向（turn_mode=hold）：位移确认、最短/最长时长、悬崖/边缘/射程/近身怪提前松键、吞键重试、目标换边放弃 |
| `test_aoe_hold.py` | AOE 按住连发（aoe_mode=hold）：触发/单怪走轮转、满时长松键、近身威胁/怪清空/目标换边/buff 窗口提前松、间隔门与再触发、停止松键 |
| `test_phantom.py` | 目标横跳冷却（1.5s 抢占间隔）+ 影子位置限一击（漏检沿用抑制/AOE hold 不对影子开/可见恢复） |

## 与分享包（share-baseline 分支）测试的关系

分享包带 17 个单测，其中 `tests/test_decision_flow.py`、`tests/test_range_limit.py`
是决策层的"按帧回放"骨架（接管 `ctxmod.time`、构造 `Context`、断言 executor 收到的键），
质量很高。随着移植推进，依赖的接口逐项补齐：

- 已移植：`Context.frame_height/frame_width`、`_px()`、`_range_step`（活动范围）、
  `_last_cast_at`、残影 `_in_ghost_zone`、`_monster_nearby`（追击耐心）、
  后撤突围（定位丢失左右走按实机反馈已移除，不移植）
- 尚未移植：`_jump_by_distance`（跳跃攻击）、`_track_hits` / `_check_drift`
  （击退感知的卡键检测）、`_stand_*` 回位

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
