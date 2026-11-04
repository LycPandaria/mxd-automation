"""决策上下文与反应式决策引擎。

================================================================================
设计理念
================================================================================

  不再建地图、不做 A* 寻路。每一帧只看 YOLO 检测到的画面内容，
  像人类玩家一样"看到什么就做什么反应"。

  画面里有什么 → 就应该做什么:
    - 看到怪 → 判断同平台还是跨平台，走过去或爬绳跳下去
    - 看到地板 → 知道哪里能站
    - 看到绳索 → 知道哪里能爬
    - 没看到怪 → 往一个方向走探索
    - HP/MP 低 → 加血加蓝

================================================================================
架构
================================================================================

  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────┐
  │  感知层      │ →  │  DecisionEngine     │ →  │ ActionExec   │
  │ YOLO/OCR/HP │    │  (反应式决策 + FSM)  │    │ (方向键/技能) │
  └─────────────┘    └─────────────────────┘    └──────────────┘

================================================================================
决策流程（优先级从高到低）
================================================================================

  1. HP 低于阈值 → 加血键
  2. MP 低于阈值 → 加蓝键
  3. 检测到怪物:
     a. 同平台 → 按住方向键走过去，进入 200px 攻击范围后停止移动原地攻击
     b. 怪在上方 + 有绳索 → 走到绳索正下方，跳跃 + 按住上键爬绳
     c. 怪在下方/跨平台 → 按住方向键移动 + 按需跳跃
     d. 都不满足 → Tab 选怪 + 原地攻击
  4. 没怪 → 探索（按住方向键往一个方向走，遇坑跳）

【移动方式】所有移动（追击/攀爬/探索）都是"按住方向键不松手"，
  进入攻击范围或攻击时才释放方向键，攻击期间完全不移动。

================================================================================
Context 字段说明
================================================================================

  monsters:       YOLO 检测到的怪物列表
  floors:         地板列表
  ropes:          绳索列表
  self_position:  自身脚底坐标 (cx, cy) 或 None
  self_center:    自身角色中心点坐标 (cx, cy) 或 None（OCR 定位时记录）
  hp_ratio:       血量比例 0.0~1.0
  mp_ratio:       蓝量比例 0.0~1.0
  detections:     全部 YOLO 检测结果（含所有类别）
"""
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from ..perception.yolo_detector import Detection
from ..execution.action_executor import ActionExecutor
from ..utils.config_loader import Config
from .fsm import FSM, State
from .distance import (
    estimate_path_distance,
    JUMP_HEIGHT,
    PLATFORM_JUMP_GAP_X,
    SAME_LEVEL_Y_TOLERANCE,
    ROPE_REACH_Y,
    PathEstimate,
)


# =============================================================================
# 反应式决策的阈值常量
# =============================================================================

ATTACK_RANGE_X = 200
"""攻击范围默认值：自身与怪物 X 坐标差小于此值才开始攻击（像素）。

仅长手(远程)使用：实际生效值取 config.attack_range
（exe 界面"攻击距离px"可修改），此常量仅作 __init__ 里的兜底默认值。
短手(近战)固定 MELEE_ATTACK_RANGE_X=50，不读此配置。
"""

MELEE_ATTACK_RANGE_X = 50
"""短手(近战)固定贴脸距离（像素），【不允许配置修改】。

短手与长手攻击距离完全独立：长手读 exe 配置(attack_range)，
短手固定 50px。贴脸判定用"角色到怪近侧身体边缘"的距离，
宽怪站旁边就能打，不穿越怪身体。
"""

ATTACK_MIN_RANGE_X = 70
"""长手(远程)最小有效射程（像素）：角色与怪中心水平距离小于此值时
弓箭手会"挥弓"而非射箭，伤害大幅下降 → 触发后撤拉开距离。

实际生效值取 config.attack_min_range（0 = 关闭后撤），此常量仅作
getattr 兜底；只对 attack_type=="long" 生效，短手(近战)贴脸打怪
不受影响。"""

RETREAT_HYSTERESIS_X = 30
"""后撤退出滞回（像素）：进入后撤后，直到 dx ≥ min_range+此值才停止，
避免在最小射程边界来回抖。"""

RETREAT_DEAD_ZONE = 10
"""后撤方向死区（像素）：|角色x-怪中心x| 小于此值视为水平重叠，沿用上一帧方向。"""

RETREAT_EDGE_LOOKAHEAD_X = 45
"""后撤前检查后退方向前方此距离处脚下是否有地板（防落崖）。"""

AOE_MONSTER_COUNT_THRESHOLD = 2
"""范围攻击(AOE)触发阈值：攻击方向（朝向正前方）同平台怪物数 ≥ 此值时
改放第二个技能(技能2, AOE)；否则放第一个技能(技能1)。仅长手(远程)生效。
例外：站桩模式 + stand_skill2_only(默认开) 时不做这个判定，恒定只放技能2。"""

AOE_BURST_COUNT = 3
"""AOE 连发次数：每次判定放技能2(爆炸箭)时，连发此数量的爆炸箭
（隔冷却逐发补满，成功放出一发才递减）。"""

# ---- 站桩定时微动（反"定点一动不动"的机器特征）----
# 每过 N 秒做一次"很短的左/右点按"即可，不做位置闭环（不关心挪了几像素）。

MICRO_MOVE_TAP_MIN_SECONDS = 0.02
"""单次点按下限（秒）：再短游戏可能来不及响应。"""

MICRO_MOVE_TAP_MAX_SECONDS = 0.30
"""单次点按上限（秒）：兜住配置填错（填太大就变成明显走动了）。"""

MICRO_MOVE_TAP_GAP_SECONDS = 0.08
"""两次点按之间的间隔（秒）：让游戏当成两次独立操作，也更像人。"""

MICRO_MOVE_EDGE_LOOKAHEAD_X = 15
"""防落崖前瞻距离（像素）：只覆盖一次微动的位移量级（几像素），
所以比后撤用的 RETREAT_EDGE_LOOKAHEAD_X(45) 近得多，不会动不动就跳过。"""

# ---- 追击节奏 ----
CHASE_PATIENCE_SECONDS = 1.0
"""刚出过手这么久内，攻击距离外的怪先不追，原地等它过来（有怪进入攻击距离会立刻打）。
怪多、还有人抢怪时，刚打完一只就跑去追远处的，常常追一半又掉头，来回白跑"""

CHASE_WAIT_MARGIN = 50
"""只有同一层离攻击距离不到这么远（px）的地方还有怪时才按 CHASE_PATIENCE_SECONDS 原地等。
再远的怪等也要等好一会儿，走几步过去更快。"""

CHASE_STUCK_GIVEUP = 2
"""追击中连续卡住这么多次（每次约 STUCK_FRAMES 帧不动，会先跳一下）还没追到 → 放弃该目标，
把那个位置当残影暂时忽略（地图尽头/墙/怪飘在平台外，走过去也打不到）"""

# ---- 后撤 / 翻越突围 ----
RETREAT_MAX_SECONDS = 1.5
"""后撤持续这么久还贴着怪（撞墙、被地形挡住）→ 改为翻越突围：朝怪方向起跳越过它。
后撤期间不放技能，一直退不动等于白停手。"""

ESCAPE_MAX_TIME = 2.5
"""翻越突围的最长时间（秒），超时就结束突围。"""

ESCAPE_STUCK_FRAMES = 15
"""突围时位置连续这么多帧基本不动 → 再跳一次。"""

OCCLUSION_RETREAT_X = 50
"""长手(远程)遮挡判定阈值（像素）：攻击中锁定目标消失且最后已知水平距离
< 此值时，判定为"被角色/特效遮挡"(怪贴脸)而非死亡 → 用最后位置后撤。"""

OCCLUSION_RETREAT_MAX_SECONDS = 2.0
"""长手(远程)遮挡后撤的最大持续秒数。超过仍未重新看到目标 → 判定目标
真消失（已死/离开），放弃后撤、清锁转探索。（按 fps 换算成帧）"""

HP_DROP_THRESHOLD = 0.01
"""掉血触发后撤的阈值：本帧 hp_ratio 比上一帧下降 ≥ 此值（1%）视为"被怪打到"，
作为"怪被遮挡但仍在打你"的兜底信号（仅长手）。"""

HP_DROP_ACTIVE_SECONDS = 1.5
"""掉血信号有效秒数：掉血后此时间内允许触发一次后撤。（按 fps 换算成帧）"""

HP_RETREAT_HOLD_SECONDS = 1.0
"""掉血触发后撤的持续秒数：触发后持续后撤此时间，避免只退一帧就停、又被打。
（按 fps 换算成帧）"""

TARGET_MISS_RETAIN_SECONDS = 0.5
"""长手(远程)目标漏检保持秒数（射程外）：攻击中锁定目标因技能特效被遮挡而
短时漏检时，沿用最后位置继续攻击此时间，避免"换目标/乱跑"；超过才重选。
（按 fps 换算成帧）

射程内的目标用更长的 TARGET_MISS_RETAIN_INRANGE_SECONDS（特效盖住时目标
通常还在原地，保持更久可以少换目标/少转身）。"""

TARGET_MISS_RETAIN_INRANGE_SECONDS = 1.0
"""射程内目标漏检保持秒数：目标最后位置仍在攻击距离内时，多半是技能特效/
角色遮挡导致 YOLO 置信度掉下去，目标本身没消失。保持 1.0s 让特效过去，
期间不换目标；只有明显更近（≥NEARER_GHOST_INRANGE_MARGIN_X）的可见怪
才会抢占（见 _pick_nearer_visible）。（按 fps 换算成帧）"""

SELECT_FALLBACK_SECONDS = 3.0
"""自身定位失败时，选怪允许使用"最后已知位置"的最大时效（秒）。

不超过此时间 → 仍按"同平台最近"选怪（短时漏检不影响锁定/换怪）；
超过 → 宁可本帧不选怪（转探索/等待定位），也不再按"画面最大框"
乱锁远处或上层平台的怪——10/24 日志里 500px+ 的跨半屏追击多由此产生。
（按 fps 换算成帧）"""

NEARER_SWITCH_MARGIN_X = 120
"""更近的怪抢占所需的距离优势（像素）。

锁定目标还可见时，新怪至少要近这么多才切换，避免两只怪距离接近时
左右来回换目标、反复转身抖动。"""

NEARER_TRULY_CLOSE_X = 150
"""跨侧切换的"真正贴身"距离（像素）。

跨侧换目标必须先转身（约 0.25~0.4s 停手）。只有新目标进到这个距离内、
并且确实比当前锁定目标更近时，才值得转一次身。原阈值 250px 太宽：会把
195px 的前排目标换成 229px 的背后目标（10/29 日志 26 次），收紧到 150。"""

NEARER_GHOST_INRANGE_MARGIN_X = 300
"""射程内鬼目标被抢占所需的距离优势（像素）。

原目标本帧漏检、但最后位置仍在射程内时，先当成"被特效挡住"保持锁定
（TARGET_MISS_RETAIN_INRANGE_SECONDS）；只有可见新目标比它近至少 300px
才允许抢占。跨侧另加 NEARER_CROSS_SIDE_EXTRA_X。"""

NEARER_SWITCH_FRAMES = 3
"""更近的怪连续满足条件多少帧才真正切换（吸收 YOLO 单帧漏检/抖动）。"""

NEARER_CROSS_SIDE_EXTRA_X = 150
"""跨侧换目标的额外距离优势（像素）。

当前锁定目标在角色面朝这一侧时，切到另一侧的目标必须先转身（约 0.25~0.4s
停手）。因此跨侧抢占要求新目标再多近 150px（总优势 = 120 + 150），
避免为了两只距离接近的怪反复转身；真正贴身的怪仍会抢占。"""

SAMESIDE_PREFER_X = 300
"""选怪时的"同侧优先"距离（像素）。

朝向已知时，优先打面朝这一侧的同平台怪；另一侧的怪必须比同侧最近怪
还近 150px 以上才会被选中——选背后的怪意味着转身（约 0.25~0.4s 停手），
远一点的同侧怪用现有朝向直接就能打。朝向未知（_face_dir 为空）时不做
同侧偏好，保持原来的"最近优先"。"""

ROPE_SEARCH_RANGE_X = 200
"""搜索绳索的水平范围（像素）"""

CLIMB_ALIGN_TOLERANCE = 5
"""攀爬对准容差：人物中心与绳索中心 X 差小于此值（±5px）
视为在同一竖直轴线（绳索正下方），才允许抓绳攀爬（像素）"""

CLIMB_EXIT_FRAMES = 45
"""爬绳结束后横向走出绳索的帧数（约 2 秒 @20fps），期间不重新抓绳"""

GHOST_CASTS = 6
"""同一个目标打了这么多下还在 → 判定为残影/误识别（config.ghost_casts 可改，0=关闭）。

怪物死亡后 YOLO 仍可能把尸体/消失残影、或背景里像怪的东西检测为 monster，
位置匹配会一直锁定它，导致角色对着空气打、不换下一只。
按出手次数算而不是帧数：跳跃攻击的起跳/滞空期间不做决策，按帧数累计会慢几十倍。"""

GHOST_ZONE_SECONDS = 8.0
"""判定为残影的位置，第一次忽略这么多秒（避免换走后马上又选回来）。

同一个位置再次被判定为残影，忽略时间翻倍（8 → 16 → 32 …），最长 GHOST_ZONE_MAX_SECONDS。
地图边上的固定物体被误认成怪时，会被一次次判定为残影，越罚越久，不会反复去追。"""

GHOST_ZONE_MAX_SECONDS = 300.0
"""残影位置最长忽略时间（秒）。"""

GHOST_MEMORY_SECONDS = 600.0
"""残影位置的"前科"保留多久（秒）：这段时间内再被判定，按次数翻倍计时。"""

GHOST_ZONE_RADIUS = 60
"""残影位置的忽略半径（px，768 高画面下）。"""

GHOST_MIN_SECONDS = 3.0
"""残影判定还要求对它打了至少这么久（秒）：攻击冷却设得很短时按键次数涨得快，
但出手动作没做完时多按的那几下游戏不认，不能凭次数就判残影。"""

GHOST_STATIC_PX = 20
"""从开始计数起目标位置偏离超过这么多（px，768 高画面下）就说明是活的（在走、被打得往后退、
或是好几只怪轮流经过同一个地方），要打两倍次数、两倍时间还在才放弃（多半是打不到）。
误识别的地图物体、死亡残影是不动的，按 ghost_casts 次判定。"""

STUCK_FRAMES = 60
"""卡住判定帧数：持续此帧数位置不变则视为卡住"""

EXPLORE_DIRECTION_SWITCH_FRAMES = 180
"""探索方向切换帧数：探索状态下持续此帧数没遇到怪就换方向"""

DISTANCE_LOG_FRAMES = 60
"""距离推算日志输出间隔帧数（避免刷屏）"""

FACE_TURN_X = 20
"""攻击转向判定：怪物中心 x 与角色 x 差超过此值才调整朝向（像素）。
小于此值视为怪物在正下方/重叠，保持当前朝向即可。"""

# =============================================================================
# 转向（朝向）—— 与施法分帧 + 定期重申 + 生效自检
# =============================================================================
#
# 背景（实测 2026-09-23 12:37~12:38 那一局：连续 12 秒朝着没怪的方向放技能）：
#   1. 游戏会忽略"攻击动画期间"的按键——代码在 buff 施法窗口(main.py 顶部
#      注释 + request_buff_window)里已经用到这条结论；但转向键却和技能键挤在
#      同一帧里按，30~90ms 的短按极容易被吞掉。
#   2. _face_dir（朝向记忆）只在"按键已发出"时更新，不校验游戏是否真的转了；
#      而且"need == _face_dir 就不再按方向键"——于是一次被吞掉的转向会让
#      "以为朝向对了、实际是反的"这个状态一直持续下去（日志里一条 [朝向]
#      都没有，只能看到技能一直朝反方向放）。
# 对策：
#   · 转向单独占一帧（本帧不按技能键），方向键按住 120~180ms（不是短按）；
#   · 攻击中定期重申一次朝向（4~6s 抖动），把被吞掉的转向自愈回来；
#   · 用"角色有没有朝该方向动几像素"自检按键是否被游戏接收，没接收就提前重试；
#   · 站定期间补发方向键 KEYUP，兜住"某次 KEYUP 丢失 → 角色一直朝一个方向走"。

TURN_TAP_MIN_SECONDS = 0.12
"""转向按住时长下限（秒）。press_key 是 30~90ms 的短按，实测容易被游戏吞掉；
转向需要一次"明确的按住"，让 DirectInput 的按下状态稳定被游戏采样到。"""

TURN_TAP_MAX_SECONDS = 0.18
"""转向按住时长上限（秒）。再长就是明显走动（一次约 5~15px）；转向键带一点
位移属于冒险岛的正常操作（原地转头也会挪半步），不影响贴脸判定。

上限从 0.20 收到 0.18：每次转向少阻塞约 20ms，配合下面的快速确认，
整体转向耗时明显下降。"""

TURN_MIN_GAP_SECONDS = 0.20
"""两次转向按键之间的最小间隔（秒）：目标左右横跳时防止每帧狂按方向键——
每帧都变成"转向帧"会把技能全挤掉（实测 12:37:50~52 出现过 right→left→right）。

从 0.30 收到 0.20：按键被吞后的重试/等间隔停手更短，减少技能空窗。"""

TURN_REASSERT_MIN_SECONDS = 4.0
"""朝向重申间隔下限（秒）：攻击中即使认为朝向已正确，也隔这么久重申一次，
用一次干净的转向按键把"可能已被吞掉"的朝向自愈回来。

【为什么放宽到 4~6】转向确认已改为"6px + 连续两帧/大幅位移立即确认"，
一次性被吞的转向会被自检发现并 0.2s 内重试，不再依赖高频重申兜底；
而每次重申都要占一帧不放技能。放宽后减少约 1/3 的转向帧，提升输出节奏。"""

TURN_REASSERT_MAX_SECONDS = 6.0
"""朝向重申间隔上限（秒）：与下限之间随机取值，避免按键节奏完全规律。"""

TURN_RETRY_MIN_SECONDS = 0.15
"""自检判定"这次转向没被游戏接收"后的重试间隔下限（秒）。"""

TURN_RETRY_MAX_SECONDS = 0.30
"""自检判定"这次转向没被游戏接收"后的重试间隔上限（秒）。"""

TURN_VERIFY_MIN_MOVE_X = 6
"""转向生效自检：按下方向键后角色朝该方向水平位移 ≥ 此值（像素）视为生效。

冒险岛里按一下方向键角色会挪几像素，所以"一点没动"基本等价于按键没被游戏
接收（技能动画输入锁 / 按键丢失）。

原值 2px 小于 YOLO 自身框的逐帧抖动（常见 3~10px）和被怪击退的位移，
会把"没转成功"误判成成功 → _face_dir 写错且不再纠正，随后连续朝反方向打
（10/25 21:36 现场）。提高到 6px 并要求连续两帧达标（见
TURN_VERIFY_CONFIRM_HITS），避免单帧检测抖动误确认。"""

TURN_VERIFY_CONFIRM_HITS = 2
"""转向自检连续达标帧数：位移 ≥ TURN_VERIFY_MIN_MOVE_X 需连续满足此帧数
才写入 _face_dir。

单帧位置跳变（YOLO 框抖动/被击退一帧）不足以确认转向；真实转向后角色会
持续保持在偏移位置，连续两帧达标才可信。"""

TURN_VERIFY_FAST_MOVE_X = 12
"""转向快速确认阈值（像素）：首帧位移 ≥ 此值时直接确认（1 帧即可）。

转向点按本身会移动 12~20px，一次就达到 12px 基本不可能是检测抖动；
用"6px 需连续两帧 / ≥12px 一帧即确认"的混合规则，大多数转向能少等一帧。
取 12 而不是 10：YOLO 自身框单帧抖动偶见 10px，12px 以上才足够可信。"""

TURN_VERIFY_MISSING_GRACE_SECONDS = 0.35
"""自检期间自身定位丢失的额外等待时间（秒）。

等待窗口内定位失败 → 本帧无法判断转向是否生效。多等这段时间让定位恢复；
仍不恢复则丢掉这一轮待检（不写 _face_dir），由后续重试/盲点继续纠正。
期间调用方禁止施法，避免朝未确认的朝向打。"""

TURN_VERIFY_MAX_FAILS = 3
"""连续自检失败上限：仅用于日志限频（第 1 次和达到上限时各打一条）。

纠正转向失败后始终按 TURN_RETRY 的短间隔重试，不再退回长间隔——严格模式
下宁可停手等转向生效，也不朝旧方向施法。"""

# =============================================================================
# 攻击动画锁 + 换边防抖（实测 2026-10-05 被群怪围堵时的根因对策）
# =============================================================================
#
# 现场数据（125 秒会话）：攻击键 3.0 次/秒 + 击退 0.6 次/秒 + 转向 0.55 次/秒
# ≈ 每秒 4+ 次按键，而游戏的攻击动画期间会忽略按键 → 大部分被吞：
# 表现为"释放不出技能、击退按了没放出来"、转向 35% 未生效。
# 对策：按出攻击键后进入"动画锁"，锁内不发任何攻击/击退/转向键——
# 每次按键都真正落地，反而更快。换边防抖解决群怪两侧导致的原地狂转身。

ATTACK_LOCK_SECONDS_DEFAULT = 0.6
"""攻击动画锁默认时长（秒）：按出一次攻击键（技能/击退）后，视为角色进入
攻击动画，此时间内不再发攻击/击退/转向键——游戏会忽略动画期间的按键，
多发只有害处。实际生效值取配置 attack_lock_seconds（0=关闭）。

校准：太小仍会被吞（等于没锁），太大浪费输出。默认 0.6 对应经典弓的
攻击节奏；如果怪还是压过来就降到 0.5，发现"挥弓不射箭"就加到 0.7。"""

TURN_SIDE_CONFIRM_SECONDS = 0.2
"""换边防抖（秒）：目标从一侧换到另一侧后，新侧需连续稳定此时间才真正转身。

群怪分布在角色两侧时，"最近目标"每帧左右翻 → 每次翻边都转身会让角色原地
狂转（转向键被动画吞掉 + 转向帧堵输出，实测 3 秒转 5 次、3 次白转）。
防抖后瞬时翻边不转身、【也不施法】——严格执行"未确认朝向不出手"，
避免朝旧方向空放（10/25 21:36/21:38 现场）。稳定换边才转。
按 fps 换算成帧；朝向记忆为空（首次定位）时不防抖，立即转身。"""

TURN_VERIFY_WAIT_SECONDS = 0.15
"""自检等待时间（秒）：按下转向键后等这么久（≈一次按住 + 一帧）再判定是否生效。

从 0.35 收到 0.15：方向键按住本身已阻塞 120~180ms，游戏早已处理完这次输入；
下一帧的位置就是"按完后的位置"，不需要再等 0.35s。配合 ≥10px 一帧确认，
转向确认耗时从约 0.6s 降到约 0.25~0.35s。

等待期间如果"朝向确实与目标侧不符"，本帧不施法——对着反方向放技能等于白放，
等转到位再打反而更划算。严格模式下本帧 / 本段等待都不会放技能。"""

FACE_WAIT_LOG_SECONDS = 2.0
"""朝向未确认而暂停施法时，等待告警的日志间隔（秒）。

严格模式下未确认就不出手，可能短暂停手；这条日志会带上已等待时长，
便于现场判断是"按键被吞/定位丢失"还是彻底卡死。"""

FACE_WAIT_COMMIT_SECONDS = 1.0
"""目标侧持续抖动时的最长等待（秒）：超过就不再等"换边稳定"，直接朝当前侧转。

严格模式会在目标左右翻边时停手避免反向施法；但如果两侧怪一直交替成为最近
目标，不能无限停手。等待超过此时长后按当前目标侧发起转向（仍需自检确认
后才施法），保证不会完全不出手。"""

FACE_DIAG_LOG_FRAMES = 30
"""朝向自检日志间隔帧数（约 3 秒 @10fps）：把"朝向记忆 / 目标在哪侧 / 本窗口
转向次数 / 其中自检未生效次数"打进日志。

"锁定目标一直在反侧、而本窗口一次转向都没有"就是朝向脱钩的特征（本次问题的
现场），有这行日志下次一眼就能看出来。"""

STUCK_KEY_IDLE_SECONDS = 1.0
"""站定期间（决策层没按方向键）补发方向键 KEYUP 的间隔（秒），兜底防按键卡住。"""

STUCK_KEY_DRIFT_STEP_X = 3
"""判定"角色在无按键下漂移"的单帧水平位移阈值（像素）。

fps=10 时约等于 30px/s；取 3px 是为了避开精灵换姿势（攻击/待机）导致的
检测框中心抖动（实测同一位置不同姿势的中心可差 ±5px）。"""

STUCK_KEY_DRIFT_FRAMES = 3
"""无按键却连续朝同一方向漂移的帧数：达到就立刻补发方向键 KEYUP。

决策层没按方向键（_held_key 为空）时角色却持续朝一个方向走，最可能是某次
KEYUP 没被游戏收到（SendInput 成功但游戏没采到），游戏侧一直"按住"那个方向。"""

# =============================================================================
# 近身击退（长手专用）
# =============================================================================
#
# 用途：怪贴到触发距离以内时，改放"击退技能"把它推开，推开后继续站定输出。
# 与"贴脸后撤"(_retreat) 的区别：
#   · 不移动 → 不落崖、不走进怪群、不需要再走回目标，站定特征不变；
#   · 击退技能本身带伤害，而后撤期间是不放技能的（纯输出损失）。
# 设计要点（避免出现"一直卡在击退分支"的问题）：
#   · 只替代"本帧这一次技能按键"——真按出击退的那一帧才跳过普通技能；
#     冷却期内返回 False，调用方照常走普通技能选择。因此即使击退被游戏吞掉、
#     对怪免疫（boss）、或怪没被推开，普通攻击也永远不会被挡住；
#   · 不需要追踪"怪有没有被推远"：怪被推开后 dx 自然变大，下一帧就落回普通
#     攻击逻辑（每帧重新决策本身就是闭环）；
#   · 只对长手(远程)生效：短手(近战)的目标就是贴近怪，把怪推开是反效果。

KNOCKBACK_RANGE_DEFAULT = 240
"""击退触发距离默认值（px，角色与怪中心 x 的差）。

取自用户提供的三张参考帧实测（test/mxd_20260923_1234 54 / 123633）：角色朝向侧
最近的同平台怪分别在 224px 与 232px（对应精灵框间隙 115~118px），取 240 覆盖
两个样本。参考：attack_range 通常 400px，击退把怪推远后仍落在射程内，不用追。"""

KNOCKBACK_COOLDOWN_DEFAULT = 0.3
"""击退技能最小间隔默认值（秒）。比技能列表里的冷却短得多，表示"近身时基本
按这个节奏用"，具体填多少以游戏内实际冷却为准（填短了多按的会被游戏丢掉）。"""

# =============================================================================
# 拟人化抖动（降低"输入完全规律"的机器特征，减少被反外挂盯上的概率）
# =============================================================================

HEAL_REACT_SECONDS = 0.5
"""加血/加蓝的"反应延迟"（秒）：血量/蓝量跌破阈值后不立即按键，先延迟此时间。
人类反应时间本就比程序慢且每次不固定，跌破阈值同一帧就按是明显的机器特征。"""

REACT_RESET_SECONDS = 0.3
"""加血/加蓝反应延迟的"复位门槛"（秒）：血/蓝量要连续恢复正常这么多时间，
延迟计数才清零复位。

血/蓝条读数会跳（日志实测：血量 24% 的中途突然读成 99%，一秒内来回 5 次）。
若单帧"正常"就立刻复位，刚装载的 0.5s 反应延迟会被反复重置，
真低血时反而一直等不到按键。"""

SKILL_COOLDOWN_JITTER_MIN = 0.9
"""技能冷却抖动下限倍数：实际冷却 = 技能冷却 × U(此值, 上限)。"""

SKILL_COOLDOWN_JITTER_MAX = 1.15
"""技能冷却抖动上限倍数：避免技能释放间隔恒定（如精确每 0.3s）。"""

MELEE_HIT_TOL_X = 10
"""短手贴脸"命中容差"（像素）。

攻击距离 50px 指攻击特效的有效命中距离。角色到怪近侧身体边缘
≤ 攻击距离+此容差 才判定"能打到"、站定攻击；容差覆盖攻击特效
判定框冗余 + 检测抖动（±10px），避免怪在边缘 50~60px 处时
"判定未贴脸→追→怪又贴近→又未贴脸"的抖动。
超过此容差(>60px)攻击特效够不到，必须追击，不停在原地空打。
"""

MELEE_HYSTERESIS_X = 40
"""短手(近战)攻击中"确认怪物走远"的防抖窗口宽度（像素）。

贴脸命中区 = 攻击距离 + MELEE_HIT_TOL_X（例: 50+10=60px）。
攻击中角色到怪近侧边缘距离超过命中区、但 ≤ 攻击距离+此窗口
（60~90px）时，先用防抖帧数 MELEE_LEAVE_FRAMES 吸收瞬时抖动
（攻击位移/怪物被推开 1~3 帧），连续超限才转追击——避免
"怪被推一下就走远 → 追 → 怪弹回 → 追"的来回抖动。
超过此窗口(>90px)说明怪真走远或已切换目标，立即追击，不停在
原地对着够不着的怪空打。
"""

MELEE_LEAVE_FRAMES = 6
"""短手(近战)攻击中连续超出滞回窗口的帧数阈值（约 0.3 秒 @20fps）。

超过滞回窗口后不立即切追击：攻击位移/怪物被推开通常是 1~3 帧的
瞬时抖动，连续超出该帧数才判定"怪物真走远了"转为追击，避免攻击
状态 1 帧就断、攻击断断续续。防抖期间原地继续攻击（不移动）。
"""

MELEE_EDGE_MAX_HALF_W = 50
"""短手贴脸判定中怪半宽的上限（像素）。

近战攻击命中怪身体任意部位即可，贴脸判定用"角色到怪近侧身体边缘"
的距离（= 怪中心距 - 怪半宽）。普通怪 bbox 半宽约 25~70px，直接减
半宽会让超宽怪（boss/大怪）离老远就判定贴脸；封顶 50px 后最大有效
贴脸距离 = 攻击距离 + 50，既消除大怪穿越、又不会离远就空打。
"""

MELEE_FACE_DEADZONE_X = 25
"""短手攻击前转向的"重叠死区"（像素）。

先扭头再攻击：攻击前判定怪在角色哪一侧，背对怪就按方向键转身再
施法，避免朝反方向空打。但转身按方向键会让角色朝怪移动一小步，
若角色已与怪身体重叠/极近（角色到怪近侧边缘 ≤ 死区），转身会穿过
怪身体造成"左右来回顶"——此时保持当前朝向直接攻击（怪就在身前/
身侧，攻击可命中），不转向。
"""

OCCLUSION_HALF_WIDTH_X = 40
"""遮挡判定水平容差（像素）。

短手贴脸攻击时角色站在怪正前方，角色模型+攻击特效会遮住怪物，
YOLO 置信度骤降被 conf 阈值过滤 → 本帧看不到锁定目标。
角色脚底 x 与怪中心 x 的最大距离约等于攻击距离(attack_range)，
再加角色半宽+怪半宽+检测抖动容差(40px)即视为"角色在怪跟前"，
目标本帧消失 → 判定为被角色遮挡而不是怪真消失。
"""

OCCLUSION_MAX_FRAMES = 300
"""目标被遮挡时沿用最后已知位置的帧数上限（约 10 秒 @30fps）。

超过该帧数仍未被 YOLO 重新看到 → 判定怪真消失（被击退/逃出
画面/已死亡），放弃攻击。正常近战 2~4 秒内击杀或怪露出，
但厚血怪/被击退后角色追错位置时遮挡可持续更久；若上限太短
(150=5 秒)会出现"厚血怪被遮 5 秒 → 判定消失 → 转探索/换远处
目标 → 角色离开 → 怪露出 → 重新贴脸 → 又被遮"的反复空转，
表现为在怪物堆里到处跑不攻击。
"""

SELF_POS_STALE_FRAMES = 60
"""自身定位(OCR)连续失败的帧数上限（约 2 秒 @30fps）。

站定攻击中角色位置不变，短时定位失败（技能特效遮挡角色名字/
怪物名与角色名重叠/OCR 抖动）可用最后已知位置(_last_self_pos)
继续攻击，避免"特效挡名字 → 放弃目标转探索 → 角色离开 →
特效消失 → 重新定位"的反复空转；
超过该帧数仍定位不到 → 判定真丢失，避免用过期坐标乱跑。
"""

BLIND_CHASE_SECONDS = 2.0
"""自身定位连续失败时，按最后已知位置继续移动的时长上限（秒）。

追击时定位一失败就松键，角色会永远停在遮挡物后面（定位失败 →
不移动 → 继续被遮挡），实测 10/23 角色走进装饰铁门/藤蔓后卡死。
但盲走太久又会按过时坐标冲过目标甚至落崖，所以只给 2 秒预算：
够走过栅栏/藤蔓（约 1~2 秒），又不会冲出攻击范围。
超过预算松键等待定位恢复，恢复后立即继续追/打。
"""

# =============================================================================
# 活动范围（只在地图左半边 / 右半边打怪）
# =============================================================================
#
# 镜头跟着角色走，只看画面不知道自己在地图哪里，所以用小地图上的黄点：
# 黄点离分界线多远 × minimap_scale = 画面里分界线离自己多远。
# 比例不准只影响离分界线远时对怪的判断，走到线上时距离是 0，人不会因此过线。

RANGE_LINE_TOLERANCE = 20
"""活动范围：怪物中心过了分界线这么多（px，768 高画面下）以内还算范围内。
小地图 1px 就是画面十几 px，分界线本身有这么大的误差，正好骑在线上的怪不用来回判"""

RANGE_EDGE_MARGIN = 40
"""活动范围：离分界线这么近（px，768 高画面下）就当走到了尽头：探索掉头、后撤不再往那边撤。
松键后还会滑一段，贴到线上才掉头就过线了"""

RANGE_RETURN_START = 60
"""活动范围：过线超过这么多（px，768 高画面下）才走回来，比 RANGE_LINE_TOLERANCE 大，
追线上的怪、滑过一点不会触发。走回到线内 RANGE_EDGE_MARGIN 才停"""

RANGE_RETURN_CONFIRM_FRAMES = 3
"""活动范围：连续过线这么多帧才走回来，定位跳一两帧不算"""

RANGE_RETURN_GIVEUP = 3
"""活动范围：走回时卡住（跳了也过不去）这么多次就先不走，RANGE_RETRY_SECONDS 秒后再试"""

RANGE_RETRY_SECONDS = 5.0
"""活动范围：走回被挡住后过多久再试；这期间照常打范围内够得着的怪"""

RANGE_DOT_GRACE_SECONDS = 1.0
"""活动范围：小地图上的黄点这么久内看到过就沿用（被别的点盖住一下），再久就算不知道自己在哪"""

RANGE_STATUS_SECONDS = 10.0
"""活动范围：每隔这么久写一行当前位置，实机确认分界线对不对"""

RANGE_NOTE_SECONDS = 5.0
"""活动范围：线外有怪不打、认不出位置这类日志的最短间隔"""

EDGE_MARGIN = 80
"""画面边缘区域宽度（px，768 高画面下）。冒险岛镜头跟随角色，只有走到地图
尽头时角色才会贴近画面左右边缘：探索走进这个区域就掉头。"""


@dataclass
class Context:
    """感知层 → 决策层的数据载体（每帧一份）。

    self_position 由 OCR 识别得到（窗口内坐标，脚底）。
    self_center 为角色中心点（名字中心 - 人物高度一半，向上），用于距离推算。
    """
    monsters: List[Detection] = field(default_factory=list)
    floors: List[Detection] = field(default_factory=list)
    ropes: List[Detection] = field(default_factory=list)
    self_position: Optional[Tuple[int, int]] = None
    self_center: Optional[Tuple[int, int]] = None
    hp_ratio: Optional[float] = None
    mp_ratio: Optional[float] = None
    detections: List[Detection] = field(default_factory=list)
    frame_height: Optional[int] = None  # 当前帧高度，用于像素参数随分辨率缩放
    frame_width: Optional[int] = None   # 当前帧宽度，用于判断是否走到地图尽头
    minimap_x: Optional[float] = None   # 小地图上自己的黄点 x（相对框选区域左边，px），None = 这帧没找到
    minimap_width: Optional[int] = None  # 框选的小地图区域宽度（px）


class DecisionEngine:
    """反应式决策引擎：根据画面实时内容决定下一步动作。

    【核心理念】
    不做地图、不做全局规划。每一帧只看 YOLO 检测结果，
    模拟人类玩家的反应模式。

    【状态机】
    使用 FSM 管理 7 个状态：
      IDLE → CHASING → ATTACKING  （同平台追击）
      IDLE → CLIMBING              （爬绳追怪）
      IDLE → DROPPING              （跳下追怪）
      任意 → HEALING / RECOVERING  （生存优先）

    Args:
        config:   全局配置
        executor: 动作执行器
        on_log:   日志回调
    """

    def __init__(self, config: Config, executor: ActionExecutor,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.executor = executor
        self._log = on_log or (lambda m: None)
        self._skill_index = 0

        self._fsm = FSM(on_log=self._log)

        self._target_monster: Optional[Detection] = None
        self._explore_direction = "right"
        self._last_self_pos: Optional[Tuple[int, int]] = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0

        # 移动键按住状态（持续移动/攀爬）
        self._held_key: Optional[str] = None      # 当前按住的键（left/right/up/down）
        self._climbing = False                    # 是否正在沿绳索攀爬
        self._climb_exit_frames = 0               # 脱离绳索后横向走出的剩余帧数
        self._climb_log_count = 0                 # 攀爬日志限频计数
        # ---- 残影判定（出手次数 + 静止判定 + 位置记忆）----
        self._casts_on_target = 0                 # 对当前锁定目标已出手的次数
        self._ghost_anchor: Optional[Tuple[int, int]] = None  # 开始计数时目标的位置，动过就重新计数
        self._ghost_since = 0.0                   # 开始计数的时间
        self._ghost_moved = False                 # 计数期间目标动过（活的）
        # 残影位置：{"x","y","until"(忽略到何时),"count"(被判定次数),"last"(最近判定时间)}
        self._ghost_zones: List[dict] = []
        # 角色朝向：只记录"已被游戏确认"的朝向（按键后实际位移验证过），
        # 未确认的转向只存在 _turn_dir / _face_attempt_side 里，绝不写进来——
        # 否则游戏吞掉转向键时会变成"以为转了、其实没转"（10/25 实测脱钩）。
        self._face_dir: Optional[str] = None      # 已确认的角色朝向（left/right/None=未知）
        # ---- 转向（朝向）状态：与施法分帧 / 定期重申 / 生效自检 ----
        self._face_last_press_at = 0.0            # 上次发出转向按键的时间（最小间隔用）
        self._face_next_assert = 0.0              # 下次允许"重申朝向"的时间
        self._face_attempt_side: Optional[str] = None  # 正在纠正的目标侧（None=无未确认转向）
        self._turn_dir: Optional[str] = None      # 待自检的转向方向（None=无待检）
        self._turn_time = 0.0                     # 发出该转向的时间
        self._turn_x0: Optional[int] = None       # 发出转向时的角色 x（位移自检基准）
        self._turn_verify_hits = 0                # 转向自检连续达标帧数（≥2 才确认）
        self._face_verify_fails = 0               # 连续自检失败次数（决定是否提前重试）
        self._face_assert_count = 0               # 本日志窗口内转向次数
        self._face_unverified_count = 0           # 本日志窗口内自检未生效次数
        self._face_diag_frames = 0                # 朝向自检日志限频计数
        self._stand_x: Optional[int] = None       # 站定期间上一次角色 x（漂移检测）
        self._stand_drift_dir = 0                 # 漂移方向（±1，0=无）
        self._stand_drift_frames = 0              # 连续同向漂移帧数
        self._stuck_release_at = 0.0              # 上次补发方向键 KEYUP 的时间
        self._face_wait_since = 0.0               # 本段"朝向未确认暂停施法"的起始时间（0=未在等）
        self._face_wait_log_at = 0.0              # "暂停施法等待朝向"日志限频
        self._face_edge_skip_log_at = 0.0          # "前方无地板跳过转向"日志限频
        # ---- 攻击动画锁 + 换边防抖 ----
        self._attack_busy_until = 0.0             # 动画锁截止时刻（此前不发攻击/击退/转向键）
        self._face_need_side: Optional[str] = None  # 换边防抖：上一帧"需要的朝向"
        self._face_need_frames = 0                # 该朝向已连续保持的帧数
        self._drift_suppress_until = 0.0          # 转向/微动点按后短暂抑制漂移检测
        self._melee_leave_frames = 0              # 短手攻击中连续超限帧数（防抖计数）
        self._occluded_frames = 0                 # 目标被角色遮挡的连续帧数（虚拟目标维持）
        self._self_pos_stale_frames = 0           # 自身定位连续失败的帧数（最后已知位置时效）
        self._blind_chase_warned = False          # 本次定位丢失是否已打过"盲追"日志
        self._target_wait_warned = False          # 本次定位丢失是否已打过"保留目标等待"日志
        self._nearer_candidate: Optional[Detection] = None  # 更近怪抢占据防抖：候选目标
        self._nearer_frames = 0                   # 该候选连续满足条件的帧数
        self._retreating = False                  # 长手是否正在后撤拉开距离（保持最小射程）
        self._retreat_started_at = 0.0            # 本次后撤开始时间（超时转突围用）
        self._escape_started = 0.0                # 本次翻越突围开始时间，0 = 没在突围
        self._escape_dir: Optional[str] = None    # 突围方向（朝怪物，越过它）
        self._escape_retry_at = 0.0               # 突围没成功时，到这个时间才允许再突围
        self._chase_stuck = 0                     # 追当前目标时连续卡住的次数
        self._chase_stuck_target: Optional[Detection] = None
        self._occluded_retreat_frames = 0         # 长手遮挡后撤的连续帧数（防无限后撤）
        self._prev_hp_ratio: Optional[float] = None  # 上一帧血量（掉血检测）
        self._hp_drop_active_frames = 0           # 掉血信号剩余有效帧数
        self._hp_retreat_hold_frames = 0          # 掉血触发后撤的剩余持续帧数
        self._aoe_burst_left = 0                  # AOE连发剩余次数（爆炸箭二连发）
        self._target_miss_frames = 0              # 锁定目标连续漏检帧数（特效遮挡保持）
        self._buff_hold_frames = 0                # >0 时暂停攻击/移动，给 buff 让路（主循环请求）
        # ---- 站桩定时微动 ----
        self._micro_cooldown_frames = self._frames(
            float(getattr(config, "stand_micro_move_interval", 300.0) or 300.0)
        )                                          # 距下次微动的剩余帧数
        self._hp_react_frames = -1                # 加血反应延迟剩余帧数（-1=未触发）
        self._mp_react_frames = -1                # 加蓝反应延迟剩余帧数（-1=未触发）
        self._hp_react_ok = 0                     # 血量连续正常帧数（抗单帧读数毛刺）
        self._mp_react_ok = 0                     # 蓝量连续正常帧数（抗单帧读数毛刺）

        # ---- 像素参数缩放（活动范围等按 768 高画面填的阈值）----
        self._scale = 1.0
        self._last_cast_at = 0.0                  # 上次真正按下技能键的时间（换频道判"这层没怪"用）
        self._range_reset()

    def _range_reset(self):
        """清空活动范围的状态（开始/测谎恢复/补飞镖之后，人可能已经不在原来的位置）。"""
        self._range_line_x: Optional[float] = None  # 本帧分界线在画面里的 x（可能在画面外），None = 不限制或不知道
        self._range_dot: Optional[Tuple[float, int]] = None  # 最近看到的 (黄点 x, 小地图宽)
        self._range_dot_at = 0.0                  # 最近看到黄点的时间
        self._range_lost = False                  # 正处于"不知道自己在地图哪里"的状态
        self._range_announced = False             # 开始时那行说明写过了
        self._range_status_at = 0.0               # 上次写当前位置日志的时间
        self._range_note_at = 0.0                 # 上次写"线外有怪/认不出位置"日志的时间
        self._range_over_frames = 0               # 连续过线的帧数
        self._range_returning = 0.0               # 开始走回范围的时间，0 = 没在走
        self._range_stuck = 0                     # 这次走回卡住的次数
        self._range_retry_at = 0.0                # 走回被挡住后，到这个时间再试

    def update_config(self, config: Config):
        self.config = config

    def reset(self):
        self._skill_index = 0
        self._target_monster = None
        self._explore_direction = "right"
        self._last_self_pos = None
        self._stuck_counter = 0
        self._explore_frame_count = 0
        self._distance_log_frame_count = 0
        self._casts_on_target = 0
        self._ghost_anchor = None
        self._ghost_since = 0.0
        self._ghost_moved = False
        self._ghost_zones = []
        self._face_dir = None
        self._face_last_press_at = 0.0
        self._face_next_assert = 0.0
        self._face_attempt_side = None
        self._turn_dir = None
        self._turn_time = 0.0
        self._turn_x0 = None
        self._turn_verify_hits = 0
        self._face_verify_fails = 0
        self._face_assert_count = 0
        self._face_unverified_count = 0
        self._face_diag_frames = 0
        self._stand_x = None
        self._stand_drift_dir = 0
        self._stand_drift_frames = 0
        self._stuck_release_at = 0.0
        self._face_wait_since = 0.0
        self._face_wait_log_at = 0.0
        self._face_edge_skip_log_at = 0.0
        self._attack_busy_until = 0.0
        self._face_need_side = None
        self._face_need_frames = 0
        self._drift_suppress_until = 0.0
        self._melee_leave_frames = 0
        self._occluded_frames = 0
        self._self_pos_stale_frames = 0
        self._blind_chase_warned = False
        self._target_wait_warned = False
        self._nearer_candidate = None
        self._nearer_frames = 0
        self._retreating = False
        self._retreat_started_at = 0.0
        self._escape_started = 0.0
        self._escape_dir = None
        self._escape_retry_at = 0.0
        self._chase_stuck = 0
        self._chase_stuck_target = None
        self._occluded_retreat_frames = 0
        self._prev_hp_ratio = None
        self._hp_drop_active_frames = 0
        self._hp_retreat_hold_frames = 0
        self._aoe_burst_left = 0
        self._target_miss_frames = 0
        self._buff_hold_frames = 0
        self._hp_react_frames = -1
        self._mp_react_frames = -1
        self._hp_react_ok = 0
        self._mp_react_ok = 0
        self._last_cast_at = 0.0
        self._scale = 1.0
        self._range_reset()
        # 站桩微动：让第一次微动在间隔之后才发生
        self._micro_cooldown_frames = self._frames(self._micro_interval())
        self.release_keys()
        self._fsm.reset()
        self.executor.reset()

    def release_keys(self):
        """释放所有按住的移动键（停止时调用，防止方向键卡住）。"""
        self._release_move()
        self._climbing = False
        self._climb_exit_frames = 0
        self._climb_log_count = 0
        self._retreating = False

    @property
    def state_name(self) -> str:
        """当前状态名（供 UI 显示）。"""
        return self._fsm.state_name

    @property
    def last_cast_at(self) -> float:
        """上次真正按出技能键的时间戳（没放过技能为 0）。

        自动换频道用它判断"这层是不是几分钟没怪了"。
        """
        return self._last_cast_at

    # =========================================================================
    # 决策主入口
    # =========================================================================

    def decide(self, ctx: Context):
        """每帧调用一次，根据画面内容执行动作。

        Args:
            ctx: 当前帧的感知数据
        """
        self._fsm.tick()
        self._update_scale(ctx)

        # 检测卡住 + 自身定位时效统计
        if ctx.self_position:
            self._self_pos_stale_frames = 0
            self._blind_chase_warned = False
            self._target_wait_warned = False
            if self._last_self_pos and self._last_self_pos == ctx.self_position:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            self._last_self_pos = ctx.self_position
        else:
            # 定位失败帧计数：站定攻击中用最后已知位置兜底的时效依据
            self._self_pos_stale_frames += 1

        # 站桩模式：位置恒定是预期，不判"卡住"（否则会触发卡住跳跃）
        if self._is_stand_mode():
            self._stuck_counter = 0

        # ---- 方向键卡住兜底：自己没按方向键时补发 KEYUP ----
        # 放在决策之前：万一游戏侧"按住"了某个方向键（KEYUP 丢失），
        # 角色会一直朝一个方向走且朝向错，本帧先把它清掉再决策。
        self._stuck_direction_guard(ctx)

        # ---- 掉血检测（长手遮挡兜底信号）：本帧血比上帧显著下降 → 记最近掉血 ----
        if ctx.hp_ratio is not None:
            if self._prev_hp_ratio is not None \
                    and ctx.hp_ratio < self._prev_hp_ratio - HP_DROP_THRESHOLD:
                self._hp_drop_active_frames = self._frames(HP_DROP_ACTIVE_SECONDS)
            self._prev_hp_ratio = ctx.hp_ratio
        if self._hp_drop_active_frames > 0:
            self._hp_drop_active_frames -= 1

        # ---- 优先级 1: 没血加血（跌破阈值后延迟 ~0.5s 再按，模拟人类反应时间）----
        hp_low = (ctx.hp_ratio is not None
                  and ctx.hp_ratio < self.config.hp_threshold
                  and ctx.hp_ratio < 0.95)   # 满状态(>=95%)不触发，防止刚加完又按
        self._hp_react_frames, self._hp_react_ok = self._tick_react_delay(
            self._hp_react_frames, self._hp_react_ok, hp_low)
        # 只在"血瓶真能按下"时才切 HEALING 并收手：press_key 带 1.5s 冷却，
        # 冷却期内它返回 False。若此时仍然切状态，就会每帧 HEALING↔ATTACKING
        # 翻一次（血低时冷却一直在，等于每帧翻）；而直接 return 又会让角色在
        # 等冷却的 1.5s 里完全停手、不输出。冷却期内不切状态也不 return
        # → 继续正常战斗，血瓶一好立刻按。
        if hp_low and self._hp_react_frames == 0 \
                and self.executor.can_press(self.config.hp_key, cooldown=1.5):
            self._release_move()  # 加血时站住不动
            self._fsm.transition(State.HEALING)
            if self.executor.press_key(self.config.hp_key, cooldown=1.5):
                self._log(
                    f"[加血] HP={ctx.hp_ratio:.0%} < {self.config.hp_threshold:.0%}，"
                    f"按下 {self.config.hp_key}"
                )
            return

        # ---- 优先级 2: 没蓝加蓝（同样带 0.5s 反应延迟）----
        mp_low = (ctx.mp_ratio is not None
                  and ctx.mp_ratio < self.config.mp_threshold
                  and ctx.mp_ratio < 0.95)
        self._mp_react_frames, self._mp_react_ok = self._tick_react_delay(
            self._mp_react_frames, self._mp_react_ok, mp_low)
        # 与加血同理：只在蓝瓶真能按下时才切 RECOVERING 并收手，
        # 冷却期内继续正常战斗（否则同样会每帧 RECOVERING↔ATTACKING 翻转）。
        if mp_low and self._mp_react_frames == 0 \
                and self.executor.can_press(self.config.mp_key, cooldown=1.5):
            self._release_move()  # 加蓝时站住不动
            self._fsm.transition(State.RECOVERING)
            if self.executor.press_key(self.config.mp_key, cooldown=1.5):
                self._log(
                    f"[加蓝] MP={ctx.mp_ratio:.0%} < {self.config.mp_threshold:.0%}，"
                    f"按下 {self.config.mp_key}"
                )
            return

        # ---- buff 施法窗口：暂停攻击/移动，给 buff 让路（加血/加蓝不受影响）----
        # 主循环检测到 buff 到期时，通过 request_buff_window() 打开此窗口，
        # 避免按 buff 键的瞬间角色正在攻击动画中 → 游戏忽略该按键 → buff 加不上。
        if self._buff_hold_frames > 0:
            self._buff_hold_frames -= 1
            self._release_move()
            return

        # ---- 站桩定时微动：每过 N 秒反方向挪 DISTANCE 像素再回原位（朝向不变）----
        # 目的：反"定点一动不动"这类最容易被反外挂盯上的特征。
        # 位置放在加血/加蓝/buff 之后（保命优先）、怪物处理之前：
        # 微动期间这一帧不攻击（和 buff 施法窗口同一个套路）。
        if self._micro_move_tick(ctx):
            return

        # ---- 活动范围：去掉线外的怪、走出范围先走回来 ----
        # 放在怪物处理之前：之后选目标/追击/后撤/探索都看不到线外的怪。
        # 站桩模式只过滤不走动（站位自己选在范围内）。
        if self._range_step(ctx):
            return

        # ---- 优先级 3: 检测到怪物 ----
        if ctx.monsters:
            self._handle_monsters(ctx)
        elif self._in_combat() and self._target_monster is not None:
            # 战斗中整帧看不到任何怪：短手贴脸时角色+攻击特效可能把怪
            # 完全遮住 → YOLO 整帧漏检。先走遮挡判定维持虚拟目标继续攻击，
            # 避免"怪消失 → 转探索乱走 → 怪露出 → 重选目标"的反复空转。
            occluded = self._occluded_target(ctx)
            if occluded is not None:
                if getattr(self.config, "attack_type", "long") == "short":
                    self._handle_melee(ctx, occluded)
                else:
                    self._attack(ctx, occluded)
                return
            # 长手：远程技能特效遮挡 → 保持锁定继续攻击（不后撤、不探索）
            retained = self._retain_missed_target(ctx)
            if retained is not None:
                self._dispatch_retained(ctx, retained)
                return
            # 长手：目标消失但最后位置很近 → 被遮挡 → 后撤；或掉血兜底后撤
            if self._handle_no_monster_retreat(ctx):
                return
            self._fsm.transition(State.IDLE)
            # 画面中已没有怪物：立即解除锁定并清理攀爬等残留状态，
            # 防止"上帧还锁着怪/在爬绳"的状态影响后续探索与重新选怪
            self._target_monster = None
            self._casts_on_target = 0
            self._occluded_retreat_frames = 0
            self._aoe_burst_left = 0
            self._target_miss_frames = 0
            self._climbing = False
            self._climb_exit_frames = 0
            self._retreating = False
            self._explore(ctx)
        else:
            # 无怪且未锁定目标：掉血兜底后撤（怪可能在贴脸打你）
            if self._handle_no_monster_retreat(ctx):
                return
            self._fsm.transition(State.IDLE)
            # 画面中已没有怪物：立即解除锁定并清理攀爬等残留状态，
            # 防止"上帧还锁着怪/在爬绳"的状态影响后续探索与重新选怪
            self._target_monster = None
            self._casts_on_target = 0
            self._occluded_retreat_frames = 0
            self._aoe_burst_left = 0
            self._target_miss_frames = 0
            self._climbing = False
            self._climb_exit_frames = 0
            self._retreating = False
            self._explore(ctx)

    # =========================================================================
    # 怪物处理
    # =========================================================================

    def _handle_monsters(self, ctx: Context):
        """处理画面中的怪物（就近攻击）。

        【就近原则】每帧重新选择附近最近的怪物，谁在附近打谁：
          1. 怪物消失/离开画面后，下一帧自动选到别的怪，不用等
          2. 【残影防护】对同一目标出手 ghost_casts 次（且超过 3 秒）仍未消失
             → 判定为死尸残影/误识别，把该位置记入忽略表，跳过它重新选目标，
             避免原地打空气；动过的目标要 2 倍次数/时间才判定（活怪打不到）
        """
        target = self._resolve_locked_target(ctx)

        # ---- 无有效目标（没有同平台怪 / 全在残影忽略位置）----
        # 不走 CRASH 路径，释放移动键并转探索，避免角色卡在上一帧的移动状态
        if target is None:
            # 攻击中目标短暂从画面消失（YOLO 单帧漏检/波动）→ 沿用旧锁定
            # 目标继续攻击。若此时重选别的怪，dx 会瞬间翻转、角色左右乱转。
            # 若怪真被杀/消失，残影检测会兜底换目标。
            if self._fsm.current == State.ATTACKING and self._target_monster is not None:
                target = self._target_monster
            else:
                self._target_monster = None
                self._casts_on_target = 0
                self._release_move()
                self._fsm.transition(State.IDLE)
                self._explore(ctx)
                return

        # ---- 残影防护：同一个目标打了 ghost_casts 下还在 → 残影/误识别 ----
        # 换了目标（不是同一只）就重新计数；遮挡/漏检宽限期间的虚拟目标沿用计数
        if self._occluded_frames == 0 and (self._target_monster is None
                                           or not self._is_same_monster(self._target_monster, target)):
            self._casts_on_target = 0
        if self._occluded_frames == 0:
            # 只有一直不动的目标才可能是残影：动过就是活的，重新计数
            c = target.center
            a = self._ghost_anchor
            if a is None or self._casts_on_target == 0:
                self._ghost_anchor = c
                self._ghost_since = time.time()
                self._ghost_moved = False
            elif abs(c[0] - a[0]) + abs(c[1] - a[1]) > self._px(GHOST_STATIC_PX):
                self._ghost_moved = True  # 动过：是活的，要打更久才算"打不到"
        limit = int(getattr(self.config, "ghost_casts", GHOST_CASTS) or 0)
        # 不动的：打 limit 下、3 秒还在 → 残影/误识别；
        # 动过的（活怪）：打 2×limit 下、6 秒还在 → 多半打不到（比如飘在高一截的地方）
        k = 2 if self._ghost_moved else 1
        if limit > 0 and self._casts_on_target >= limit * k \
                and time.time() - self._ghost_since >= GHOST_MIN_SECONDS * k:
            seconds, count = self._mark_ghost(target)
            self._log(f"[换目标] 对同一目标出手 {self._casts_on_target} 次仍未消失"
                      f"（疑似残影/误识别），第 {count} 次判定，{seconds:.0f} 秒内忽略该位置")
            self._casts_on_target = 0
            self._ghost_anchor = None
            self._target_monster = None
            # 该位置已进忽略表，重新选目标时会被过滤掉
            target = self._pick_best_target(ctx)
            if target is None:
                # 画面里只剩打不死的残影 → 不空耗，转探索
                self._fsm.transition(State.IDLE)
                self._explore(ctx)
                return

        self._set_locked_target(target)

        # ---- 短手（近战）走独立攻击流程，长手走远程流程 ----
        # 近战判定与远程完全不同：必须贴脸才打，未贴脸就径直追上，
        # 攻击中怪物走远立即转为追击，没有远程的"站定攻击+大滞回"。
        if getattr(self.config, "attack_type", "long") == "short":
            self._handle_melee(ctx, target)
            return

        mx, my = target.center

        # ---- 攻击判定（规则1/2/3/4）----
        # 同一平台 = 角色脚底y 与 怪物脚底y(bbox底部) 的垂直差 ≤ 垂直容差
        #   —— 脚底对"是否站在同一条地面线"最准确。
        #      若用"中心点y"对比，高大怪物会被误判为跨层(见 _pick_best_target)。
        # 攻击距离 = 水平方向 |角色x - 怪物中心x| ≤ 攻击距离px（规则2）
        foot = self._effective_self_pos(ctx)
        if foot is None:
            # 定位失败超过 5s（技能特效/场景遮挡导致 YOLO 看不到 player）。
            # 有最后已知位置时不要丢目标转随机探索：_explore 的方向可能和怪
            # 相反，会把角色越带越远。保留锁定目标停在原地等定位恢复，
            # 恢复后立刻继续追/打；盲走预算已在 _chase 里用尽，这里不移动。
            if self._last_self_pos is not None:
                if not self._target_wait_warned:
                    self._target_wait_warned = True
                    secs = self._self_pos_stale_frames / max(1, self.config.fps)
                    self._log(
                        f"[目标] 自身定位丢失 {secs:.1f}s，保留目标等待定位恢复"
                        f"（不随机探索，疑似被场景遮挡/模型漏检）"
                    )
                self._release_move()
                self._fsm.transition(State.CHASING)
                return
            # 从启动起就没定位到过：无法判断距离，不能攻击，只能探索
            self._target_monster = None
            self._casts_on_target = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return
        px, py = foot
        monster_foot = (mx, target.y + target.h)

        # 路径推算：人物脚底 / 怪物 bbox 底部，与 YOLO 平台/绳索检测框
        # 的 y 语义一致，只用于"怎么走"。
        est = estimate_path_distance(
            foot, monster_foot, ctx.floors, ctx.ropes,
            same_level_tolerance=getattr(self.config, "attack_range_y", 60),
        )

        # 距离日志（限频输出，避免刷屏）
        self._distance_log_frame_count += 1
        if self._distance_log_frame_count >= DISTANCE_LOG_FRAMES:
            self._distance_log_frame_count = 0
            vy = abs(py - monster_foot[1])
            same_plat = self._same_platform(py, monster_foot[1])
            plan_str = ""
            if est.path_type == "jump" and est.path_floors:
                seq = "→".join(
                    f"({f.center[0]},{f.y})" for f in est.path_floors[:6]
                )
                plan_str = f" 路线平台: {seq}"
            elif est.path_type == "rope" and est.climb_rope is not None:
                r = est.climb_rope
                plan_str = f" 绳索: ({r.center[0]},{r.y}) 长{r.h}"
            self._log(
                f"[距离] 人物脚底({px},{py}) → 怪脚底({monster_foot[0]},{monster_foot[1]}) "
                f"同平台={'是' if same_plat else '否'}"
                f"(垂直差{vy} 容差{getattr(self.config, 'attack_range_y', 60)}) "
                f"路径={est.path_type} 距离={est.distance}px "
                f"(水平={abs(px - mx)})"
                f" [{('短手' if getattr(self.config, 'attack_type', 'long') == 'short' else '长手')}"
                f"有效攻击距={self._get_attack_range()}px]"
                + (f" 绳长={est.rope_length}" if est.path_type == "rope" else "")
                + (f" 跳数={est.jump_count}" if est.path_type == "jump" else "")
                + f"){plan_str}"
            )

        # 同一平台（脚底垂直差 ≤ 容差）：
        #   · 水平差 < 最小射程 → 后撤拉开距离（防弓箭手贴脸挥弓，仅长手）
        #   · 最小射程 ≤ 水平差 ≤ 攻击距离 → 站定攻击（不移动、不乱跑）
        #   · 否则 → 朝怪物直线移动逼近，进入攻击距离后再打
        if self._same_platform(py, monster_foot[1]):
            # 贴脸判定看"同一层离自己最近的怪"，不只是锁定目标：
            # 锁定目标可能在射程边缘，而另一只已经贴到脸上。
            min_range = self._get_min_attack_range()
            threat = self._nearest_threat(ctx, target) or target
            tdx = abs(px - threat.center[0])
            want_retreat = (min_range > 0 and not self._is_stand_mode()
                            and self._should_retreat(tdx, min_range))
            if not want_retreat:
                # 不再需要后撤 → 结束可能还在进行的突围（否则下次后撤会沿用旧突围状态）
                self._escape_started = 0.0
                self._escape_dir = None
            if want_retreat:
                self._fsm.transition(State.CHASING)
                self._retreat(ctx, threat, reason="贴脸挥弓")
            elif self._can_attack(ctx, target):
                self._retreating = False
                self._fsm.transition(State.ATTACKING)
                self._attack(ctx, target)
            elif time.time() - self._last_cast_at < CHASE_PATIENCE_SECONDS \
                    and self._monster_nearby(ctx, px, py):
                # 刚出过手、附近还有怪：先原地等一下，别马上跑去追远处的
                # （怪多/有人抢怪时，追一半又掉头，来回白跑）
                self._retreating = False
                self._release_move()
            else:
                self._retreating = False
                self._fsm.transition(State.CHASING)
                self._chase(ctx, target)
            return

        # 不同平台：直接放弃该目标，只打同平台怪物。
        # 不爬绳、不跳跃、不兜底追击——目标在另一层时当前层打不到，
        # 跨层追过去成本高且容易卡地形。释放移动转探索，
        # 等画面里出现同平台怪物再攻击。
        self._target_monster = None
        self._casts_on_target = 0
        self._release_move()
        self._fsm.transition(State.IDLE)
        self._explore(ctx)

    # =========================================================================
    # 短手（近战）独立攻击流程
    # =========================================================================

    def _handle_melee(self, ctx: Context, target: Detection):
        """短手（近战）独立攻击判定。

        与长手（远程）完全不同，核心是【贴脸】:
          - 未贴脸（到怪近侧身体边缘距离 > 攻击距离）→ 不攻击，径直走向怪物贴身
          - 贴脸（边缘距离 ≤ 攻击距离 且同平台）→ 停止移动，转向 + 攻击
          - 攻击中怪物走远 → 下一帧判定未贴脸 → 立即转为追击，不停在原地空打
        没有远程那套"站定攻击 + 大滞回窗口"的逻辑。
        攻击距离固定 MELEE_ATTACK_RANGE_X(50px)，不随 exe 配置变化，
        距离判定用"角色到怪近侧身体边缘"，宽怪站旁边就打、不穿越。

        贴脸命中区 = 攻击距离 + MELEE_HIT_TOL_X（50+10=60px），超过
        即攻击特效够不到，立即追击不原地空打。攻击中(ATTACKING)超出
        命中区但 ≤ 攻击距离+MELEE_HYSTERESIS_X（60~90px）时用防抖帧数
        MELEE_LEAVE_FRAMES 吸收瞬时抖动（攻击位移/怪物被推开 1~3 帧），
        连续超限才转追击；超过 90px 说明怪真走远/已换目标，立即追击。
        """
        if target is None:
            self._target_monster = None
            self._casts_on_target = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return

        foot = self._effective_self_pos(ctx)
        if foot is None:
            # 无法定位自身 → 默认无法判断贴脸，原地等待下一帧定位，
            # 不切 IDLE 不转探索，避免 OCR 定位"时有时无"导致的高频抖动。
            # 例外：目标仍在最后已知位置的贴脸范围内（近战贴脸怪基本不动，
            # 角色站定没走远）→ 直接用最后位置继续攻击，避免
            # "怪已经连上(贴脸)但 OCR 恰好失败 → 角色干等不攻击"。
            if self._last_self_pos is not None \
                    and self._self_pos_stale_frames <= SELF_POS_STALE_FRAMES \
                    and abs(self._last_self_pos[0]
                            - self._melee_edge_x(self._last_self_pos[0], target)) \
                        <= self._get_attack_range():
                foot = self._last_self_pos
            else:
                self._release_move()
                return

        sx, sy = foot
        # 贴脸距离用"角色到怪近侧身体边缘"，不用怪中心：
        # 宽怪站怪旁边（离怪身体 0~攻击距离）就能攻击，不穿越怪身体。
        dx = abs(sx - self._melee_edge_x(sx, target))
        dy = abs(sy - (target.y + target.h))
        melee_range = self._get_attack_range()
        ry = getattr(self.config, "attack_range_y", 60)

        # 不同平台：放弃该目标（与长手一致，不跨层）
        if dy > ry:
            self._target_monster = None
            self._casts_on_target = 0
            self._release_move()
            self._fsm.transition(State.IDLE)
            self._explore(ctx)
            return

        # 贴脸命中判定：
        # 角色到怪近侧边缘 ≤ 攻击距离+命中容差 → 站定攻击（特效够得到）。
        # 攻击中超出命中区但 ≤ 攻击距离+滞回窗口 → 防抖帧数吸收瞬时
        # 抖动后仍继续攻击；明显超限(目标切换/怪真走远) → 立即追击，
        # 不停在原地对着够不着的怪空打。
        melee_hit = melee_range + MELEE_HIT_TOL_X
        if dx > melee_hit:
            if self._fsm.current == State.ATTACKING \
                    and dx <= melee_range + MELEE_HYSTERESIS_X:
                # 攻击中超出命中区但还在防抖窗口内：先用防抖帧数吸收
                # 瞬时抖动（攻击位移/怪物被推开 1~3 帧）。期间【原地继续
                # 攻击、绝不移动】——攻击动画期间按方向键会边打边跑、
                # 追过头穿越怪物，表现为"贴着怪左右跑"。连续超出
                # MELEE_LEAVE_FRAMES 帧才判定怪物真走远，转追击。
                self._melee_leave_frames += 1
                if self._melee_leave_frames < MELEE_LEAVE_FRAMES:
                    self._melee_attack(ctx, target)
                    return
            self._melee_leave_frames = 0
            self._fsm.transition(State.CHASING)
            self._melee_chase(ctx, target)
            return

        # 已贴脸命中 → 转向 + 攻击（回到命中区即清零防抖计数）
        self._melee_leave_frames = 0
        self._fsm.transition(State.ATTACKING)
        self._melee_attack(ctx, target)

    def _melee_chase(self, ctx: Context, target: Detection):
        """近战追击：径直走向怪物，进入攻击距离即停手攻击。

        与长手语义一致：角色到怪近侧身体边缘 ≤ 攻击距离(50px) 就开始
        攻击，不需要走到怪物脸上（不追到 攻击距离-5px）。
        """
        if ctx.self_position is None:
            # 定位暂时失败（遮挡/漏检）不要立刻松键：用最后已知位置继续追一段，
            # 否则角色会永远停在遮挡物后面（定位失败 → 不移动 → 继续被遮挡）。
            if self._last_self_pos is None or not self._blind_move_allowed():
                self._release_move()
                return
            self._log_blind_move("近战追击")
            sx = self._last_self_pos[0]
        else:
            sx = ctx.self_position[0]
        # 追击目标点 = 怪近侧身体边缘（走到离怪身体 攻击距离-5px 处停下）
        tx = self._melee_edge_x(sx, target)
        melee_range = self._get_attack_range()

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[近战] 卡住了，尝试跳跃")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 进入攻击距离即停手攻击，不走到怪物脸上。
        # 停点 = 攻击距离(50px)；命中容差(+10px)保证停住后攻击能命中，
        # 不会在边界"差一步打不到 → 蹭一下 → 又超距"来回蹭。
        stop_at = max(5, melee_range)
        if tx > sx + stop_at:
            self._hold_move("right")
        elif tx < sx - stop_at:
            self._hold_move("left")
        else:
            self._release_move()

    def _melee_attack(self, ctx: Context, target: Detection):
        """近战攻击：先判定怪在哪边扭头面向它，再站定释放技能。

        核心：每次攻击前判定怪物相对角色的方位（看怪中心 x 的符号），
        只要怪在角色背后就按方向键转身——先扭头再攻击，避免朝反方向
        空打。转向走 _face_step（按住 120~180ms + 定期重申 + 生效自检，
        理由见该方法说明），因此【转向那一帧不放技能】。
        转身会让角色朝怪移动一小步，因此两个例外【不转向】:
          - 角色已与怪身体重叠/极近（edge_dx ≤ MELEE_FACE_DEADZONE_X）：
            转身会穿过怪身体左右来回顶，保持原朝向直接攻击（怪就在
            身前/身侧，攻击可命中）。
          - 防抖窗口内(_melee_leave_frames>0)/遮挡虚拟目标期间
            (_occluded_frames>0)：怪刚走远正在确认、或位置是最后的，
            转向=边打边追穿越怪，表现为"贴着怪左右晃动"。
        """
        self._release_move()  # 攻击时站定
        self._stuck_counter = 0  # 站定攻击不算"卡住"（位置不变是正常的）

        # 攻击动画锁：动画中不发攻击/转向键（发了也会被游戏吞）
        if self._attack_busy():
            return

        # 先判定怪在哪边（怪中心 x 相对角色 x 的符号），背对怪就扭头。
        # 方向看怪中心（符号决定面朝哪侧），距离看怪近侧身体边缘
        # （决定能否安全转身）：贴脸死区内不按方向键（会朝怪迈步/穿怪）。
        if target is not None and self._melee_leave_frames == 0 \
                and self._occluded_frames == 0:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                self._cast_attack_skill()
                return
            sx = foot[0]
            edge_dx = abs(sx - self._melee_edge_x(sx, target))
            allow_press = edge_dx > MELEE_FACE_DEADZONE_X
            if self._face_step(ctx, target, allow_press=allow_press):
                return  # 本帧只转向，不放技能


        self._cast_attack_skill()

    # =========================================================================
    # 按住移动
    # =========================================================================

    def _hold_move(self, direction: str):
        """按住方向键持续移动。

        切换方向时先释放旧键再按住新键，避免两个方向键同时按下。
        移动期间不松手，直到调用 _release_move() 停止。
        站桩模式：直接 return（不做任何位移，也不更新朝向）。
        """
        if self._is_stand_mode():
            return
        if direction not in ("left", "right", "up", "down"):
            return
        if direction in ("left", "right"):
            self._face_dir = direction  # 移动方向即角色朝向
            # 按住方向键移动一定会被游戏采样到 → 朝向可信，清掉待自检状态
            self._face_note_movement(direction)
        if self._held_key == direction:
            return
        if self._held_key:
            self.executor.key_up(self._held_key)
        self.executor.key_down(direction)
        self._held_key = direction

    def _release_move(self):
        """释放当前按住的移动键（停止移动/攀爬）。"""
        if self._held_key:
            self.executor.key_up(self._held_key)
            self._held_key = None

    # =========================================================================
    # 转向（朝向）：与施法分帧 + 定期重申 + 生效自检
    # =========================================================================

    def _face_needed(self, sx: int, target: Detection) -> Optional[str]:
        """目标在角色的哪一侧（±FACE_TURN_X 死区内返回 None）。

        死区内的怪基本在角色正上方/重叠，任何朝向都能打到，不需要转向。
        """
        dx = target.center[0] - sx
        if dx > FACE_TURN_X:
            return "right"
        if dx < -FACE_TURN_X:
            return "left"
        return None

    def _turn_tap(self, direction: str) -> float:
        """朝 direction 按住一小段（转向专用，同步阻塞）。

        与 press_key 的区别：press_key 是 30~90ms 的随机短按，且按下/抬起之间
        还会被同帧的技能键抢走采样窗口；这里按住 120~180ms，让游戏稳稳吃到
        这次方向输入（游戏会忽略攻击动画期间的按键，见模块顶部说明）。

        Returns:
            实际按住的秒数；0 表示按键没发出去（键无效/窗口未锁定）。
        """
        # 先强制释放左右两个方向键：上一次 KEYUP 若被游戏漏采，游戏侧会一直
        # 按住反向键，反向键会盖掉这次点按（实测 10/25 13:52 左转连续 3 次
        # 不生效、角色被带着往右走）。KEYUP 不会改变游戏内朝向，补发是安全的。
        self._force_release_directions()
        seconds = random.uniform(TURN_TAP_MIN_SECONDS, TURN_TAP_MAX_SECONDS)
        if not self.executor.key_down(direction):
            return 0.0
        time.sleep(seconds)
        self.executor.key_up(direction)
        # 再补一次 force KEYUP：本次 KEYUP 同样可能丢，丢了就会变成卡键漂移
        self.executor.key_up(direction, force=True)
        # 点按本身会让角色朝该方向挪几像素（预期行为），短暂抑制漂移检测，
        # 避免被"无按键却持续漂移"卡键兜底误判（实测一局误触发 15 次）
        self._drift_suppress_until = time.time() + 0.5
        return seconds

    def _face_clear_pending(self):
        """清掉"待自检的转向"状态。"""
        self._turn_dir = None
        self._turn_time = 0.0
        self._turn_x0 = None
        self._turn_verify_hits = 0

    def _face_note_movement(self, direction: str):
        """按住方向键移动已真实改变朝向 → 清自检、推迟下一次重申。

        移动（key_down 按住）一定被游戏采样到，是"朝向可信"的最强证据；
        移动期间不需要重申，否则每走一步就打一次转向按键。
        """
        self._face_clear_pending()
        self._face_attempt_side = None
        self._face_verify_fails = 0
        self._face_next_assert = time.time() + random.uniform(
            TURN_REASSERT_MIN_SECONDS, TURN_REASSERT_MAX_SECONDS)

    def _face_settle(self, ctx: Context) -> bool:
        """判定上一轮转向按键是否被游戏接收（用"角色有没有朝该方向动几像素"）。

        【只有生效的转向才算朝向】_face_dir 只在这里（以及真实移动时）写入；
        按出方向键时不再乐观地改记忆。游戏吞键时 _face_dir 保持旧值 →
        调用方（_face_step）会继续暂停施法并按间隔重试，而不是朝旧方向打。

        【防误确认】位移需连续 TURN_VERIFY_CONFIRM_HITS 帧达到
        TURN_VERIFY_MIN_MOVE_X(6px) 才确认——单帧 YOLO 框抖动/被击退一帧
        不会再被误判成"转向成功"（10/25 21:36 现场：2px 阈值下记忆=left、
        实际朝 right，随后十几秒没有纠正）。

        【定位丢失】自检期间没定位 → 先额外等一会儿；仍不恢复就丢掉待检状态
        （不写 _face_dir），返回 True 让调用方继续停手，等定位恢复后重试。

        Returns:
            True 表示"仍在等待/无法判定生效"（调用方在朝向与目标侧不符时
            必须暂停施法）。
        """
        if self._turn_dir is None or self._turn_x0 is None:
            return False
        if ctx.self_position is None:
            # 本帧没定位 → 无法判断这次转向是否生效。先多等一会儿让定位恢复；
            # 仍不恢复就丢掉待检状态（不写 _face_dir），后续由重试/盲点继续纠正。
            if time.time() - self._turn_time \
                    < TURN_VERIFY_WAIT_SECONDS + TURN_VERIFY_MISSING_GRACE_SECONDS:
                return True
            direction = self._turn_dir
            self._face_clear_pending()
            self._face_verify_fails += 1
            self._face_unverified_count += 1
            self._log_face_wait(f"转向 {direction} 无法确认（自身定位丢失）")
            return True
        dx = ctx.self_position[0] - self._turn_x0
        moved = dx if self._turn_dir == "right" else -dx
        if moved >= TURN_VERIFY_MIN_MOVE_X:
            self._turn_verify_hits += 1
            # 大幅位移（≥10px）一帧即确认；6~9px 的边界情况仍要连续两帧，
            # 防止检测抖动误确认。多数转向点按会移动 10~20px，少等一帧。
            need_hits = 1 if moved >= TURN_VERIFY_FAST_MOVE_X \
                else TURN_VERIFY_CONFIRM_HITS
            if self._turn_verify_hits >= need_hits:
                # 生效：方向键被游戏接收（游戏内朝向此时已置为该侧）→ 写入记忆
                old = self._face_dir
                self._face_dir = self._turn_dir
                self._face_attempt_side = None
                self._face_clear_pending()
                self._face_verify_fails = 0
                self._face_next_assert = time.time() + random.uniform(
                    TURN_REASSERT_MIN_SECONDS, TURN_REASSERT_MAX_SECONDS)
                if old != self._face_dir:
                    self._log(
                        f"[朝向] 转向 {self._face_dir} 已确认生效"
                        f"（位移 {moved}px / {self._turn_verify_hits} 帧）"
                    )
                return False
            return True       # 边界位移，再等一帧确认；期间不施法
        self._turn_verify_hits = 0
        if time.time() - self._turn_time < TURN_VERIFY_WAIT_SECONDS:
            return True       # 还没到判定时刻（按住本身也有时长），再等一帧
        # 判定：一点没动 → 这次按键大概率被吞了（技能动画输入锁/按键丢失）
        direction = self._turn_dir
        was_assert = direction == self._face_dir   # 重申（朝向本已确认）失败
        self._face_clear_pending()
        self._face_verify_fails += 1
        self._face_unverified_count += 1
        if was_assert:
            # 重申只是健康检查：点按被吞不代表朝向错了。稍后重试即可。
            self._face_next_assert = time.time() + random.uniform(
                TURN_RETRY_MIN_SECONDS, TURN_RETRY_MAX_SECONDS)
        elif self._held_key is None:
            # 纠正朝向失败：顺手补发左右 KEYUP，清掉可能卡住的反向键
            # （卡住的反向键会持续盖掉左/右转向点按）。
            self._force_release_directions()
        if self._face_verify_fails in (1, TURN_VERIFY_MAX_FAILS):
            self._log(
                f"[朝向] 转向 {direction} 未被游戏接收"
                f"（角色未朝该方向移动），"
                f"{'稍后重申' if was_assert else '暂停施法并重试'}"
                f"（连续第 {self._face_verify_fails} 次）"
            )
        return False

    def _face_diag(self, ctx: Context, target: Detection, sx: int):
        """朝向自检日志（限频）：朝向记忆 / 目标在哪侧 / 转向次数 / 未生效次数。

        排查用：如果日志里长期是"目标在右"而"朝向记忆=left"、且转向次数为 0，
        说明朝向记忆和游戏内朝向已经脱钩（本次问题的现场特征）。
        """
        self._face_diag_frames += 1
        if self._face_diag_frames < FACE_DIAG_LOG_FRAMES:
            return
        self._face_diag_frames = 0
        side = "右" if target.center[0] > sx else "左"
        memory = self._face_dir or "未知"
        if self._face_attempt_side is not None and self._face_attempt_side != self._face_dir:
            memory += f"(未确认→{self._face_attempt_side})"
        self._log(
            f"[朝向自检] 朝向记忆={memory} 目标在{side}"
            f"(dx={target.center[0] - sx:+d}px) "
            f"近{FACE_DIAG_LOG_FRAMES}帧转向{self._face_assert_count}次"
            f"(自检未生效{self._face_unverified_count}次)"
        )
        self._face_assert_count = 0
        self._face_unverified_count = 0

    def _face_step(self, ctx: Context, target: Detection,
                   allow_press: bool = True) -> bool:
        """攻击前的朝向处理：转向 / 定期重申 / 生效自检。

        【为什么转向要单独占一帧】游戏会忽略攻击动画期间的按键（代码在 buff
        施法窗口里已经用到这条结论）。转向键和技能键挤在同一帧按、还是
        30~90ms 短按，很容易被吞，所以转向单独占一帧、按住 120~180ms。

        【记忆只记"已确认"的朝向】按出方向键时不再乐观地把 _face_dir 改成
        目标侧：只有自检到角色真的朝该方向移动（或按住移动）才写入。游戏吞键
        时 _face_dir 保持旧值 → 这里会继续暂停施法并按 0.2s 间隔重试，直到
        转成为止；不会出现"以为转了、其实没转，然后连续朝反方向打"（10/25
        13:52 现场：记忆=left、画面朝 right，连续 14 秒攻击打向反侧）。

        【定期重申】即使认为朝向正确，也每 TURN_REASSERT(4~6s 抖动) 重申
        一次：一次被吞掉的转向最多 4 秒就自愈，不会长期错向。
        重申点按会朝面朝方向走 1~2 步，所以有近身威胁时跳过（避免往贴身怪
        身上撞），并把计时器往后推一个完整间隔。

        【唯一放行条件】只有"朝向已确认且与目标侧一致"（或目标在正上方死区）
        才返回 False 允许施法。未确认 / 正在纠正 / 目标刚翻边 / 自身未定位
        一律返回 True 停手：宁可少打几发，也不朝反方向空放（10/25 现场）。

        Args:
            ctx:         当前帧感知数据
            target:      当前锁定目标
            allow_press: 是否允许按方向键。近战贴脸死区 / 防抖窗口 / 遮挡虚拟
                         目标期间传 False——那些场景按方向键会朝怪迈步、穿怪身体。

        Returns:
            True  → 本帧不要施法（正在转向，或朝向未确认）
            False → 朝向已确认且无需处理，调用方照常放技能
        """
        hold = self._face_step_inner(ctx, target, allow_press)
        if not hold:
            self._face_wait_since = 0.0   # 恢复施法 → 等待计时复位
        return hold

    def _log_face_wait(self, reason: str):
        """朝向未确认而暂停施法时，限频打一条带已等待时长的日志。"""
        now = time.time()
        if self._face_wait_since <= 0.0:
            self._face_wait_since = now
        if now - self._face_wait_log_at < FACE_WAIT_LOG_SECONDS:
            return
        self._face_wait_log_at = now
        self._log(
            f"[朝向] {reason}，暂停施法等待朝向确认"
            f"（已等待 {now - self._face_wait_since:.1f}s）"
        )

    def _face_step_inner(self, ctx: Context, target: Detection,
                         allow_press: bool = True) -> bool:
        """_face_step 的实现体（返回值语义见 _face_step）。"""
        if self._is_stand_mode() or target is None:
            self._face_clear_pending()
            self._face_attempt_side = None
            return False
        if not allow_press:
            self._face_clear_pending()
            self._face_attempt_side = None
            return False
        foot = self._effective_self_pos(ctx)
        if foot is None:
            # 连最后已知位置都没有 → 无法判断目标在哪侧，不盲打（等定位恢复）
            self._log_face_wait("无法定位自身")
            return True
        sx = foot[0]
        need = self._face_needed(sx, target)

        # ---- 换边防抖：统计"需要的朝向"已连续保持多少帧 ----
        # 群怪分布在角色两侧时最近目标每帧左右翻，每次翻边都转身 = 原地狂转
        # （转向键被动画吞 + 转向帧堵输出，实测 3 秒转 5 次、3 次白转）。
        # 瞬时翻边不转、也不施法；新侧稳定满 TURN_SIDE_CONFIRM_SECONDS 才转。
        if need == self._face_need_side:
            self._face_need_frames += 1
        else:
            self._face_need_side = need
            self._face_need_frames = 1

        self._face_diag(ctx, target, sx)

        now = time.time()

        # ---- 1) 先判定上一轮转向是否生效 ----
        settling = self._face_settle(ctx)
        if need is None:
            return False      # 目标近似正上方/重叠：保持朝向，正常施法

        # 本帧无法自检（自身定位丢失）：绝不能朝未确认的朝向施法。
        # 按最小间隔盲点一下方向键（可能被游戏接收），等定位恢复后再核实。
        if ctx.self_position is None:
            if need != self._face_dir:
                if now >= self._face_last_press_at + TURN_MIN_GAP_SECONDS \
                        and self._floor_ahead(ctx, sx, foot[1], need,
                                              MICRO_MOVE_EDGE_LOOKAHEAD_X):
                    self._turn_tap(need)
                    self._face_last_press_at = time.time()
                    self._face_assert_count += 1
                    # 盲点无法自检：实际朝向已不可信 → 标记未知，
                    # 定位恢复后必须重新转向并确认，绝不沿用旧记忆施法。
                    self._face_dir = None
                self._log_face_wait("自身未定位")
                return True
            return False

        # 上一轮转向还在等判定：严格禁止施法，等确认/失败重试。
        if settling and need != self._face_dir:
            self._log_face_wait("转向未确认")
            return True

        # 目标回到"已确认朝向"这一侧、且没有待检转向 → 之前那次纠正作废
        # （否则 _face_attempt_side 会一直挂着，击退等方向性动作被永久拦掉）。
        if need == self._face_dir and self._turn_dir is None:
            self._face_attempt_side = None

        # ---- 2) 判断要不要（重新）按方向键 ----
        if need != self._face_dir:
            if self._face_dir is None \
                    or self._face_need_frames >= self._frames(TURN_SIDE_CONFIRM_SECONDS) \
                    or (self._face_wait_since > 0
                        and now - self._face_wait_since >= FACE_WAIT_COMMIT_SECONDS):
                # 朝向不符（换边已稳定 / 首次定位），或目标侧抖动太久
                # （不能无限停手）→ 朝当前目标侧转
                need_press = True
            else:
                # 瞬时翻边 → 不转，也【不施法】（绝不朝旧方向空放）
                self._log_face_wait("目标刚换边")
                return True
        elif now >= self._face_next_assert:
            if self._nearest_knockback_threat(ctx, foot) is not None:
                # 有近身威胁时跳过重申：重申点按 = 朝面朝方向走 1~2 步，
                # 此时面朝方向上就贴着怪，点按等于往怪身上撞。
                # 推迟一个完整间隔，避免威胁刚解除（怪刚被推走）就立刻点按。
                self._face_next_assert = now + random.uniform(
                    TURN_REASSERT_MIN_SECONDS, TURN_REASSERT_MAX_SECONDS)
                need_press = False
            else:
                need_press = True      # 朝向"认为"对 → 到点重申一次（自愈被吞的转向）
        else:
            need_press = False

        if need_press:
            earliest = self._face_last_press_at + TURN_MIN_GAP_SECONDS
            if now < earliest:
                # 最小间隔未到：真需要转向就停手等（不朝旧方向打）；
                # 只是重申 → 让路给施法，下一帧再说。
                if need != self._face_dir:
                    self._log_face_wait("等待转向最小间隔")
                    return True
                need_press = False

        if not need_press:
            return False

        # ---- 边缘安全：按一下方向键角色会朝那边迈几像素 ----
        # 前方很近处没有地板（悬崖/平台边缘）时不按。此时也不施法——
        # 朝向与目标侧不符，放技能等于朝反方向空放（宁可停手等目标移动）。
        if not self._floor_ahead(ctx, sx, foot[1], need,
                                 MICRO_MOVE_EDGE_LOOKAHEAD_X):
            self._face_next_assert = now + random.uniform(
                TURN_REASSERT_MIN_SECONDS, TURN_REASSERT_MAX_SECONDS)
            if need == self._face_dir:
                # 只是定期重申，朝向本来就对 → 跳过点按，照常施法
                return False
            self._log_face_wait(f"{need} 侧前方无地板，跳过转向")
            return True

        # ---- 3) 转向：按住 120~180ms，本帧不施法 ----
        seconds = self._turn_tap(need)
        if seconds <= 0:
            # 按键根本没发出去（键无效/窗口未锁定）→ 绝不能更新 _face_dir，
            # 否则又变成"以为转了、其实没转"
            self._log("[朝向] 转向按键发送失败，保持原朝向")
            self._face_next_assert = now + random.uniform(
                TURN_RETRY_MIN_SECONDS, TURN_RETRY_MAX_SECONDS)
            if need == self._face_dir:
                return False    # 只是重申失败，朝向本身正确 → 照常施法
            self._log_face_wait("转向按键发送失败")
            return True

        # 纠正（need 与已确认朝向不符）才登记为"未确认转向"；重申不登记。
        # 【关键】这里绝不写 _face_dir：还没被游戏确认，写了就会变成
        # "以为转了、其实没转"（10/25 13:52 连续 14 秒朝反方向打的根因）。
        # 只有 _face_settle 自检到实际位移（或按住移动）才写入记忆。
        is_correction = need != self._face_dir
        started = is_correction and need != self._face_attempt_side
        if is_correction:
            self._face_attempt_side = need
        # 计时基准取"按住结束"的时刻：下一次重申间隔、最小间隔、生效自检
        # 等待窗口都从按键真正发出之后开始算（按住本身要 120~180ms）。
        pressed_at = time.time()
        self._face_last_press_at = pressed_at
        self._face_assert_count += 1
        self._turn_dir = need
        self._turn_time = pressed_at
        self._turn_x0 = ctx.self_position[0] if ctx.self_position else None
        self._turn_verify_hits = 0       # 新一轮转向：重新累计达标帧数
        # 纠正转向失败 → 始终按短间隔重试（严格模式下不靠"恢复施法"兜底）；
        # 重申只是健康检查 → 用正常重申间隔。
        if is_correction:
            self._face_next_assert = pressed_at + random.uniform(
                TURN_RETRY_MIN_SECONDS, TURN_RETRY_MAX_SECONDS)
        else:
            self._face_next_assert = pressed_at + random.uniform(
                TURN_REASSERT_MIN_SECONDS, TURN_REASSERT_MAX_SECONDS)
        if started:
            self._log(
                f"[朝向] 怪物在{'右' if need == 'right' else '左'}"
                f"(dx={target.center[0] - sx:+d}px)，按住 {need} "
                f"{seconds * 1000:.0f}ms 转向（未确认前暂停施法）"
            )
        return True

    def _force_release_directions(self, reason: Optional[str] = None):
        """补发左右方向键的 KEYUP（force：本地没记录也发一次）。"""
        self.executor.key_up("left", force=True)
        self.executor.key_up("right", force=True)
        self._stuck_release_at = time.time()
        self._stand_x = None
        self._stand_drift_dir = 0
        self._stand_drift_frames = 0
        if reason:
            self._log(f"[朝向] {reason} → 已补发左右方向键释放")

    def _stuck_direction_guard(self, ctx: Context):
        """清理游戏侧可能卡住的方向键（仅当决策层自己没有按方向键时）。

        【为什么需要】移动/转向是 key_down + key_up 成对发送的，若某次 KEYUP
        没被游戏收到，游戏侧会一直"按住"该方向 → 角色持续朝一个方向走、朝向
        也固定错，而决策层 _held_key 为空、完全不知情。实测 12:38:10~12:38:14
        出现过"代码认为站在原地面向右、角色却一直向左走"的漂移。

        两个信号兜底：
          · 快：无按键却连续多帧朝同一方向漂移 → 立刻补发 KEYUP
          · 慢：站定期间每隔 STUCK_KEY_IDLE_SECONDS 补发一次（兜住"卡住但没漂移"）
        补发 KEYUP 不改变游戏内朝向（朝向只在按下方向键时变化），也不影响
        正在按住的方向键（自己在移动时直接跳过）。
        """
        if self._held_key is not None:
            self._stand_x = None            # 自己在移动：漂移是预期的，不干预
            self._stand_drift_frames = 0
            return
        if time.time() < self._drift_suppress_until:
            # 刚做过转向/微动点按：角色因点按挪几像素是预期行为，不是卡键
            self._stand_x = None
            self._stand_drift_frames = 0
            return
        pos = ctx.self_position
        if pos is None:
            self._stand_x = None
            return
        x = pos[0]
        if self._stand_x is not None:
            dx = x - self._stand_x
            if abs(dx) >= STUCK_KEY_DRIFT_STEP_X:
                sign = 1 if dx > 0 else -1
                if sign == self._stand_drift_dir:
                    self._stand_drift_frames += 1
                else:
                    self._stand_drift_dir = sign
                    self._stand_drift_frames = 1
                if self._stand_drift_frames >= STUCK_KEY_DRIFT_FRAMES:
                    self._force_release_directions(
                        "无按键却持续漂移（移动键疑似卡住）")
                    return
            else:
                self._stand_drift_frames = 0
        self._stand_x = x
        if time.time() - self._stuck_release_at >= STUCK_KEY_IDLE_SECONDS:
            self._force_release_directions()   # 静默兜底

    # =========================================================================
    # =========================================================================
    # 站桩定时微动（反"定点一动不动"的机器特征）
    # =========================================================================
    #
    # 需求：每过 N 秒做一次"很短的左/右移动"就行，不关心移动多少像素。
    # 所以这里不做任何位置闭环，就是两次极短的点按：
    #     先朝"背离站桩朝向"的方向点一下，再朝站桩朝向点一下。
    # 两次时长相同、方向相反 ≈ 原地晃一下；最后一下是朝向方向，
    # 所以结束时角色朝向必然还是站桩朝向，不需要额外转身。
    #
    # 为什么用 _micro_tap（key_down + sleep + key_up）而不是 press_key：
    #   press_key 是按 30~90ms 随机时长做"按下"，时长不可控；微动要的是
    #   可控的毫秒级点按。key_down/key_up 在同一帧内配对完成，不会残留按键。

    def _micro_interval(self) -> float:
        """微动间隔（秒）。"""
        return float(getattr(self.config, "stand_micro_move_interval", 300.0) or 300.0)

    def _micro_tap_seconds(self) -> float:
        """单次点按时长（秒）。配置项单位是毫秒。"""
        ms = float(getattr(self.config, "stand_micro_move_tap_ms", 40) or 40)
        return ms / 1000.0

    def _micro_move_on(self) -> bool:
        """是否启用站桩定时微动（站桩 + 开关开 + 间隔>0）。"""
        if not self._is_stand_mode():
            return False
        if not bool(getattr(self.config, "stand_micro_move_enabled", True)):
            return False
        return self._micro_interval() > 0

    def _micro_tap(self, direction: str, seconds: float):
        """朝 direction 极短地按一下（同步阻塞，时长精确到毫秒）。

        与 ``press_key`` 的区别：press_key 的按压时长固定 30~90ms 随机，
        这里能按配置的毫秒数精确点一下。
        """
        seconds = max(MICRO_MOVE_TAP_MIN_SECONDS,
                      min(MICRO_MOVE_TAP_MAX_SECONDS, seconds))
        self.executor.key_down(direction)
        time.sleep(seconds)
        self.executor.key_up(direction)
        # 点按会挪动角色几像素（预期），短暂抑制漂移检测，避免卡键兜底误判
        self._drift_suppress_until = time.time() + 0.5

    def _micro_reset_timer(self):
        """重置微动计时（下一次在间隔之后）。"""
        self._micro_cooldown_frames = self._frames(self._micro_interval())

    def _micro_move_tick(self, ctx: Context) -> bool:
        """站桩微动：到点就"反方向点一下 + 朝向点一下"，一帧做完。

        Returns:
            True 表示本帧做了微动 → 调用方直接 return（本帧不攻击）。
        """
        if not self._micro_move_on():
            return False
        if self._micro_cooldown_frames > 0:
            self._micro_cooldown_frames -= 1
            return False

        facing = self._stand_facing()
        away = "left" if facing == "right" else "right"
        # 防落崖：反方向很近处没有地板就跳过本轮（微动只有几像素，
        # 前瞻距离也取得很近，不会像后撤那样动不动就跳过）
        pos = ctx.self_position
        if pos is not None and not self._floor_ahead(
                ctx, pos[0], pos[1], away, MICRO_MOVE_EDGE_LOOKAHEAD_X):
            self._log("[微动] 反方向没有地板，本轮跳过（防落崖）")
            self._micro_reset_timer()
            return False

        tap = self._micro_tap_seconds()
        self._micro_tap(away, tap)
        time.sleep(MICRO_MOVE_TAP_GAP_SECONDS)
        self._micro_tap(facing, tap)
        self._face_dir = facing       # 最后一下朝朝向 → 朝向不变
        self._micro_reset_timer()
        self._log(
            f"[微动] 向{'左' if away == 'left' else '右'}"
            f"、再向{'左' if facing == 'left' else '右'}"
            f"各点 {tap * 1000:.0f}ms（朝向 {facing} 不变）"
        )
        return True

    # =========================================================================
    # 目标选择
    # =========================================================================

    def _retain_missed_target(self, ctx: Context) -> Optional[Detection]:
        """长手(远程)目标漏检保持：攻击中锁定目标因技能特效被遮挡而短时漏检时，
        沿用最后已知位置继续攻击，避免误判"目标没了"而换目标/乱跑。

        保持条件（全部满足）：
          1. 长手且处于战斗中（ATTACKING/CHASING/HEALING/RECOVERING）
          2. 最后已知位置与角色同平台、水平距离 ≥ OCCLUSION_RETREAT_X（远程，非贴脸）
          3. 连续漏检未超保持时间：射程内目标 1.0s（技能特效遮挡居多），
             射程外目标 0.5s（可能已死/走远）；均按 fps 换算成帧
        贴脸漏检(dx < OCCLUSION_RETREAT_X)由 _handle_long_range_occlusion 后撤处理。
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return None
        if self._target_monster is None:
            return None
        if not self._in_combat():
            return None
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return None
        sx, sy = foot
        t = self._target_monster
        if abs(sy - (t.y + t.h)) > getattr(self.config, "attack_range_y", 60):
            return None  # 跨层 → 非特效遮挡
        dist = abs(sx - t.center[0])
        if dist < OCCLUSION_RETREAT_X:
            return None  # 贴脸 → 交给后撤逻辑
        # 射程内目标多半只是被特效挡住 → 保持更久；射程外（可能已死/走远）保持 0.5s。
        retain_seconds = TARGET_MISS_RETAIN_INRANGE_SECONDS \
            if dist <= self._get_attack_range() else TARGET_MISS_RETAIN_SECONDS
        if self._target_miss_frames >= self._frames(retain_seconds):
            return None
        self._target_miss_frames += 1
        # 漏检保持是"继续对着最后位置打"，日志缺失时完全看不出角色在打空气
        # （本次朝向问题排查时就是因为这条被注释掉而绕了远路）→ 每个漏检
        # 片段只打一次，不刷屏。
        if self._target_miss_frames == 1:
            self._log(
                f"[目标] 锁定目标本帧漏检（技能特效遮挡/检测波动？）"
                f"→ 沿用最后位置继续攻击"
                f"(怪中心x={t.center[0]} 水平距={abs(sx - t.center[0])}px)"
            )
        return t

    def _dispatch_retained(self, ctx: Context, target: Detection):
        """把"漏检但被保住"的目标按当前距离分发给攻击或追击。

        不能无条件走 _attack：_retain_missed_target 现在对 CHASING 也生效，
        目标可能已经超出攻击距离，而 _attack 的距离守卫在这种情况下会直接
        return——既不攻击也不追赶，角色会原地站住不动。
        """
        if self._can_attack(ctx, target):
            self._fsm.transition(State.ATTACKING)
            self._attack(ctx, target)
        else:
            self._fsm.transition(State.CHASING)
            self._chase(ctx, target)

    def _pick_nearer_visible(self, ctx: Context,
                             locked: Detection) -> Optional[Detection]:
        """锁定目标不合适时，改打"更近 / 真实可见"的同层怪（带防抖）。

        两类情况（都只处理长手；近战贴脸由 _handle_melee 自己判定）：
          1. 锁定目标本帧漏检：
             - 最后位置仍在射程内（多半是技能特效遮挡）→ 保持期内先信它，
               只接受明显更近（≥NEARER_GHOST_INRANGE_MARGIN_X）的可见怪；
               跨侧另加 NEARER_CROSS_SIDE_EXTRA_X，或新目标真正贴身
               （≤NEARER_TRULY_CLOSE_X）且更近才转。
             - 最后位置在射程外（可能已死/走远）→ 真实可见且在射程内的怪优先。
          2. 锁定目标可见：新怪明显更近才抢；跨侧需要更大优势，只有真正贴身
             （≤NEARER_TRULY_CLOSE_X）且更近才值得转一次身。

        连续 NEARER_SWITCH_FRAMES 帧指向同一只怪才切换，吸收 YOLO 单帧
        漏检/抖动，避免两个距离接近的怪左右来回换目标。

        Returns:
            更该打的怪物；不满足条件时返回 None（维持原锁定）。
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return None

        foot = ctx.self_position
        if foot is None:
            if self._last_self_pos is None \
                    or self._self_pos_stale_frames > self._frames(SELECT_FALLBACK_SECONDS):
                self._nearer_candidate = None
                self._nearer_frames = 0
                return None
            foot = self._last_self_pos
        sx, sy = foot
        attack_range = self._get_attack_range()

        locked_visible = any(self._is_same_monster(locked, m) for m in ctx.monsters)
        locked_dist = abs(sx - locked.center[0])

        # 候选也按"同侧优先"分组：背后怪即使更近一点，也要先让给面朝侧的怪
        # （转身停手 0.25~0.4s）。同侧怪在射程内且没有明显更近的跨侧怪时，
        # 一律考虑同侧怪。
        facing_known = self._face_dir in ("left", "right")
        facing_right = self._face_dir == "right"
        same_best = None
        same_dist = float("inf")
        cross_best = None
        cross_dist = float("inf")
        for m in ctx.monsters:
            if self._is_same_monster(locked, m):
                continue
            if self._in_ghost_zone(m):
                continue  # 残影位置：不当成"更近的可见怪"去抢占
            if not self._same_platform(sy, m.y + m.h):
                continue
            if not self._in_front(m, sx):
                continue  # 站桩模式：只考虑朝向正前方
            d = abs(sx - m.center[0])
            if not facing_known \
                    or ((m.center[0] > sx) == facing_right):
                if d < same_dist:
                    same_best = m
                    same_dist = d
            elif d < cross_dist:
                cross_best = m
                cross_dist = d

        attack_range = self._get_attack_range()
        if same_best is not None and cross_best is not None:
            cross_wins = cross_dist + SAMESIDE_PREFER_X < same_dist \
                or (cross_dist <= attack_range < same_dist)
            if cross_wins:
                best, best_dist = cross_best, cross_dist
            else:
                best, best_dist = same_best, same_dist
        elif same_best is not None:
            best, best_dist = same_best, same_dist
        else:
            best, best_dist = cross_best, cross_dist

        if best is None:
            self._nearer_candidate = None
            self._nearer_frames = 0
            return None

        # 跨侧换目标需要额外距离优势：当前锁定目标在面朝侧、新目标在另一侧时，
        # 切过去必须先转身（约 0.25~0.4s 停手）。不额外加门槛的话，两只距离
        # 接近的怪来回抢占会让角色反复转身、技能空窗明显（10/28 实测 315 次
        # 转向/28 分钟）。跨侧只有"真正贴身且更近"或"优势足够大"才转。
        facing = self._face_dir
        cross_side = False
        if facing in ("left", "right"):
            facing_right = facing == "right"
            cross_side = ((best.center[0] > sx) != facing_right) \
                and ((locked.center[0] > sx) == facing_right)

        locked_in_range = locked_dist <= attack_range
        # 射程内鬼目标保持期内（多半是特效遮挡）→ 先信它，只有明显更近才抢
        ghost_retaining = (not locked_visible and locked_in_range
                           and self._target_miss_frames
                           < self._frames(TARGET_MISS_RETAIN_INRANGE_SECONDS))

        if ghost_retaining:
            if cross_side:
                ok = ((best_dist <= NEARER_TRULY_CLOSE_X
                       and best_dist < locked_dist)
                      or (best_dist <= attack_range
                          and best_dist + NEARER_GHOST_INRANGE_MARGIN_X
                          + NEARER_CROSS_SIDE_EXTRA_X < locked_dist))
            else:
                ok = (best_dist <= attack_range
                      and best_dist + NEARER_GHOST_INRANGE_MARGIN_X < locked_dist)
        elif not locked_visible:
            # 原目标漏检且不在射程内（可能已死/走远）：真实可见且在射程内的怪优先
            ok = best_dist <= attack_range
        else:
            # 锁定目标可见：只在明显更近时抢；跨侧要求更高
            if cross_side:
                ok = ((best_dist <= NEARER_TRULY_CLOSE_X
                       and best_dist < locked_dist)
                      or (best_dist <= attack_range
                          and best_dist + NEARER_SWITCH_MARGIN_X
                          + NEARER_CROSS_SIDE_EXTRA_X < locked_dist))
            else:
                ok = (best_dist <= attack_range
                      and best_dist + NEARER_SWITCH_MARGIN_X < locked_dist)

        if not ok:
            self._nearer_candidate = None
            self._nearer_frames = 0
            return None

        if self._nearer_candidate is not None \
                and self._is_same_monster(self._nearer_candidate, best):
            self._nearer_frames += 1
        else:
            self._nearer_candidate = best
            self._nearer_frames = 1
        if self._nearer_frames < NEARER_SWITCH_FRAMES:
            return None

        self._nearer_candidate = None
        self._nearer_frames = 0
        self._log(
            f"[目标] 更近的可见怪优先 新目标x={best.center[0]} 距离={best_dist}px "
            f"→ 原目标x={locked.center[0]} 距离={locked_dist}px"
            + ("" if locked_visible else "（原目标本帧漏检）")
        )
        return best

    def _resolve_locked_target(self, ctx: Context) -> Detection:
        """解析当前应攻击的目标怪物（就近原则 + 攻击期保持锁定）。

        优先沿用已锁定目标：只要同一只怪仍在画面中就继续打它，
        避免 YOLO 帧间检测波动导致目标来回切换、攻击刚触发就中断；
        已锁定目标消失/被击杀/残影超时/离开同平台后，才重新选目标。
        首次选目标时按"同平台最近"原则（_pick_best_target）。

        Returns:
            当前应攻击的怪物；若画面中没有同平台怪，返回 None。
        """
        if self._target_monster is not None:
            # 先看有没有"更该打"的可见同层怪：
            #   - 身边的怪优先于远处还在追的目标
            #   - 真实可见的怪优先于本帧已漏检、只剩最后位置的"鬼目标"
            # 连续 NEARER_SWITCH_FRAMES 帧成立才切换，避免来回抖。
            nearer = self._pick_nearer_visible(ctx, self._target_monster)
            if nearer is not None:
                self._occluded_frames = 0
                self._occluded_retreat_frames = 0
                self._target_miss_frames = 0
                return nearer
            ghost_locked = False
            for m in ctx.monsters:
                if self._in_ghost_zone(m):
                    # 该位置已被判定为残影：不再锁回来（哪怕是原来的锁定目标）
                    if self._is_same_monster(self._target_monster, m):
                        ghost_locked = True
                    continue
                if self._is_same_monster(self._target_monster, m):
                    # 锁定目标仍在画面，但已不在角色同一平台
                    # （跨层/掉下平台）→ 放弃锁定，只打同层怪
                    if ctx.self_position is not None and not self._same_platform(
                            ctx.self_position[1], m.y + m.h):
                        break
                    # 站桩模式：目标跑到身后 → 放弃锁定（不转向，打不到）
                    if ctx.self_position is not None \
                            and not self._in_front(m, ctx.self_position[0]):
                        break
                    # 目标重新可见（遮挡解除）→ 清除遮挡/漏检计数
                    self._occluded_frames = 0
                    self._occluded_retreat_frames = 0
                    self._target_miss_frames = 0
                    return m
            if ghost_locked:
                # 锁定目标就在残影位置：直接放手，不能走"遮挡/漏检沿用最后位置"
                # （否则刚判定的残影会被保留逻辑再打 0.5~1 秒）
                self._target_monster = None
                self._casts_on_target = 0
                self._occluded_frames = 0
                self._target_miss_frames = 0
            # 锁定目标本帧不在画面：可能是被角色自身遮挡（贴脸攻击）
            occluded = self._occluded_target(ctx)
            if occluded is not None:
                return occluded
            # 长手：远程技能特效遮挡 → 保持锁定继续打（沿用最后位置）
            retained = self._retain_missed_target(ctx)
            if retained is not None:
                return retained
            self._occluded_frames = 0
            self._target_miss_frames = 0
            self._target_monster = None
        return self._pick_best_target(ctx)

    def _is_same_monster(self, a: Detection, b: Detection) -> bool:
        """按中心点距离判断两个检测是否可能是同一只怪。

        容差与锁定匹配一致（目标 bbox 宽度的 1.5 倍，保底 40px），
        用于攻击超时统计。
        """
        if a is None or b is None:
            return False
        tolerance = max(int(a.w * 1.5), 40)
        d = abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
        return d <= tolerance

    def _mark_ghost(self, m: Detection) -> Tuple[float, int]:
        """把该位置记为残影；同一位置屡次被判定，忽略时间翻倍。返回 (忽略秒数, 第几次)。"""
        now = time.time()
        r = self._px(GHOST_ZONE_RADIUS)
        cx, cy = m.center
        for z in self._ghost_zones:
            if abs(cx - z["x"]) + abs(cy - z["y"]) <= r:
                z["count"] += 1
                z["x"], z["y"], z["last"] = cx, cy, now
                seconds = min(GHOST_ZONE_SECONDS * 2 ** (z["count"] - 1),
                              GHOST_ZONE_MAX_SECONDS)
                z["until"] = now + seconds
                return seconds, z["count"]
        self._ghost_zones.append({"x": cx, "y": cy, "until": now + GHOST_ZONE_SECONDS,
                                  "count": 1, "last": now})
        return GHOST_ZONE_SECONDS, 1

    def _in_ghost_zone(self, m: Detection) -> bool:
        """该检测框是否落在正在忽略的残影位置（顺便清理很久以前的记录）。"""
        if not self._ghost_zones:
            return False
        now = time.time()
        self._ghost_zones = [z for z in self._ghost_zones
                             if z["until"] > now or now - z["last"] < GHOST_MEMORY_SECONDS]
        r = self._px(GHOST_ZONE_RADIUS)
        cx, cy = m.center
        return any(z["until"] > now and abs(cx - z["x"]) + abs(cy - z["y"]) <= r
                   for z in self._ghost_zones)

    def _set_locked_target(self, target: Detection):
        """更新锁定目标；锁定对象真正更换时打印新目标坐标。

        每帧都会调用（目标位置本身有检测抖动），所以用 _is_same_monster
        判断还是同一只怪：同一只不打印，只有换怪（含首次锁定）才打印，
        方便事后按日志核对"为什么打这只/为什么换目标"。
        """
        old = self._target_monster
        self._target_monster = target
        if old is not None and self._is_same_monster(old, target):
            return
        cx, cy = target.center
        self._log(
            f"[目标] 锁定新目标 中心=({cx},{cy}) "
            f"脚底=({cx},{target.y + target.h}) "
            f"框=({target.x},{target.y},{target.w}x{target.h}) "
            f"conf={target.confidence:.2f}"
        )

    @property
    def locked_target(self) -> Optional[Detection]:
        """当前锁定的怪物检测框（供 UI 预览画红框/调试）。"""
        return self._target_monster

    def _occluded_target(self, ctx: Context) -> Optional[Detection]:
        """目标从画面消失时，判断是否被角色遮挡并返回可继续攻击的虚拟目标。

        短手贴脸攻击时角色站在怪正前方，角色模型+攻击特效会遮住怪物，
        YOLO 置信度骤降被 conf 阈值过滤 → 本帧"看不到"锁定目标。
        此时怪其实还在原处（贴脸近战怪基本不动），用最后已知位置继续打，
        避免目标消失引发：
          - 重选别的怪 → dx 符号翻转 → 左右乱转
          - 攻击中断 → 转探索 → 角色离开 → 怪重新可见 → 反复贴脸失败

        判定条件（全部满足才视为"被遮挡"，否则按怪真消失处理）：
          1. 短手模式且处于战斗中（角色站位稳定，见 _in_combat）
          2. 角色与目标最后位置在同一平台
          3. 角色与目标最后位置水平距离很近（贴脸距离 + 宽度容差）
          4. 连续遮挡未超上限（OCCLUSION_MAX_FRAMES，超时视为怪真消失）
        """
        if getattr(self.config, "attack_type", "long") != "short":
            return None  # 长手站远程打，不会贴脸遮挡，保持原逻辑
        if self._target_monster is None:
            return None
        if not self._in_combat():
            return None
        # 用"有效位置"而非实时位置：贴脸时角色名字也可能被怪/攻击特效遮挡，
        # OCR 定位失败 → ctx.self_position 为 None。攻击中角色位置不变，
        # 用最后已知位置(_last_self_pos)兜底判遮挡，避免"怪被遮 + 名字被遮"
        # 同时发生时遮挡判定失败 → 清空锁定 → 转探索乱走的死循环。
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return None
        sx, sy = foot
        t = self._target_monster
        if not self._same_platform(sy, t.y + t.h):
            return None  # 已跨层 → 非遮挡
        # 遮挡判定同样用"到怪近侧边缘"的距离：贴脸打怪时角色就在怪
        # 身体旁边（边缘距离≈0），中心距离可能超过攻击距离的怪（宽怪）
        # 也能正确判定"角色在怪跟前"，不会误判非遮挡而清空锁定。
        if abs(sx - self._melee_edge_x(sx, t)) \
                > self._get_attack_range() + OCCLUSION_HALF_WIDTH_X:
            return None  # 角色不在目标跟前 → 非遮挡
        self._occluded_frames += 1
        if self._occluded_frames > OCCLUSION_MAX_FRAMES:
            self._log("[目标] 目标被遮挡超时，判定已消失，放弃攻击")
            return None
        if self._occluded_frames == 1:
            self._log("[目标] 目标被角色遮挡，沿用最后位置继续攻击")
        return t

    def _pick_best_target(self, ctx: Context,
                          exclude: Optional[Detection] = None) -> Detection:
        """选择【同平台】怪物：朝向已知时同侧优先，否则取最近。

        只打同平台的怪：角色脚底与怪物脚底（bbox 底部）的垂直差
        ≤ 垂直容差（attack_range_y）才视为可攻击目标，距离只算水平方向。
        跨层怪物直接忽略（不爬绳/不跳跃/不追击），画面里没有同平台怪
        时返回 None，转探索等自己走到那一层再打。

        【同侧优先】朝向已知（_face_dir 非空）时优先打面朝侧的怪：
        选背后的怪 = 转身停手 0.25~0.4s。另一侧的怪必须比同侧最近怪
        还近 SAMESIDE_PREFER_X(150px) 才会被选中。朝向未知时退化为
        "最近优先"。

        说明：同平台必须用"脚底 vs 脚底"。若用"角色中心y vs 怪物bbox中心y"，
        高大怪物（如 150px 高）的中心点比角色中心高出 40~50px，
        超过 30px 容差 → 同平台的怪被误判为跨层 → 一直爬绳/跳跃/乱跑不攻击。

        Args:
            ctx:     当前帧感知数据
            exclude: 需要跳过的怪物（如疑似残影），可选
        """
        # 残影位置整片忽略：判定过残影的地方（可能是误识别的地图物体）不再选
        monsters = [m for m in ctx.monsters if not self._in_ghost_zone(m)]
        if not monsters:
            return None
        if exclude is not None:
            monsters = [m for m in monsters
                        if not self._is_same_monster(exclude, m)]
            if not monsters:
                return None
        player = ctx.self_position
        if player is None:
            # 自身定位失败：不要退回"画面最大框"——那会锁到远处/上层平台的怪
            # （10/24 日志里 500~1000px 的跨屏追击大多来自这里）。
            # 用最后已知位置继续按"同平台最近"选；位置太旧就本帧不选，
            # 宁可转探索/等定位，也不跨半张地图乱追。
            if self._last_self_pos is None \
                    or self._self_pos_stale_frames > self._frames(SELECT_FALLBACK_SECONDS):
                return None
            player = self._last_self_pos

        # 【只打同平台】仅考虑与角色在同一平台的怪物：
        # 脚底垂直差 ≤ 垂直容差（attack_range_y）。
        # 跨层怪物直接忽略——不爬绳不跳跃，等角色自己走到那层再打。
        # 【同侧优先】长手模式且朝向已知时优先打面朝侧的怪：换边意味着
        # 转身停手，另一侧的怪必须明显更近（≥SAMESIDE_PREFER_X）才选它。
        # 短手（近战）要跑过去贴脸，朝向随移动变化，不做同侧偏好。
        facing = self._face_dir \
            if getattr(self.config, "attack_type", "long") == "long" else None
        same_best = None
        same_dist = float("inf")
        cross_best = None
        cross_dist = float("inf")
        for m in monsters:
            mfoot = m.y + m.h  # 怪物脚底（bbox 底部）
            if not self._same_platform(player[1], mfoot):
                continue
            if not self._in_front(m, player[0]):
                continue  # 站桩模式：只考虑朝向正前方的怪
            dx = abs(player[0] - m.center[0])
            same_side = (m.center[0] > player[0]) == (facing == "right") \
                if facing in ("left", "right") else True
            if same_side:
                if dx < same_dist:
                    same_best = m
                    same_dist = dx
            elif dx < cross_dist:
                cross_best = m
                cross_dist = dx

        if same_best is not None and cross_best is not None:
            attack_range = self._get_attack_range()
            cross_wins = cross_dist + SAMESIDE_PREFER_X < same_dist \
                or (cross_dist <= attack_range < same_dist)
            if cross_wins:
                # 跨侧怪明显更近 / 同侧超出射程而跨侧在射程内 → 值得转一次身
                return cross_best
        if same_best is not None:
            return same_best
        return cross_best

    # =========================================================================
    # 地板判定
    # =========================================================================

    def _has_floor_under(self, ctx: Context, pos: Tuple[int, int]) -> bool:
        """检查指定位置下方是否有地板。

        判断: 地板检测框的 Y 范围是否覆盖了该位置的 Y 坐标附近。
        """
        px, py = pos
        for f in ctx.floors:
            if f.x <= px <= f.x + f.w:
                if f.y - 10 <= py <= f.y + f.h + 10:
                    return True
        return True  # 没检测到地板时默认认为可以站（宽容处理）

    # =========================================================================
    # 攻击范围判定
    # =========================================================================

    def _in_combat(self) -> bool:
        """是否处于"战斗中"（含加血/加蓝这类瞬态状态）。

        战斗连续性判定（用最后已知位置兜底、保持锁定目标、遮挡虚拟目标）
        若只认 State.ATTACKING，加血的那一帧就会集体失效：自身定位恰好失败
        → 判定"无法判断距离" → 清锁 → 转探索（_explore 还会真的按住方向键
        走起来，可能越走离怪越远）→ 定位恢复后又重新锁怪。表现为每秒十几次
        状态翻转、每次攻击只放一两发技能。

        加血/加蓝会让 FSM 短暂离开 ATTACKING，但那仍是"正在打仗"，
        所以这里看"是否在打仗"，而不是看某一个具体状态。
        """
        return self._fsm.current in (State.ATTACKING, State.CHASING,
                                     State.HEALING, State.RECOVERING)

    def _effective_self_pos(self, ctx: Context) -> Optional[Tuple[int, int]]:
        """返回当前帧决策用的自身脚底坐标。

        OCR 定位成功 → 实时坐标。
        定位暂时失败（技能特效遮挡角色名字 / 怪物名与角色名重叠 /
        OCR 抖动）但正处于【战斗中】(_in_combat) → 用最后已知位置兜底
        （_last_self_pos，带 SELF_POS_STALE_FRAMES 时效）。
        战斗中用旧坐标基本准确且安全，避免"特效挡名字 → 放弃目标 →
        转探索乱走" 的反复空转。
        其余状态（探索/待命）定位失败 → 返回 None，
        由各调用方按原有逻辑处理（不盲打/不追错方向）。
        """
        if ctx.self_position is not None:
            return ctx.self_position
        if self._in_combat() and self._last_self_pos is not None \
                and self._self_pos_stale_frames <= SELF_POS_STALE_FRAMES:
            return self._last_self_pos
        return None

    def _blind_move_allowed(self) -> bool:
        """定位连续失败时，是否还允许按最后已知位置继续移动。

        追击/近战追击里定位一失败就松键，角色会永远停在遮挡物后面
        （定位失败 → 不移动 → 继续被遮挡）。这里给一个很短的盲走预算：
        够走过栅栏/藤蔓这类装饰物，恢复检测后立刻回到正常追击；
        超过预算则松键等待，避免用过期坐标冲过目标或落崖。
        """
        return self._self_pos_stale_frames <= self._frames(BLIND_CHASE_SECONDS)

    def _log_blind_move(self, where: str):
        """定位丢失但仍继续移动时打一次日志（每次丢失只打一条）。

        丢失不足 0.5s 不打（技能特效挡一两帧很常见），超过才提示，
        避免正常抖动把日志刷满。
        """
        if self._blind_chase_warned \
                or self._self_pos_stale_frames < self._frames(0.5):
            return
        self._blind_chase_warned = True
        secs = max(0.1, self._self_pos_stale_frames / max(1, self.config.fps))
        self._log(
            f"[{where}] 自身定位丢失 {secs:.1f}s → 按最后已知位置继续移动"
            f"（盲走上限 {BLIND_CHASE_SECONDS:.1f}s，疑似被场景遮挡/模型漏检）"
        )

    def _update_scale(self, ctx: Context):
        """按当前帧高度计算像素参数缩放倍数（只供活动范围等新增逻辑使用）。

        游戏把画面等比放大（如 1366×768 → 1920×1080）时，怪物/人物/距离的
        像素尺寸都跟着放大。活动范围的阈值按 pixel_base_height（默认 768）
        高度填写，运行时乘以 当前帧高 / pixel_base_height。
        pixel_base_height 设为 0 关闭缩放（客户端提高分辨率只是"看到更多地图"、
        人物大小不变时应关闭）。
        """
        base = float(getattr(self.config, "pixel_base_height", 768) or 0)
        if base > 0 and ctx.frame_height:
            self._scale = ctx.frame_height / base
        else:
            self._scale = 1.0

    def _px(self, value: float) -> int:
        """把 pixel_base_height 下的像素值换算到当前画面。"""
        return int(round(value * self._scale))

    def _get_attack_range(self) -> int:
        """返回当前生效的攻击距离（像素）。

        - 短手(近战): 固定 MELEE_ATTACK_RANGE_X(50)，【不允许配置修改】，
          与长手完全独立，不读 exe 的 attack_range。
        - 长手(远程): 读 config.attack_range（exe 界面"攻击距离px"可改）。
        切换的是攻击判定代码：
        - 长手: 远程站定攻击（_can_attack/_chase/_attack）
        - 短手: 近战贴脸攻击（_handle_melee/_melee_chase/_melee_attack）
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return MELEE_ATTACK_RANGE_X
        return getattr(self.config, "attack_range", ATTACK_RANGE_X)

    def _get_min_attack_range(self) -> int:
        """返回当前生效的最小有效射程（像素），仅长手(远程)使用。

        - 短手(近战): 返回 0（近战贴脸打，无最小射程概念）。
        - 长手(远程): 读 config.attack_min_range（0 = 关闭后撤，行为与旧版一致）。
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return 0
        return max(0, int(getattr(self.config, "attack_min_range", 0) or 0))

    def _frames(self, seconds: float) -> int:
        """把"秒"换算成帧数（按配置 fps），避免帧数阈值随 fps 漂移。"""
        return max(1, int(round(seconds * max(1, self.config.fps))))

    def _tick_react_delay(self, frames: int, ok_frames: int,
                          low: bool) -> Tuple[int, int]:
        """加血/加蓝的"反应延迟"计数（模拟人类反应时间，降低机器特征）。

        - low 成立 → 连续正常帧数清零；未装载则装载 HEAL_REACT_SECONDS 秒延迟，
          装载后每帧递减，减到 0 表示"可以按键了"（保持 0，每帧重试按键，
          由按键自身的冷却决定何时真正按下）
        - low 不成立 → 连续正常帧数累加，连续正常够 REACT_RESET_SECONDS 才复位

        复位要求"连续正常够久"是防读数毛刺：血/蓝条会跳（实测血量 24% 中途
        读成 99%），单帧正常就清零会把刚装载的延迟反复重置，真低血时反而一直
        等不到按键。

        Args:
            frames:    当前剩余帧数（-1=未触发）
            ok_frames: 当前连续正常帧数
            low:       本帧血/蓝量是否低于阈值

        Returns:
            (更新后的剩余帧数, 更新后的连续正常帧数)
        """
        if low:
            ok_frames = 0
            if frames < 0:
                frames = self._frames(HEAL_REACT_SECONDS)
            elif frames > 0:
                frames -= 1
        else:
            ok_frames += 1
            if ok_frames >= self._frames(REACT_RESET_SECONDS):
                frames = -1
        return frames, ok_frames

    def _jitter_cooldown(self, cooldown) -> float:
        """给技能冷却加随机抖动（×U(0.9, 1.15)），避免释放节奏完全恒定。"""
        try:
            cd = float(cooldown)
        except (TypeError, ValueError):
            return 0.0
        if cd <= 0:
            return cd
        return cd * random.uniform(SKILL_COOLDOWN_JITTER_MIN,
                                   SKILL_COOLDOWN_JITTER_MAX)

    def _is_stand_mode(self) -> bool:
        """站桩模式：不对角色做任何移动（不移动、不转向、不后撤、不探索）。"""
        return bool(getattr(self.config, "stand_mode", False))

    def _stand_facing(self) -> str:
        """站桩模式下的固定朝向（right/left）—— 用于判断"正前方"。"""
        d = getattr(self.config, "stand_facing", "right")
        return d if d in ("left", "right") else "right"

    def _in_front(self, m: Detection, sx: int) -> bool:
        """怪物是否在朝向正前方；非站桩模式恒 True（不做前方过滤）。"""
        if not self._is_stand_mode():
            return True
        if self._stand_facing() == "right":
            return m.center[0] > sx
        return m.center[0] < sx

    def request_buff_window(self, frames: int):
        """请求 buff 施法窗口：接下来 frames 帧内决策层不放技能、不移动。

        主循环在 buff 到期时调用，让角色先停手，等攻击动画收尾后再按 buff，
        避免攻击动画吞掉 buff 按键导致"buff 加不上"。
        可重复调用，取较大值（多 buff 连放时用来续窗）。加血/加蓝不受影响。
        """
        self._buff_hold_frames = max(self._buff_hold_frames, int(frames))

    def _melee_edge_x(self, sx: int, target: Detection) -> int:
        """短手贴脸判定用的"怪近侧身体边缘 x"。

        近战攻击特效命中怪身体任意部位即可，角色不用走到怪中心。
        贴脸/追击/转向的距离判定统一用"角色到怪近侧边缘"：
        宽怪站在旁边就能打，避免为贴中心而穿进怪身体、穿到另一侧
        又超距折返的"左右来回穿"抖动。
        怪半宽取 min(bbox 半宽, MELEE_EDGE_MAX_HALF_W) 封顶，
        防止超宽怪（boss）离老远就判定贴脸。
        注：此函数只用于"距离"判定；"怪在哪边/面朝哪侧"仍看怪中心。
        """
        half_w = min(target.w / 2.0, MELEE_EDGE_MAX_HALF_W)
        if target.center[0] > sx:
            return target.center[0] - half_w
        return target.center[0] + half_w

    def _in_attack_range(self, sx: int, tx: int) -> bool:
        """判断是否在攻击距离内（规则2：水平距离 ≤ 当前生效攻击距离）。

        长手/短手共用 attack_range，此处仅做水平距离判定。
        """
        return abs(sx - tx) <= self._get_attack_range()

    def _same_platform(self, y1: int, y2: int) -> bool:
        """判断两个纵坐标是否在同一平台（规则1：垂直容差内）。

        入参为"角色脚底y"与"怪物脚底y(bbox底部)"，垂直差 ≤ 配置的
        attack_range_y（exe 界面"垂直容差px"）即视为同一平台；
        攻击必须满足此条件才允许发起。
        """
        return abs(y1 - y2) <= getattr(self.config, "attack_range_y", 60)

    def _can_attack(self, ctx: Context, target: Detection) -> bool:
        """综合攻击判定（规则1/2 + 滞回防抖）。

        仅长手(远程)模式走这里；短手(近战)走独立的 _handle_melee。
        - 规则1: 同一平台 = 角色脚底y 与 怪物脚底y(bbox底部) 垂直差 ≤ 容差
        - 规则2: 攻击距离 = 角色x 与 怪物中心x 水平差 ≤ 当前生效攻击距离
          (长手/短手共用 attack_range)
        攻击中（FSM 处于 ATTACKING）且轻微超限时仍允许攻击，
        避免 OCR/YOLO 帧间几像素抖动导致"攻击刚触发就中断、来回跑"。
        目标已明显离开（超过滞回窗口）才判定不可攻击。
        """
        foot = self._effective_self_pos(ctx)
        if foot is None or target is None:
            return False
        sx, sy = foot
        mx = target.center[0]
        mfoot = target.y + target.h
        dx = abs(sx - mx)
        dy = abs(sy - mfoot)
        ry = getattr(self.config, "attack_range_y", 60)
        attack_range = self._get_attack_range()

        # 硬阈值：同平台 + 在攻击距离内
        if dx <= attack_range and dy <= ry:
            return True
        # 滞回：攻击中轻微超出（怪物中心/bbox 波动、OCR 抖动、角色攻击位移）不中断
        # 滞回窗口: 长手 +60px (近战不走这里)
        if self._fsm.current == State.ATTACKING:
            if dx <= attack_range + 60 and dy <= ry + 40:
                return True
        return False

    # =========================================================================
    # 追击（同平台）
    # =========================================================================

    def _monster_nearby(self, ctx: Context, px: int, py: int) -> bool:
        """同一层离攻击距离不到 CHASE_WAIT_MARGIN 的地方有没有怪（不含残影位置）。

        决定"刚出过手要不要原地等它过来"：只有它真的快到了才值得等，
        远处的怪等一秒不如走几步过去。
        """
        reach = self._get_attack_range() + self._px(CHASE_WAIT_MARGIN)
        return any(self._same_platform(py, m.y + m.h) and not self._in_ghost_zone(m)
                   and abs(m.center[0] - px) <= reach for m in ctx.monsters)

    def _chase(self, ctx: Context, target: Detection):
        """按住方向键持续走向怪物（同一平台内直线接近）。

        仅长手(远程)模式走这里（短手走独立的 _melee_chase 贴脸追击）。
        长手模式: 到达 attack_range 前不松手，进入攻击范围后释放方向键。

        贴近目标后保持朝向（不翻转），交给攻击判定，避免原地乱跑抖动。
        """
        if ctx.self_position is None:
            # 定位暂时失败（遮挡/漏检）不要立刻松键：用最后已知位置继续追一段，
            # 否则角色会永远停在遮挡物后面（定位失败 → 不移动 → 继续被遮挡）。
            if self._last_self_pos is None or not self._blind_move_allowed():
                self._release_move()
                return
            self._log_blind_move("追击")
        foot = self._effective_self_pos(ctx) or self._last_self_pos
        if foot is None:
            self._release_move()
            return
        sx = foot[0]
        tx = target.center[0]
        attack_range = self._get_attack_range()
        is_melee = getattr(self.config, "attack_type", "long") == "short"

        # ---- 卡住：先跳一下，连续卡两次就放弃该目标 ----
        # 地图尽头/墙/怪飘在平台外时，跳也过不去；一直顶着走只会原地空转。
        if self._chase_stuck_target is None or not self._is_same_monster(self._chase_stuck_target, target):
            self._chase_stuck = 0
        self._chase_stuck_target = target
        if self._stuck_counter >= STUCK_FRAMES:
            self._stuck_counter = 0
            self._chase_stuck += 1
            if self._chase_stuck >= CHASE_STUCK_GIVEUP:
                # 跳了也过不去 → 放弃它，暂时忽略这个位置（避免反复去撞）
                seconds, count = self._mark_ghost(target)
                self._log(f"[换目标] 追击卡住 {self._chase_stuck} 次还没追到"
                          f"（{'右' if tx > sx else '左'}边 {abs(tx - sx)}px），"
                          f"第 {count} 次判定，{seconds:.0f} 秒内忽略该位置")
                self._chase_stuck = 0
                self._target_monster = None
                self._casts_on_target = 0
                self._release_move()
                return
            self._log("[追击] 卡住了，尝试跳跃")
            self._release_move()
            self._press_jump()
            return

        # ---- 方向滞回：x 差超过死区才切换方向 ----
        # 短手模式死区更小（近战需要更精确对位），长手模式保持原有逻辑
        if is_melee:
            dead_zone = min(15, max(8, attack_range // 3))
        else:
            dead_zone = min(30, max(15, self.config.attack_range // 4))

        if tx > sx + dead_zone:
            self._hold_move("right")
        elif tx < sx - dead_zone:
            self._hold_move("left")
        else:
            # 已贴近目标：若仍在攻击范围外，则保持原方向小步逼近；
            # 已进入攻击范围则停止，交给攻击判定。
            # 追击停止阈值: 长手在攻击范围+10px 处停下，短手在攻击范围+5px 处停下
            stop_margin = 5 if is_melee else 10
            if abs(sx - tx) <= attack_range + stop_margin:
                self._release_move()
            else:
                # 保持当前朝向逼近（不翻转），避免边缘抖动左右跑
                self._hold_move(self._face_dir or "right")

    # =========================================================================
    # 后撤（长手保持最小射程）
    # =========================================================================

    def _should_retreat(self, dx: int, min_range: int) -> bool:
        """是否应后撤拉开距离（带滞回防抖）。

        首次触发: 水平距离 < 最小射程 → 后撤。
        后撤中:   后撤多退一点（≤ min_range + RETREAT_HYSTERESIS_X）才停，
                 避免在最小射程边界来回抖。
        """
        if self._retreating and self._fsm.current == State.CHASING:
            return dx < min_range + RETREAT_HYSTERESIS_X
        return dx < min_range

    def _retreat(self, ctx: Context, target: Optional[Detection] = None,
                 direction: Optional[str] = None, reason: str = "后撤"):
        """长手(远程)后撤：朝怪物反方向或指定方向移动，拉开最小射程距离。

        direction 明确给出时直接朝该方向退（掉血兜底用）；否则按 target 反方向。
        reason 为后撤原因（贴脸挥弓/目标被遮挡/掉血兜底），触发首帧打印一次日志。
        解决"弓箭手贴脸挥弓"问题：贴脸时先退到最小射程之外再射箭，
        后撤期间不放技能（避免边退边挥弓）。
        边缘安全：后退方向前方没地板时不后退；有目标时原地攻击兜底。
        撤不动（前方没地板/到地图边缘/退超时还贴脸）→ 翻越突围：朝怪方向起跳越过它，
        到另一侧再拉开距离（一直退不动等于白停手，后撤期间是不放技能的）。
        """
        foot = self._effective_self_pos(ctx)
        if foot is None:
            self._release_move()
            return
        sx, sy = foot
        now = time.time()
        min_range = self._get_min_attack_range()
        # 只有"开了最小射程 + 有目标 + 长手"才考虑跳过怪物突围：
        # 短手的目标就是贴脸，跳过去是反效果；掉血兜底没有目标时也无从判断往哪跳。
        can_escape = (min_range > 0 and target is not None
                      and getattr(self.config, "attack_type", "long") != "short")

        # ---- 突围中：按住朝怪方向跳过去，越过/超时/到边缘就结束 ----
        if self._escape_started:
            if target is None or self._escape_dir is None:
                # 目标丢了/状态异常：结束突围，别按着方向键继续冲
                self._escape_started = 0.0
                self._escape_dir = None
                self._release_move()
                return
            tx = target.center[0]
            dir_sign = 1 if self._escape_dir == "right" else -1
            passed = (tx - sx) * dir_sign < 0        # 已经到怪物另一侧
            dist = abs(tx - sx)
            if (passed and dist >= min_range) \
                    or now - self._escape_started >= ESCAPE_MAX_TIME \
                    or self._at_edge(ctx, self._escape_dir):
                self._escape_started = 0.0
                self._escape_dir = None
                self._stuck_counter = 0
                self._retreating = False
                self._release_move()
                if not passed or dist < min_range:
                    # 没跳过去 / 跳过去还贴脸：先原地打一会儿，别原地反复跳
                    self._escape_retry_at = now + 2.0
                    self._log("[后撤] 突围没拉开距离，先原地打，稍后再试")
                return
            if self._stuck_counter >= ESCAPE_STUCK_FRAMES:
                self._stuck_counter = 0
                self._escape_jump()
            self._hold_move(self._escape_dir)
            return

        # 卡住 → 跳（与 _chase 一致）
        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[后撤] 卡住了，尝试跳跃")
            self._release_move()
            self._press_jump()
            self._stuck_counter = 0
            return

        # 后退方向：明确指定则直接用；否则 = 怪物反方向（重叠时沿用当前/朝向反方向）
        if direction is None:
            if target is None:
                return
            tx = target.center[0]
            if tx > sx + RETREAT_DEAD_ZONE:
                direction = "left"
            elif tx < sx - RETREAT_DEAD_ZONE:
                direction = "right"
            else:
                if self._held_key in ("left", "right"):
                    direction = self._held_key
                else:
                    direction = "right" if self._face_dir == "left" else "left"

        # 边缘安全：前方没地板 → 不能后退
        if not self._floor_ahead(ctx, sx, sy, direction):
            if can_escape and now >= self._escape_retry_at:
                self._start_escape(target, sx, "前方没地板/已到边缘")
                return
            self._retreating = False
            if target is not None:
                self._fsm.transition(State.ATTACKING)
                self._attack(ctx, target)
            return

        # 退超时还贴着怪（撞墙/被地形挡住）→ 跳过怪物到另一侧
        if can_escape and self._retreating \
                and now - self._retreat_started_at >= RETREAT_MAX_SECONDS \
                and abs(sx - target.center[0]) < min_range \
                and now >= self._escape_retry_at:
            self._start_escape(target, sx, "后撤超时仍被贴脸")
            return

        if not self._retreating:
            self._retreat_started_at = now
            dx_str = f", dx={abs(sx - target.center[0])}px" if target is not None else ""
            self._log(f"[后撤] 触发后撤({reason}){dx_str}，方向={direction}")
        self._hold_move(direction)
        self._retreating = True

    def _nearest_threat(self, ctx: Context, target: Optional[Detection]) -> Optional[Detection]:
        """同一层水平距离离自己最近的怪（含锁定目标，不含残影位置）。

        后撤/贴脸判定要躲的是"离自己最近的那只"，不只是锁定目标：
        锁定目标可能在射程边缘，而另一只已经贴到脸上——只看锁定目标会以为还安全。
        """
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return target
        sx, sy = foot
        cands = [m for m in ctx.monsters
                 if self._same_platform(sy, m.y + m.h) and not self._in_ghost_zone(m)]
        if target is not None:
            cands.append(target)
        if not cands:
            return target
        return min(cands, key=lambda m: abs(m.center[0] - sx))

    def _start_escape(self, target: Detection, sx: int, why: str):
        """开始翻越突围：朝怪物方向起跳，越过它跑到另一侧。"""
        self._escape_dir = "right" if target.center[0] > sx else "left"
        self._escape_started = time.time()
        self._stuck_counter = 0
        self._retreating = False
        self._log(f"[后撤] {why}，跳过怪物到另一侧再拉开距离")
        self._hold_move(self._escape_dir)
        self._escape_jump()

    def _escape_jump(self):
        """突围起跳（按住方向键的状态下按跳跃）。"""
        self._press_jump()

    def _press_jump(self) -> bool:
        """按跳跃键（1 秒冷却，防止连按被游戏丢）。"""
        return self.executor.press_key(self.config.jump_key, cooldown=1.0)

    def _floor_ahead(self, ctx: Context, sx: int, sy: int, direction: str,
                     lookahead: int = RETREAT_EDGE_LOOKAHEAD_X) -> bool:
        """检查前方 lookahead 像素处脚下是否有地板（防落崖）。

        有地板数据(ctx.floors 非空)时，前方无地板 → 返回 False（禁止往那边走，
        防止落崖）；无地板数据时返回 True（不额外限制，交给卡住检测兜底）。

        Args:
            lookahead: 前瞻距离（像素）。默认按后撤的 RETREAT_EDGE_LOOKAHEAD_X，
                站桩微动只挪几像素，会传一个更近的值（MICRO_MOVE_EDGE_LOOKAHEAD_X）。
        """
        if not ctx.floors:
            return True
        px = sx - lookahead if direction == "left" else sx + lookahead
        for f in ctx.floors:
            if f.x <= px <= f.x + f.w and f.y - 30 <= sy <= f.y + f.h + 30:
                return True
        return False

    def _handle_long_range_occlusion(self, ctx: Context) -> bool:
        """长手遮挡后撤：锁定目标消失且最后已知位置很近 → 判定被遮挡，用最后位置后撤。

        与短手 _occluded_target 对应：短手遮挡后"继续原地攻击"，长手遮挡后"后撤拉开"。
        返回 True 表示已处理后撤，False 表示目标真消失（应清锁）。
        """
        t = self._target_monster
        if t is None:
            return False
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return False
        sx, sy = foot
        if abs(sy - (t.y + t.h)) > getattr(self.config, "attack_range_y", 60):
            return False  # 跨层 → 非遮挡
        if abs(sx - t.center[0]) >= OCCLUSION_RETREAT_X:
            return False  # 最后位置太远 → 已死/离开，非遮挡
        self._occluded_retreat_frames += 1
        if self._occluded_retreat_frames > self._frames(OCCLUSION_RETREAT_MAX_SECONDS):
            self._log("[后撤] 目标被遮挡后撤超时，判定已消失")
            return False
        self._fsm.transition(State.CHASING)
        self._retreat(ctx, t, reason="目标被遮挡")
        return True

    def _handle_no_monster_retreat(self, ctx: Context) -> bool:
        """画面无怪时的后撤决策（长手遮挡 + 掉血兜底）。

        返回 True 表示本帧处于后撤中（维持或新触发），False 表示应探索。
        仅长手(远程)生效；短手走独立的 _occluded_target 逻辑。
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return False
        if self._is_stand_mode():
            return False  # 站桩模式：不做任何后撤

        # 1) 掉血后撤持续中：继续按当前方向退，限时后停止
        if self._retreating and self._hp_retreat_hold_frames > 0:
            self._hp_retreat_hold_frames -= 1
            if self._hp_retreat_hold_frames > 0:
                direction = self._held_key if self._held_key in ("left", "right") \
                    else ("right" if self._face_dir == "left" else "left")
                self._retreat(ctx, direction=direction)
                return True
            self._retreating = False
            self._release_move()
            return False

        # 2) 遮挡后撤：有锁定目标、处于战斗中、最后位置很近
        if self._target_monster is not None and self._in_combat():
            if self._handle_long_range_occlusion(ctx):
                return True

        # 3) 掉血兜底：最近掉血且未在后撤中 → 触发一次后撤
        if self._hp_drop_active_frames > 0 and not self._retreating:
            self._hp_drop_active_frames = 0
            self._hp_retreat_hold_frames = self._frames(HP_RETREAT_HOLD_SECONDS)
            self._fsm.transition(State.CHASING)
            if self._target_monster is not None:
                self._retreat(ctx, self._target_monster, reason="掉血兜底")
            else:
                direction = "right" if self._face_dir == "left" else "left"
                self._retreat(ctx, direction=direction, reason="掉血兜底")
            return True

        return False

    # =========================================================================
    # 攻击
    # =========================================================================

    def _count_attack_dir_monsters(self, ctx: Context,
                                   target: Optional[Detection]) -> int:
        """统计"攻击方向"内的同平台怪物数（用于 AOE 技能选择）。

        "攻击方向" = 角色朝向正前方（_face_dir 一侧）。计入条件（全部满足）：
          1. 与角色同平台（脚底垂直差 ≤ attack_range_y）
          2. 在角色朝向正前方（怪中心x 在 _face_dir 一侧）
        朝向未知时回退为"朝锁定目标"方向；仍无法确定则默认向右。
        """
        foot = self._effective_self_pos(ctx)
        if foot is None:
            return 0
        sx, sy = foot

        if self._face_dir in ("left", "right"):
            direction = self._face_dir
        elif target is not None:
            direction = "right" if target.center[0] >= sx else "left"
        else:
            direction = "right"

        ry = getattr(self.config, "attack_range_y", 60)
        count = 0
        for m in ctx.monsters:
            if self._in_ghost_zone(m):              # 残影不算"前方有怪"
                continue
            mx = m.center[0]
            if abs(sy - (m.y + m.h)) > ry:          # 跨层不计
                continue
            if direction == "right" and mx <= sx:   # 不在正前方
                continue
            if direction == "left" and mx >= sx:
                continue
            count += 1
        return count

    def _attack(self, ctx: Context, target: Detection):
        """在攻击范围内原地释放技能（攻击时不移动）。

        长手模式: 在攻击距离外原地释放远程技能。
        短手模式: 在极近距离释放近战技能，角色贴脸攻击。

        攻击前【每次】都判断怪物在角色左边还是右边，朝向不对就先转向再打。
        转向不再用 30~90ms 的 press_key 短按，而是走 _face_step：单独占一帧
        （本帧不按技能键）、方向键按住 120~180ms，并按间隔重申 + 生效自检——
        因为游戏会忽略攻击动画期间的按键，和技能键挤在同一帧的短按会被吞掉，
        被吞之后朝向记忆就会和游戏内朝向长期脱钩（实测连续 12 秒朝反方向放技能）。

        【距离守卫】攻击前统一校验：必须满足两个条件才发动攻击：
        1. 同一平台：垂直差 ≤ attack_range_y
        2. 水平差 < 当前生效攻击距离
        否则直接放弃攻击（不释放技能），避免怪物不在附近时一直空打。
        """
        self._release_move()  # 攻击时保持不动

        # ---- 攻击动画锁：上次攻击还在动画中 → 本帧不发攻击/击退/转向键 ----
        # 游戏会忽略动画期间的按键，此时硬发只会被吞（白按还挤掉能生效的输入）。
        if self._attack_busy():
            return

        # ---- 距离守卫（规则1/2）：脚底判同平台 + 中心x判攻击距离 ----
        # 不满足 → 直接放弃攻击，避免怪物不在附近时一直空打/乱跑。
        # 攻击中带滞回（_can_attack）：轻微抖动/怪物中心波动不中断攻击，
        # 防止"打一下就跑"的横跳。
        if target is not None:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                # OCR 定位失败且不在站定攻击中 → 无法判断距离，放弃攻击
                self._log("[攻击] 无法定位自身位置，放弃攻击")
                return
            if not self._can_attack(ctx, target):
                sx, sy = foot
                mx = target.center[0]
                self._log(
                    f"[攻击] 不同平台或怪物不在攻击范围内"
                    f"(水平={abs(sx - mx)} 垂直={abs(sy - (target.y + target.h))}"
                    f" 容差={getattr(self.config, 'attack_range_y', 60)})，"
                    f"停止攻击"
                )
                return

            # ---- 近身威胁扫描（击退提速核心）：扫"面朝方向"的全部同平台怪 ----
            # 只看锁定目标的话，锁定目标远在射程边缘而同侧另一只怪已贴身时，
            # 击退永远不会触发。威胁必须与角色面朝方向同侧（击退是方向性技能，
            # 不转身打背后），且只在"击退就绪"时才认。
            threat = self._nearest_knockback_threat(ctx, foot)

            # ---- 转向（与施法分帧）：朝向不对就本帧只转向、不放技能 ----
            if self._face_step(ctx, target):
                return

            # ---- 近身击退：有同侧近身威胁打威胁，否则照旧看锁定目标 ----
            # 只吃掉本帧这一次技能按键；冷却期内 _try_knockback 返回 False，
            # 普通技能照常输出（所以不需要追踪怪有没有被推开）。
            kb_target = threat if threat is not None else target
            if self._try_knockback(kb_target, foot[0]):
                return

        # ---- 技能选择 ----
        # 站桩模式：只放技能2(群攻)，冷却中就等下一帧（见 _cast_attack_skill）
        if self._stand_skill2_only():
            self._cast_attack_skill()
            return

        # 普通模式：攻击方向怪数 ≥ 阈值 → 连发 AOE_BURST_COUNT 发技能2(爆炸箭)，
        #      爆炸箭冷却空档用技能1兜底；否则只放技能1
        if self._aoe_burst_left == 0 \
                and self._count_attack_dir_monsters(ctx, target) >= AOE_MONSTER_COUNT_THRESHOLD:
            self._aoe_burst_left = AOE_BURST_COUNT

        if self._aoe_burst_left > 0:
            # 爆炸箭连发中：成功放出一发才递减；冷却空档连击兜底
            if self._cast_skill(force_index=1):
                self._aoe_burst_left -= 1
            else:
                self._cast_skill(force_index=0)
        else:
            self._cast_skill(force_index=0)

    def _tab_attack(self, ctx: Context, target: Detection = None):
        """Tab 选怪 + 原地攻击（兜底方案，不移动）。

        与 _attack 相同的距离守卫：能拿到自身位置和目标时，
        水平差/垂直差超限就停止攻击，防止怪物不在附近空打。
        无法定位自身时直接放弃攻击，不盲打。
        """
        self._release_move()
        if self._attack_busy():
            return

        # ---- 距离守卫（规则1/2）：脚底判同平台 + 中心x判攻击距离 ----
        # 无法定位自身 → 放弃攻击（不盲打）
        if target is not None:
            foot = self._effective_self_pos(ctx)
            if foot is None:
                self._log("[攻击] 无法定位自身位置，放弃攻击")
                return
            if not self._can_attack(ctx, target):
                sx, sy = foot
                mx = target.center[0]
                self._log(
                    f"[攻击] 不同平台或怪物不在攻击范围内"
                    f"(水平={abs(sx - mx)} 垂直={abs(sy - (target.y + target.h))}"
                    f" 容差={getattr(self.config, 'attack_range_y', 60)})，"
                    f"停止攻击"
                )
                return

        self.executor.press_key(self.config.target_key, cooldown=0.8)
        self._cast_skill()

    # =========================================================================
    # 攀爬
    # =========================================================================

    def _rope_reachable(self, rope: Detection, foot_y: int) -> bool:
        """判断人物脚底能否够到绳底端（可跳抓）。

        绳底端 y（rope.y + rope.h）最多高出人物脚底 ROPE_REACH_Y。
        若绳底远高于人物脚底（如挂在半空/画面顶部的高绳），
        人物跳抓不到，爬不了这条绳。
        """
        rope_bottom = rope.y + rope.h
        return (foot_y - rope_bottom) <= ROPE_REACH_Y

    def _try_climb(self, ctx: Context, target: Detection,
                   planned_rope: Optional[Detection] = None) -> bool:
        """攀爬追怪：走到绳索正下方 → 跳跃 + 按住上键爬绳。

        流程:
          1. 脱离绳索后的横向走出阶段（不重新抓绳）
          2. 用路径规划选定的绳索（est.climb_rope，基于当前截图
             YOLO 分析结果）；规划绳不在附近时回退找最近绳索
          3. 【对准判定】人物中心与绳索中心是否在同一竖直轴线
             （X 差 <= 5px）→ 否，先水平移动到绳索正下方
          4. 已对准 → 按跳跃 + 按住上键沿绳索向上爬
          5. 爬到怪物所在高度（人物中心与怪物中心 Y 差 <= 30px）
             或爬到绳顶 → 停止爬绳，横向走出绳索继续追击

        返回 True 表示找到了绳索并执行了动作，False 表示没找到。
        """
        # 用人物中心点（不是脚底），回退脚底
        player = ctx.self_center or ctx.self_position
        if player is None:
            self._release_move()
            return False
        sx, sy = player
        tx = target.center[0]

        # 刚爬完绳，横向走出绳索（此阶段不重新抓绳）
        if self._climb_exit_frames > 0:
            self._climb_exit_frames -= 1
            if tx > sx:
                self._hold_move("right")
            else:
                self._hold_move("left")
            return True

        # 找要爬的绳索：优先用路径规划选定的绳（YOLO 分析出的路线）。
        # 只要水平距离在搜索范围内就视为附近，不再用"绳底够不着"
        # 过滤——YOLO 绳框常只覆盖绳子上段，绳底远高于人物脚底是
        # 检测框不完整造成的，实际地图绳索通常垂到地面，都能爬上。
        foot_y = ctx.self_position[1] if ctx.self_position is not None else sy
        nearest_rope = None
        min_dist = float("inf")
        if planned_rope is not None:
            rxd = abs(planned_rope.center[0] - sx)
            if rxd < ROPE_SEARCH_RANGE_X:
                nearest_rope = planned_rope
                min_dist = rxd
        if nearest_rope is None:
            for r in ctx.ropes:
                rx = r.center[0]
                dist = abs(rx - sx)
                if dist < ROPE_SEARCH_RANGE_X and dist < min_dist:
                    nearest_rope = r
                    min_dist = dist

        if nearest_rope is None:
            self._climbing = False
            self._release_move()
            return False

        rx, ry = nearest_rope.center

        # ---- 阶段 1: 对准判定（人物中心与绳索中心同一竖直轴线）----
        if not self._climbing:
            if abs(rx - sx) > CLIMB_ALIGN_TOLERANCE:
                # 不在绳索正下方 → 水平移动对准
                if rx > sx:
                    self._hold_move("right")
                else:
                    self._hold_move("left")
                return True
            # 人物中心与绳索中心在同一竖直轴线（±5px）→ 跳跃 + 按住上键爬绳
            self._release_move()
            self._climbing = True
            self._log("[攀爬] 对准绳索，跳跃并开始攀爬")
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._hold_move("up")
            return True

        # ---- 阶段 2: 正在爬绳 ----
        ty = target.center[1]

        # 到达条件: 人物中心与怪物中心 Y 差 <= 30px（已爬到怪物所在层）
        if abs(sy - ty) <= SAME_LEVEL_Y_TOLERANCE:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到达怪物所在高度，脱离绳索")
            return True

        # 爬到绳顶（人物中心已接近绳索顶部）仍没到怪物高度 → 停止，横向走出
        if sy <= ry - (nearest_rope.h / 2) + 10:
            self._climbing = False
            self._climb_exit_frames = CLIMB_EXIT_FRAMES
            self._release_move()
            self._log("[攀爬] 已到绳顶仍追不上，脱离绳索")
            return True

        # 还没到 → 继续按住上键向上爬（日志限频，防刷屏）
        self._hold_move("up")
        self._climb_log_count += 1
        if self._climb_log_count % 15 == 1:
            self._log("[攀爬] 沿绳索向上")
        return True

    # =========================================================================
    # 跨层跳跃追击（无绳索时）
    # =========================================================================

    def _jump_chase(self, ctx: Context, target: Detection,
                    est: Optional[PathEstimate] = None):
        """无绳索时，按平台路径逐层跳跃追击怪物（动态路线规划）。

        核心思路（与 YOLO 检测到的平台/绳索动态规划路线）：
          1. 优先：怪物所在平台高度差 <= 跳跃高度 → 直接跳上
          2. 否则：找到"往怪物方向、人物能跳上去"的下一层平台，
             先横向走到其正下方/附近，再起跳，一层一层往上/靠近
          3. 若 BFS 已给出规划路径(est.path_floors)，优先用它确定
             中间过渡平台；否则动态在 ctx.floors 里找可跳平台
          4. 脚下没地板（边缘/空中）→ 下落

        平台高度按"最上面的 y 坐标"（floor.y，top_y）计算。
        """
        if ctx.self_position is None:
            self._release_move()
            return
        sx, sy = ctx.self_position
        tx = target.center[0]
        # 怪物站立高度用 bbox 底部（脚底），比框中心更接近其所在平台
        ty = target.y + target.h

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[跳跃追击] 卡住了，起跳")
            self._release_move()
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 怪物所在平台的 top_y
        target_top = self._find_floor_top(ctx, tx, ty)
        target_top = target_top if target_top is not None else ty

        # 人物脚下的平台 top_y
        foot_top = self._find_floor_top(ctx, sx, sy)
        on_floor = self._has_floor_under(ctx, ctx.self_position)

        # ---- 情况1: 可以直接跳上目标平台 ----
        if on_floor and target_top is not None and ty < sy - 5:
            # 怪物在上层，高度差在跳跃高度内 → 横向走到其下方后起跳
            if (sy - target_top) <= JUMP_HEIGHT:
                # 对齐用"怪物所在平台"（比怪物 bbox 更可靠），找不到则用怪物
                goal_floor = self._find_floor_object(ctx, (tx, ty))
                align_ref = goal_floor if goal_floor is not None else target
                need_x = self._align_x_for_jump(sx, tx, align_ref, ctx)
                if need_x:
                    return  # 还在水平对准中
                self._log(
                    f"[跳跃追击] 目标平台 top_y={target_top}，"
                    f"高度差 {sy - target_top}px <= {JUMP_HEIGHT}px，起跳"
                )
                self._release_move()
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
                return

        # ---- 情况2: 高度差太大，需逐层跳（动态规划中间平台）----
        if on_floor and target_top is not None and ty < sy - 5:
            next_floor = self._pick_next_floor(ctx, target, est)
            if next_floor is not None:
                nx, ny = next_floor.center
                ntop = next_floor.y
                # 需要走到该平台上方/附近才能跳上去
                align = self._align_x_for_jump(sx, nx, next_floor, ctx,
                                               target_ty=ntop)
                if align:
                    return  # 还在水平对准中间平台
                # 高度差在跳跃高度内 → 起跳
                if (sy - ntop) <= JUMP_HEIGHT:
                    self._log(
                        f"[跳跃追击] 逐层跳: 目标平台top={ntop}，"
                        f"高度差 {sy - ntop}px <= {JUMP_HEIGHT}px，起跳"
                    )
                    self._release_move()
                    self.executor.press_key(self.config.jump_key, cooldown=1.0)
                    return
                # 中间平台也不够 → 继续往怪物方向走（保持按住方向键）
                self._hold_toward(sx, tx)
                return

            # 找不到中间平台 → 继续往怪物方向走
            self._hold_toward(sx, tx)
            return

        # ---- 情况3: 脚下没地板（边缘/空中）→ 下落 ----
        if not on_floor:
            self._log("[跳跃追击] 脚下没地板，下落")
            self.executor.press_key("down", cooldown=0.3)
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            return

        # 其他情况：往怪物方向走
        self._hold_toward(sx, tx)

    def _hold_toward(self, sx: int, tx: int):
        """按住方向键朝目标 x 移动（移动时按住方向键）。"""
        if tx > sx + 10:
            self._hold_move("right")
        elif tx < sx - 10:
            self._hold_move("left")
        else:
            self._release_move()

    def _align_x_for_jump(self, sx: int, tx: int, floor: Detection,
                          ctx: Context, target_ty: Optional[int] = None) -> bool:
        """把人物横向移动到目标平台覆盖范围内，准备起跳。

        Returns:
            True 表示还在水平对准（需要继续移动，本次不应跳）；
            False 表示已经对齐（可以起跳）。
        """
        # 目标平台的横向覆盖范围（向内收缩 10px，避免站边缘起跳）
        f_left = floor.x + 10
        f_right = floor.x + floor.w - 10

        # 已站在平台覆盖范围内
        if f_left <= sx <= f_right:
            self._release_move()
            return False

        # 平台在右边 → 向右走；平台在左边 → 向左走
        if f_right < sx:
            self._hold_move("left")
        else:
            self._hold_move("right")
        return True

    def _pick_next_floor(self, ctx: Context, target: Detection,
                         est: Optional[PathEstimate] = None) -> Optional[Detection]:
        """规划"下一步跳往的中间平台"。

        优先用 BFS 规划路径(path_floors)中的人物脚下平台的下一个平台；
        否则动态在 ctx.floors 中找：位于人物上方、怪物方向侧、
        高度差 <= JUMP_HEIGHT、能被人物跳到的最远/最近平台。
        """
        # 优先用 BFS 规划路径
        if est is not None and est.path_floors:
            foot = self._find_floor_object(ctx, ctx.self_position)
            if foot is not None:
                for f in est.path_floors:
                    if self._is_same_floor(foot, f):
                        # 找到当前平台在序列中的位置，返回下一层
                        idx = est.path_floors.index(f)
                        if idx + 1 < len(est.path_floors):
                            nxt = est.path_floors[idx + 1]
                            # 中间平台必须高于当前，且高度差可跳
                            if nxt.y < foot.y - 5 and (foot.y - nxt.y) <= JUMP_HEIGHT:
                                return nxt
                            continue
            # BFS 路径不适用（人物已偏离起始平台）→ 回退动态搜索

        if ctx.self_position is None:
            return None
        sx, sy = ctx.self_position
        tx = target.center[0]

        # 动态搜索：人物上方、高度差可跳、在怪物方向一侧的平台
        candidates = []
        for f in ctx.floors:
            # 必须在人物上方（更高的平台，y 更小）
            if f.y >= sy - 5:
                continue
            if (sy - f.y) > JUMP_HEIGHT:
                continue
            # 横向位置应靠近人物或位于怪物方向
            if abs(f.center[0] - sx) > PLATFORM_JUMP_GAP_X:
                continue
            candidates.append(f)

        if not candidates:
            return None
        # 选水平距离最近（先到达）的平台
        candidates.sort(key=lambda f: abs(f.center[0] - sx))
        return candidates[0]

    def _is_same_floor(self, a: Detection, b: Detection) -> bool:
        """判断两个平台检测是否为同一平台（按位置重叠）。"""
        if a is None or b is None:
            return False
        ax0, ay0 = a.x, a.y
        ax1, ay1 = a.x + a.w, a.y + a.h
        bx0, by0 = b.x, b.y
        bx1, by1 = b.x + b.w, b.y + b.h
        ox = min(ax1, bx1) - max(ax0, bx0)
        oy = min(ay1, by1) - max(ay0, by0)
        return ox > 5 and oy > 5

    def _find_floor_object(self, ctx: Context, pos) -> Optional[Detection]:
        """找到覆盖人物脚底 (x, y) 的平台对象，找不到返回 None。"""
        if pos is None:
            return None
        x, y = pos
        best = None
        best_key = float("inf")
        for f in ctx.floors:
            if f.x <= x <= f.x + f.w:
                if f.y - 30 <= y <= f.y + f.h + 30:
                    d = abs(f.y - y)
                    if d < best_key:
                        best_key = d
                        best = f
        return best

    def _find_floor_top(self, ctx: Context, x: int, y: int) -> Optional[int]:
        """找到覆盖 (x, y) 的平台，返回其"最上面的 y"（top_y）。

        找不到返回 None。
        """
        best = None
        best_key = float("inf")
        for f in ctx.floors:
            if f.x <= x <= f.x + f.w:
                if f.y - 30 <= y <= f.y + f.h + 30:
                    d = abs(f.y - y)
                    if d < best_key:
                        best_key = d
                        best = f.y
        return best

    # =========================================================================
    # 探索
    # =========================================================================

    def _default_attack(self, ctx: Context):
        """站桩默认攻击：没有有效目标（模型漏检）时也按技能键盲打。

        不移动、不转向、不触发 AOE 连发逻辑。
        技能走 _cast_attack_skill()：站桩且 stand_skill2_only 时只放技能2(群攻)，
        否则两个技能交替（各自受冷却约束）。
        """
        self._release_move()
        if self._attack_busy():
            return
        self._cast_attack_skill()

    def _explore(self, ctx: Context):
        """画面里没怪时，往一个方向走探索。

        行为:
          - 往探索方向走
          - 遇到平台边缘（脚下没地板）就跳
          - 卡住时反向走
          - 长时间没遇到怪就换方向
        站桩模式：无怪就站着不动（不探索、不跳跃）；开启默认攻击时改为盲打。
        """
        if self._is_stand_mode():
            self._release_move()
            if getattr(self.config, "stand_default_attack", True):
                self._default_attack(ctx)
            return
        self._explore_frame_count += 1

        if self._stuck_counter >= STUCK_FRAMES:
            self._log("[探索] 卡住了，跳跃并反向")
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            self._stuck_counter = 0
            return

        # 走到地图尽头（贴近画面边缘）或活动范围的分界线 → 掉头
        if self._at_edge(ctx, self._explore_direction):
            at_line = self._at_range_line(ctx, self._explore_direction)
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self._explore_frame_count = 0
            self._log(f"[探索] 走到{'分界线' if at_line else '地图边缘'}，掉头 → {self._explore_direction}")

        # 长时间探索没遇到怪，换方向
        elif self._explore_frame_count >= EXPLORE_DIRECTION_SWITCH_FRAMES:
            self._explore_direction = "left" if self._explore_direction == "right" else "right"
            self._explore_frame_count = 0
            self._log(f"[探索] 换方向 → {self._explore_direction}")

        # 检测脚下是否有地板
        if ctx.self_position and not self._has_floor_under(ctx, ctx.self_position):
            self._log("[探索] 脚下没地板，跳跃")
            self.executor.press_key(self.config.jump_key, cooldown=1.0)
            return

        # 按住方向键持续往前走
        self._hold_move(self._explore_direction)

    # =========================================================================
    # 活动范围（只在地图左半边 / 右半边）
    # =========================================================================

    def _range_mode(self) -> Optional[str]:
        """限制在哪一边："left" / "right"，不限制返回 None。"""
        mode = getattr(self.config, "range_limit", "none")
        return mode if mode in ("left", "right") else None

    def _range_step(self, ctx: Context) -> bool:
        """每帧算一次分界线、去掉线外的怪、走出范围就走回来。返回 True 表示本帧到此为止。

        镜头跟着角色走，只看画面不知道自己在地图哪里，所以用小地图上的黄点：
        黄点离分界线多远 × 小地图比例（minimap_scale）= 画面里分界线离自己多远。
        比例不准只影响离分界线远时对怪的判断，走到线上时距离是 0，人不会因此过线。
        """
        self._range_line_x = None
        mode = self._range_mode()
        if mode is None:
            return False
        now = time.time()
        split = max(0.0, min(1.0, float(getattr(self.config, "range_split", 0.5))))
        side = "左" if mode == "left" else "右"
        if not self._range_announced:
            self._range_announced = True
            self._log(f"[范围] 只在{side}半边活动，分界线在小地图 {split:.0%} 处"
                      f"（小地图 1px 按画面 {self._range_scale():g}px 算）")
        if ctx.minimap_x is not None and ctx.minimap_width:
            # 镜头跟着人走，走路时画面里的 x 也可能不变：黄点动了才说明真的在走，不算卡住
            if self._range_dot is not None and abs(ctx.minimap_x - self._range_dot[0]) >= 0.5:
                self._stuck_counter = 0
            self._range_dot = (ctx.minimap_x, ctx.minimap_width)
            self._range_dot_at = now
            if self._range_lost:
                self._range_lost = False
                self._log("[范围] 重新在小地图上认出自己，恢复范围限制")
        stand = self._is_stand_mode()
        foot = ctx.self_position or self._last_self_pos
        if self._range_dot is None or now - self._range_dot_at > RANGE_DOT_GRACE_SECONDS or foot is None:
            return self._range_hold(ctx, foot, now, stand)

        dot_x, width = self._range_dot
        # 过线距离（画面 px）：正 = 在线外，负 = 在线内
        past = (dot_x - width * split) * self._range_scale() * (1 if mode == "left" else -1)
        line = foot[0] + past * (-1 if mode == "left" else 1)
        self._range_line_x = line

        if now - self._range_status_at >= RANGE_STATUS_SECONDS:
            self._range_status_at = now
            where = f"在线外 {int(past)}px" if past > 0 else f"离分界线 {int(-past)}px"
            self._log(f"[范围] 小地图上你在 {dot_x / width:.0%}（分界 {split:.0%}），{where}")

        self._range_filter_monsters(ctx, mode, line, foot, now)
        if stand:
            return False  # 站撸不走动，只是不打线外的怪；站位自己选在范围内
        return self._range_walk_back(ctx, mode, past, now)

    def _range_scale(self) -> float:
        """小地图 1px 对应画面多少 px。游戏窗口整体缩放时小地图和画面一起变，这个比例不随分辨率变。"""
        return max(1.0, float(getattr(self.config, "minimap_scale", 16) or 16))

    def _range_filter_monsters(self, ctx: Context, mode: str, line: float,
                               foot: Tuple[int, int], now: float):
        """把分界线外的怪从这一帧的怪物列表里去掉：之后选目标、追击、后撤就都看不到它们。"""
        sign = 1 if mode == "right" else -1
        tol = self._px(RANGE_LINE_TOLERANCE)
        inside = [m for m in ctx.monsters if (m.center[0] - line) * sign >= -tol]
        if len(inside) == len(ctx.monsters):
            return
        outside = [m for m in ctx.monsters if (m.center[0] - line) * sign < -tol]
        ctx.monsters = inside
        # 锁定的怪走出了范围：马上放手，不当成"被挡住/漏检"继续打它最后的位置
        t = self._target_monster
        if t is not None and any(self._is_same_monster(t, m) for m in outside):
            self._target_monster = None
            self._occluded_frames = 0
            self._log("[范围] 锁定的怪走到分界线外，不打了，换范围内的")
        if now - self._range_note_at >= RANGE_NOTE_SECONDS:
            self._range_note_at = now
            near = min(outside, key=lambda m: abs(m.center[0] - foot[0]))
            dx = near.center[0] - foot[0]
            self._log(f"[范围] 分界线外有 {len(outside)} 只怪不打（最近的在{'右' if dx > 0 else '左'}边 "
                      f"{abs(dx)}px），范围内 {len(inside)} 只")

    def _range_hold(self, ctx: Context, foot: Optional[Tuple[int, int]],
                    now: float, stand: bool) -> bool:
        """不知道自己在地图哪里：不走动（免得走进对方那一半），只打已经够得着的怪，等黄点回来。"""
        self._range_returning = 0.0
        self._range_over_frames = 0
        if not self._range_lost:
            self._range_lost = True
            self._range_note_at = 0.0
        if now - self._range_note_at >= RANGE_NOTE_SECONDS:
            self._range_note_at = now
            if not getattr(self.config, "minimap_region", None):
                self._log("[范围] 还没框选小地图，不知道自己在地图哪里：先不走动，只打够得着的怪。"
                          "请点「框选小地图」框住小地图的地图部分")
            else:
                self._log("[范围] 小地图上找不到自己的黄点：先不走动，只打够得着的怪。"
                          "看一下小地图是不是收起来了、被挡住了，或者框选的位置不对（换地图要重新框）")
        if stand:
            return False
        if foot is None:
            self._release_move()
            return True
        ctx.monsters = [m for m in ctx.monsters if self._range_in_reach(foot, m)]
        if ctx.monsters:
            return False
        self._release_move()
        self._target_monster = None
        self._fsm.transition(State.IDLE)
        return True

    def _range_in_reach(self, foot: Tuple[int, int], m: Detection) -> bool:
        """这只怪站着不动就打得到：同一层、在攻击距离内、不在残影位置。"""
        if not self._same_platform(foot[1], m.y + m.h) or self._in_ghost_zone(m):
            return False
        if getattr(self.config, "attack_type", "long") == "short":
            return abs(foot[0] - self._melee_edge_x(foot[0], m)) <= self._get_attack_range()
        return abs(m.center[0] - foot[0]) < self._get_attack_range()

    def _range_walk_back(self, ctx: Context, mode: str, past: float, now: float) -> bool:
        """走出范围（被击退、开始时就在线外）→ 走回线内。返回 True 表示本帧在走（不打怪）。"""
        margin = self._px(RANGE_EDGE_MARGIN)
        if self._range_returning:
            if past <= -margin:
                self._range_returning = 0.0
                self._range_over_frames = 0
                self._release_move()
                self._log("[范围] 回到范围内，继续打怪")
                return False
            if self._stuck_counter >= STUCK_FRAMES:
                self._stuck_counter = 0
                self._range_stuck += 1
                if self._range_stuck >= RANGE_RETURN_GIVEUP:
                    self._range_returning = 0.0
                    self._range_over_frames = 0
                    self._range_retry_at = now + RANGE_RETRY_SECONDS
                    self._release_move()
                    self._log(f"[范围] 走不回去（被挡住？还在线外 {int(past)}px），"
                              f"先打够得着的怪，{RANGE_RETRY_SECONDS:.0f} 秒后再试")
                    return False
                self._log("[范围] 走回去时卡住了，跳一下")
                self.executor.press_key(self.config.jump_key, cooldown=1.0)
        else:
            if past <= self._px(RANGE_RETURN_START) or now < self._range_retry_at:
                self._range_over_frames = 0
                return False
            self._range_over_frames += 1
            if self._range_over_frames < RANGE_RETURN_CONFIRM_FRAMES:
                # 确认的这几帧也不打：人在线外，朝里出手会打到挡在中间的线外的怪
                self._release_move()
                return True
            self._range_returning = now
            self._range_stuck = 0
            self._stuck_counter = 0
            self._target_monster = None
            self._log(f"[范围] 走出范围了（过线 {int(past)}px），往{'左' if mode == 'left' else '右'}走回去")
        self._fsm.transition(State.CHASING)
        self._hold_move(mode)
        return True

    def _at_range_line(self, ctx: Context, direction: str) -> bool:
        """再往 direction 走就要过分界线了。"""
        line = self._range_line_x
        if line is None or ctx.self_position is None:
            return False
        margin = self._px(RANGE_EDGE_MARGIN)
        mode = self._range_mode()
        if mode == "left" and direction == "right":
            return ctx.self_position[0] >= line - margin
        if mode == "right" and direction == "left":
            return ctx.self_position[0] <= line + margin
        return False

    def _at_edge(self, ctx: Context, direction: str) -> bool:
        """再往 direction 走就贴到画面边缘（地图尽头）或活动范围的分界线了。"""
        if self._at_range_line(ctx, direction):
            return True
        if ctx.self_position is None or not ctx.frame_width:
            return False
        margin = self._px(EDGE_MARGIN)
        x = ctx.self_position[0]
        if direction == "right":
            return x >= ctx.frame_width - margin
        if direction == "left":
            return x <= margin
        return False

    # =========================================================================
    # 技能释放
    # =========================================================================

    def _stand_skill2_only(self) -> bool:
        """站桩模式是否"攻击只用技能2(群攻)"。"""
        return (self._is_stand_mode()
                and bool(getattr(self.config, "stand_skill2_only", True)))

    # =========================================================================
    # 近身击退（长手专用）
    # =========================================================================

    def _attack_lock_seconds(self) -> float:
        """攻击动画锁时长（秒），配置 attack_lock_seconds，0=关闭。"""
        try:
            v = float(getattr(self.config, "attack_lock_seconds",
                              ATTACK_LOCK_SECONDS_DEFAULT) or 0)
        except (TypeError, ValueError):
            v = ATTACK_LOCK_SECONDS_DEFAULT
        return max(0.0, v)

    def _attack_busy(self) -> bool:
        """是否处于攻击动画锁内（上次攻击键按下后的动画时长内）。"""
        return time.time() < self._attack_busy_until

    def _mark_attack_pressed(self):
        """按出一次攻击键（技能/击退）后开启动画锁。

        游戏会忽略攻击动画期间的按键（代码在 buff 施法窗口里用到了同一条
        结论）。实测每秒发 4+ 次按键时大部分被吞——表现为"释放不出技能、
        击退按了没放出来"。锁内不发任何攻击/击退/转向键，让每次按键都落地。
        加血/加蓝/buff/移动不受影响（在 decide() 更早的阶段处理）。
        """
        self._last_cast_at = time.time()  # 换频道判"这层多久没打过怪"用
        lock = self._attack_lock_seconds()
        if lock > 0:
            self._attack_busy_until = time.time() + lock

    def _knockback_conf(self) -> Optional[dict]:
        """读取击退技能配置；未配置（按键为空）→ None（功能关闭）。

        配置来自 config.knockback_skill = {name, key, cooldown, range}，
        任何一项缺失/非法都用默认值兜底（配置文件可能被手工改坏）。
        """
        conf = getattr(self.config, "knockback_skill", None) or {}
        if not isinstance(conf, dict):
            return None
        key = str(conf.get("key", "") or "").strip()
        if not key:
            return None      # 没配按键 = 不启用（保持原行为，零风险）
        try:
            cooldown = float(conf.get("cooldown", KNOCKBACK_COOLDOWN_DEFAULT) or 0)
        except (TypeError, ValueError):
            cooldown = KNOCKBACK_COOLDOWN_DEFAULT
        try:
            rng = int(conf.get("range", KNOCKBACK_RANGE_DEFAULT) or 0)
        except (TypeError, ValueError):
            rng = KNOCKBACK_RANGE_DEFAULT
        return {
            "name": str(conf.get("name", "") or key),
            "key": key,
            "cooldown": max(0.0, cooldown),
            "range": max(1, rng),
        }

    def _nearest_knockback_threat(self, ctx: Context,
                                  foot: Tuple[int, int]) -> Optional[Detection]:
        """击退范围内的最近同平台怪，且必须在角色面朝的方向一侧。

        与只看锁定目标的差别：锁定目标可能远在射程边缘，而同侧另一只怪已经
        贴身——逐帧扫描全部怪才能第一时间发现近身威胁（锁定目标不死亡/不
        消失就一直被锁定，贴身的怪永远轮不到）。

        过滤条件（全部满足才算威胁）：
          · 击退已启用且当前可用（冷却已过）——冷却中不认威胁，避免
            "为它调整朝向后却打不出击退"；
          · 与角色同平台（脚底垂直差 ≤ attack_range_y）；
          · 在角色面朝方向一侧（按 _face_dir；朝向未知时不认，
            先由转向流程建立朝向，不盲打）；
          · 与角色中心 x 距离 < knockback_skill.range。
        站桩模式：_in_front 自动按站桩朝向过滤（只认正前方）。

        Returns:
            最近的威胁怪；没有则 None。
        """
        conf = self._knockback_conf()
        if conf is None or foot is None:
            return None
        if not self.executor.can_press(conf["key"], conf["cooldown"]):
            return None          # 击退冷却中：不认威胁（本帧照常普攻）
        sx, sy = foot
        facing = self._face_dir
        if facing not in ("left", "right"):
            return None          # 朝向未知：先由转向流程建立
        ry = getattr(self.config, "attack_range_y", 60)
        best = None
        best_dx = conf["range"]
        for m in ctx.monsters:
            if self._in_ghost_zone(m):
                continue         # 残影不当作近身威胁（别把击退浪费在尸体上）
            if not self._in_front(m, sx):
                continue         # 站桩：按站桩朝向过滤；普通模式恒过
            if abs(sy - (m.y + m.h)) > ry:
                continue         # 跨层不打
            mx = m.center[0]
            if facing == "right" and mx < sx:
                continue         # 不在面朝方向
            if facing == "left" and mx > sx:
                continue
            dx = abs(mx - sx)
            if dx < best_dx:
                best = m
                best_dx = dx
        return best

    def _try_knockback(self, target: Detection, sx: int) -> bool:
        """近身击退：怪贴到触发距离以内时，本帧改放击退技能（原地，不移动）。

        只应在 _attack 的距离守卫之后调用（目标已确认同平台且在攻击范围内）。

        【为什么用它顶替"贴脸后撤"】后撤要真的走位（可能落崖、走进别的怪、
        退完还得走回目标），且后撤期间不放技能；击退技能原地放、顺带造成伤害，
        把怪推开后继续站定输出。

        【为什么只吃掉一帧】"近身"是个持续成立的状态（怪没被推开 / 对击退免疫 /
        按键被游戏吞掉时 dx 一直很小）。本方法只在真正按出击退键的那一帧返回
        True（该帧不再放普通技能）；冷却期内返回 False，调用方照常走普通技能
        选择——普通攻击永远不会被击退挡住，也就不需要追踪"有没有推开"：
        怪被推开后 dx 自然变大，下一帧就落回普通攻击逻辑。

        【为什么只长手生效】短手(近战)的目标就是贴近怪，把怪推开是反效果。

        Returns:
            True 已按出击退技能（调用方本帧不再放普通技能）
        """
        if getattr(self.config, "attack_type", "long") == "short":
            return False
        conf = self._knockback_conf()
        if conf is None or target is None:
            return False
        if self._attack_busy():
            return False      # 动画锁内发击退也会被游戏吞
        # 朝向必须已确认：未知 / 正在纠正（_face_attempt_side 未落地）/
        # 正在等自检生效 → 一律不按。击退是有方向的技能，朝旧朝向放等于白放。
        if self._face_dir not in ("left", "right"):
            return False
        if self._face_attempt_side is not None \
                and self._face_attempt_side != self._face_dir:
            return False
        if self._turn_dir is not None:
            return False
        dx = abs(target.center[0] - sx)
        if dx >= conf["range"]:
            return False
        # 目标必须已在面朝的那一侧：换边防抖窗口内目标刚翻边、角色还没转过去，
        # 此时放击退等于朝反方向空放。
        if (target.center[0] > sx) != (self._face_dir == "right"):
            return False
        if not self.executor.press_key(conf["key"], conf["cooldown"]):
            return False      # 冷却中 / 按键无效 → 让普通技能照常输出
        self._mark_attack_pressed()   # 击退也是一次攻击：同样进入动画锁
        self._log(
            f"[击退] 怪近身 dx={dx}px < {conf['range']}px，"
            f"按 {conf['name']}({conf['key']})"
        )
        return True

    def _cast_attack_skill(self):
        """一次攻击的技能释放（站桩/普通模式的统一入口）。

        站桩 + stand_skill2_only（默认开）：只放技能2（爆炸箭这类群攻），
          技能2 冷却中就是本帧不攻击 —— 不退回技能1（需求）。
          只配了 1 个技能时退化为轮转，避免站桩完全不攻击。
        其余情况：走 _cast_skill() 的轮转（技能1/技能2 交替，各自受冷却约束）。
          调用点：近战攻击、站桩盲打。
        注意：长手普通模式的技能选择不走本方法 —— 它在 _attack 里另有
          AOE 判定（怪数 ≥ 阈值时技能2 连发、空档用技能1 兜底）。
        """
        if self._stand_skill2_only() and len(self.config.skills or []) >= 2:
            self._cast_skill(force_index=1)
            return
        self._cast_skill()

    def _cast_skill(self, force_index: Optional[int] = None) -> bool:
        """释放技能：可指定技能下标，或按轮转顺序释放。

        force_index 给定且有效 → 直接释放该技能（受冷却约束，冷却中则本帧不释放）。
        force_index 为 None → 按原轮转顺序释放（跳过冷却中的技能）。
        实际使用的冷却带 ±随机抖动（见 _jitter_cooldown），避免释放节奏恒定。

        Returns:
            True 表示本帧实际释放了技能，False 表示未释放（冷却中/无技能）。
        """
        skills = self.config.skills
        if not skills:
            return False

        if force_index is not None and 0 <= force_index < len(skills):
            skill = skills[force_index]
            if self.executor.press_key(skill["key"],
                                       self._jitter_cooldown(skill["cooldown"])):
                self._mark_attack_pressed()
                self._casts_on_target += 1   # 残影判定：对当前目标又出手一次
                self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                return True
            return False

        for _ in range(len(skills)):
            skill = skills[self._skill_index % len(skills)]
            self._skill_index += 1
            if self.executor.press_key(skill["key"],
                                       self._jitter_cooldown(skill["cooldown"])):
                self._mark_attack_pressed()
                self._casts_on_target += 1   # 残影判定：对当前目标又出手一次
                self._log(f"[技能] 释放 {skill['name']} ({skill['key']})")
                return True
        return False