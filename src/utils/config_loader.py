"""配置加载与配置实体。

================================================================================
配置系统设计
================================================================================

  配置分为两层:
    - 默认配置: 内置在代码中，提供合理的出厂默认值
    - 用户配置: 保存在 config/user.yaml (或 user.json)，覆盖默认值

  双层覆盖机制:
    1. 先加载默认配置（_defaults()）
    2. 再读取 user.yaml / user.json，用用户配置覆盖同名字段
    3. 最终 Config 对象包含合并后的值

  这样用户只需要配置自己关心的字段，其余使用默认值即可。

================================================================================
YAML / JSON 双格式支持
================================================================================

  优先读取 config/user.yaml（推荐格式，支持中文注释）。
  若 user.yaml 不存在，回退读取 config/user.json（向后兼容）。
  保存时统一写入 user.yaml。

================================================================================
坐标自适应
================================================================================

  问题: 配置文件中的 HP 区域 / 自身偏移 等坐标是参考 1366×768 分辨率
       记录的，但实际运行时窗口可能不同（如 1920×1080）。

  解决: scale_region() 和 scale_offset() 根据参考分辨率与当前帧的比例，
       自动缩放坐标值。

  公式:
    scale_x = 当前帧宽 / 参考帧宽
    scale_y = 当前帧高 / 参考帧高
    scaled_x = x * scale_x
    scaled_y = y * scale_y

================================================================================
YAML 配置字段说明
================================================================================

  window_title:     游戏窗口标题（用于 FindWindow 锁定）
  reference_width:  参考分辨率宽度（坐标记录时的分辨率）
  reference_height: 参考分辨率高度
  fps:              目标帧率
  confidence:       YOLO 检测置信度阈值 (0.0~1.0)
  model_path:       YOLO 模型文件路径
  monster_classes:  怪物类别名（逗号分隔）
  floor_classes:    地板类别名
  rope_classes:    绳索类别名
  player_classes:  玩家类别名（用于自身定位）
  self_name:        自身角色名字（已废弃，改用 YOLO player 定位）
  self_offset:      自身 HP 条底部到脚底的偏移像素数
  hp_region:        HP 条参考区域 [x, y, w, h] 或百分比
  hp_color:         HP 条颜色 [R, G, B]
  hp_tolerance:     HP 条颜色容差
  hp_threshold:     HP 加血阈值 (0.0~1.0)
  hp_key:           加血键
  mp_region:        MP 条参考区域
  mp_color:         MP 条颜色
  mp_tolerance:     MP 条颜色容差
  mp_threshold:     MP 加蓝阈值
  mp_key:           加蓝键
  target_key:       选目标键
  jump_key:         跳跃键
  skills:           技能列表 [{name, key, cooldown}, ...]
  knockback_skill:  近身击退技能 {name, key, cooldown, range}（key 为空=不启用）
  ghost_casts:      对同一目标出手多少次仍未消失就判定为残影/误识别（0=关闭）
  turn_mode:        转向模式 tap（点按 120~180ms + 自检重试）/ hold（按住方向键走出
                    确认位移再松键出手，消除"小碎步搓很久"；实测 A/B 后再改默认）
  pixel_base_height: 像素参数（攻击距离/垂直容差/活动范围阈值等）按此画面高度填写，
                    运行时乘以 当前帧高/此值；0 = 不缩放
  range_limit:      活动范围 none（不限制）/ left（只在地图左半边）/ right（只在地图右半边）
  range_split:      分界线在小地图宽度的什么位置 (0.0~1.0)，默认 0.5 = 正中间
  minimap_region:   小地图的地图部分（百分比 [x, y, w, h]，界面「框选小地图」写入）
  minimap_scale:    小地图 1px 对应画面多少 px（默认 16）
  channel_enabled / channel_idle_min: 自动换频道开关 / 多少分钟没放过技能就换
  channel_hold_key / channel_hold_seconds: 换完按住哪个键 / 按多久（秒）
  quit_enabled / quit_time: 定时结束游戏：点「开始」后下一次到这个钟点（"06:30"）就不打了，按 Esc 退出游戏
  exp_enabled:      打怪时记录经验（读状态栏的 EXP，每分钟一行写到 logs/exp_log/日期.csv）
  exp_read_seconds / exp_log_minutes: 多少秒读一次 / 多少分钟在日志里报一次（0 = 不报）
"""
import os
import sys
from typing import List, Optional, Dict, Any, Tuple

import yaml


# ---- 路径常量 ----
def _bundle_dir() -> str:
    """打包内资源目录（只读）。

    PyInstaller 打包后: sys._MEIPASS
      - onedir 模式  = <程序目录>/_internal
      - onefile 模式 = 运行时临时解压目录（每次启动都重新生成）
    开发环境:          PROJECT_ROOT（项目根目录）
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _app_dir() -> str:
    """应用数据目录（用户可见、可写的目录）。

    模型与配置文件外置于此，方便用户查看/替换，且不会因为
    onefile 解压到临时目录而丢失:
      - PyInstaller 打包后 = exe 所在目录
      - 开发环境          = 项目根目录
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return _bundle_dir()


# 兼容旧名称：BUNDLE_DIR 即原 PROJECT_ROOT
BUNDLE_DIR = _bundle_dir()
PROJECT_ROOT = BUNDLE_DIR
APP_DIR = _app_dir()

# 配置文件：优先 exe 旁边（外置、可持久化），打包内仅作为兜底默认值
CONFIG_DIR = os.path.join(APP_DIR, "config")
DEFAULT_YAML_PATH = os.path.join(CONFIG_DIR, "user.yaml")
DEFAULT_JSON_PATH = os.path.join(CONFIG_DIR, "user.json")


def _resolve_relative_file(raw_path: str) -> str:
    """把相对路径解析成"exe 旁边优先、打包内兜底"的绝对路径。

    - 绝对路径: 直接返回
    - 相对路径: 优先在 APP_DIR（exe 旁边，外置资源）查找；
      不存在时回退 BUNDLE_DIR（打包内，兼容资源被打进包的情况）。
    """
    if not raw_path:
        return raw_path
    if os.path.isabs(raw_path):
        return raw_path
    for base in (APP_DIR, BUNDLE_DIR):
        cand = os.path.normpath(os.path.join(base, raw_path))
        if os.path.isfile(cand):
            return cand
    return os.path.normpath(os.path.join(APP_DIR, raw_path))


def resolve_model_path(raw_path: str) -> str:
    """解析模型路径（规则见 _resolve_relative_file）。"""
    return _resolve_relative_file(raw_path)


def resolve_asset_path(raw_path: str) -> str:
    """解析资源路径（模板图片等，规则同 resolve_model_path）。"""
    return _resolve_relative_file(raw_path)


def config_path() -> str:
    """返回当前生效的配置文件路径。

    优先级:
      1. APP_DIR/config/user.yaml    （exe 旁边，用户配置）
      2. APP_DIR/config/user.json
      3. BUNDLE_DIR/config/user.yaml （打包内默认配置）
      4. BUNDLE_DIR/config/user.json
    都不存在时返回 APP_DIR/config/user.yaml（首次启动后保存到这里）。
    """
    for p in (
        os.path.join(APP_DIR, "config", "user.yaml"),
        os.path.join(APP_DIR, "config", "user.json"),
        os.path.join(BUNDLE_DIR, "config", "user.yaml"),
        os.path.join(BUNDLE_DIR, "config", "user.json"),
    ):
        if os.path.isfile(p):
            return p
    return os.path.join(APP_DIR, "config", "user.yaml")


def _load_yaml(path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置。"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _load_json(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载配置（向后兼容）。"""
    if not os.path.isfile(path):
        return {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_user_config() -> Dict[str, Any]:
    """加载用户配置：优先 exe 旁边（外置）user.yaml，回退打包内默认。"""
    data = _load_yaml(config_path())
    if not data:
        data = _load_json(config_path())
    return data


def _defaults() -> Dict[str, Any]:
    """返回默认配置字典。

    这里定义了所有配置项的默认值。用户 YAML/JSON 中未配置的字段
    会使用这些默认值。

    修改这些默认值会影响所有用户（除非用户在 user.yaml 中覆盖）。

    Returns:
        包含所有默认配置的字典
    """
    return {
        # ---- 窗口 ----
        "window_title": "",
        # ---- 热键 ----
        "start_stop_hotkey": "F6",
        # ---- 分辨率自适应 ----
        # 参考分辨率：坐标录制时的分辨率，所有坐标配置都基于此分辨率
        "reference_width": 1366,
        "reference_height": 768,
        # ---- 性能 ----
        "fps": 13,
        # ---- 执行 ----
        # 键盘注入模式:
        #   使用 SendInput 驱动层模拟真实全局按键，游戏窗口必须在前台。
        #   冒险岛通过 DirectInput 读取全局键盘状态，这是唯一有效的方式。
        "keyboard_mode": "sendinput",
        # ---- 模型 ----
        "confidence": 0.5,
        "model_path": "best.onnx",  # 相对路径，解析到 exe 旁边的模型
        # ---- 类别 ----
        "monster_classes": "monster",
        "floor_classes": "floor",
        "rope_classes": "rope",
        "player_classes": "player",
        # ---- 自身定位 ----
        "self_name": "",  # 已废弃，改用 YOLO player 定位
        "self_offset": 85,  # HP 条底部 → 脚底的偏移像素
        # ---- HP 检测 ----
        "hp_region": None,  # [x, y, w, h] 参考分辨率下的 HP 条区域
        "hp_color": [51, 204, 51],  # HP 条绿色 RGB
        "hp_tolerance": 30,  # 颜色容差
        "hp_threshold": 0.3,  # 低于 30% 时加血
        "hp_key": "f",  # 加血快捷键
        # ---- MP 检测 ----
        "mp_region": None,
        "mp_color": [51, 153, 255],  # MP 条蓝色 RGB
        "mp_tolerance": 30,
        "mp_threshold": 0.3,
        "mp_key": "g",  # 加蓝快捷键
        # ---- 战斗 ----
        "target_key": "tab",  # 选目标键
        "jump_key": "alt",    # 跳跃键
        "attack_type": "long",  # 攻击类型: "long"(长手) / "short"(短手)
        "attack_range": 200,  # 攻击距离（px）：长手专用（exe 界面可修改）；短手固定 50px 不读此配置
        "attack_range_y": 60, # 攻击垂直容差（px）：垂直差小于此值时怪物就在身边，
                              # 即使路径被推算为 rope/jump 也直接攻击，不绕路
        "attack_min_range": 0,  # 长手最小有效射程（px）：角色与怪中心水平距离小于此值时
                               # 弓箭手会"挥弓"而非射箭（伤害大减）→ 后撤拉开距离；
                               # 0=关闭后撤（行为与旧版一致）。只对 attack_type=="long" 生效。
        "attack_lock_seconds": 0.6,  # 攻击动画锁（秒）：按出一次攻击键（技能/击退）后，
                               # 此时间内不再发攻击/击退/转向键——游戏会忽略攻击动画
                               # 期间的按键，多发只会被吞。0=关闭。
                               # 校准：太小仍被吞，太大浪费输出（默认 0.6，怪压过来降到
                               # 0.5，经常"挥弓不射箭"加到 0.7）
        # ---- 近身击退（长手专用）----
        # 怪贴到 range 以内时改放"击退技能"把它推开，推开后继续站定输出。
        # 与"贴脸后撤"的区别：不移动（不落崖、不走进怪群、不用走回目标）、
        # 后撤期间不输出而击退本身带伤害。
        # key 为空 = 不启用；只替代"本帧这一次技能按键"，冷却期内照常走普通技能选择，
        # 所以击退被游戏吞掉或对怪免疫时普通攻击不会被挡住。
        "knockback_skill": {
            "name": "击退技能",
            "key": "",        # 击退技能按键，留空=不启用（按键不要与技能/buff/加血/加蓝/拾取键重复）
            "cooldown": 0.3,  # 两次击退之间的最小间隔（秒）
            "range": 240,     # 触发距离（px，双方中心x距离）：实测参考帧 224/232px
        },
        # ---- 残影（误识别/尸体）判定 ----
        # 怪物死后 YOLO 可能继续把尸体或像怪的背景物体检测成 monster，一直锁着它
        # 就是"对着空气打"。对同一目标出手这么多次（且超过 3 秒）还在 → 判定为残影，
        # 把该位置记入忽略表（8 秒起，同一位置再次判定翻倍，最长 300 秒）。
        # 目标在这期间动过（活怪）→ 要 2 倍次数/时间才算"打不到"。
        "ghost_casts": 6,         # 0 = 关闭残影判定
        # ---- 转向模式 ----
        # tap = 点按 120~180ms + 位移自检 + 失败重试（现状）；
        # hold = 按住方向键走出 TURN_HOLD_CONFIRM_PX 再松键出手：一次确定性输入
        # 代替"点按→自检→重试"循环，消除小碎步。turn_hold_* 参数见 context.py。
        "turn_mode": "tap",       # "tap" / "hold"
        # ---- 站桩模式 ----
        "stand_mode": False,     # 站桩模式：不对角色做任何移动（不移动/不转向/不后撤/不探索），
                                 # 只打朝向正前方射程内的怪；攻击逻辑与普通模式完全一致
        "stand_facing": "right", # 站桩模式下的固定朝向（right/left），用于判断"正前方"
        "stand_default_attack": True,  # 站桩模式：没有有效目标(模型漏检)时也按技能键盲打
                                       # （两个技能交替，各自受冷却约束；不移动不转向）
        "stand_skill2_only": True,     # 站桩模式：攻击时只放技能2(群攻/AOE)
                                       # 技能2 冷却中就等下一帧，不退回技能1
                                       # （普通模式的 AOE 连发会用技能1兜底，站桩按需求去掉）
        # ---- 站桩定时微动（反"定点一动不动"的机器特征）----
        # 每过 interval 秒：朝"背离站桩朝向"的方向极短点一下，再朝站桩朝向
        # 极短点一下（两次时长相同、方向相反 ≈ 原地晃一下）。
        # 不做位置闭环，所以不配"移动多少像素"；tap_ms 控制"点多久"。
        "stand_micro_move_enabled": True,
        "stand_micro_move_interval": 300.0,  # 微动间隔（秒），0=关闭
        "stand_micro_move_tap_ms": 40,       # 单次点按时长（毫秒）
        "skills": [
            {"name": "技能1", "key": "1", "cooldown": 1.0},
            {"name": "技能2", "key": "2", "cooldown": 3.0},
            {"name": "技能3", "key": "3", "cooldown": 8.0},
        ],
        # ---- 自动加 buff ----
        # [{name, key, cooldown}, ...]；cooldown = 重新释放间隔(秒)，
        # 建议填略小于 buff 游戏内持续时间（如 180s 的 buff 填 170）。
        # 开启自动打怪后每帧尝试按，冷却到就放（首帧立即全部释放一次）。
        "buff_skills": [],
        # ---- 拾取 ----
        "pickup_enabled": True,   # 是否启用自动拾取
        "pickup_key": "z",        # 拾取键
        "pickup_interval": 0.333, # 拾取间隔（秒），默认每秒3次
        # ---- 自动截图 ----
        "screenshot_interval": 5,  # 自动截图间隔（秒）
        # ---- 像素参数基准高度 ----
        # 代码里按像素写的阈值（活动范围的分界线容差/边缘宽度等）都按这个
        # 画面高度填写，运行时乘以 当前帧高/此值。客户端只是"看到更多地图"、
        # 人物大小不变时填 0 关闭缩放。
        "pixel_base_height": 768,
        # ---- 活动范围（只在地图左半边 / 右半边打怪）----
        # 镜头跟着角色走，只看画面不知道自己在地图哪里，所以用小地图上的黄点：
        # 黄点离分界线多远 × minimap_scale = 画面里分界线离自己多远。
        "range_limit": "none",   # none / left / right
        "range_split": 0.5,      # 分界线在小地图宽度的位置（0.5 = 正中间）
        "minimap_region": None,  # 小地图的地图部分（百分比 [x, y, w, h]，界面「框选小地图」写入）
        "minimap_scale": 16,     # 小地图 1px 对应画面多少 px
        # ---- 自动换频道 ----
        "channel_enabled": False,      # 这么久没放过技能（这层没怪了）就换一个频道
        "channel_idle_min": 3,         # 多少分钟没攻击就换
        "channel_hold_key": "right",   # 换完按住这个键走一段
        "channel_hold_seconds": 5,
        # ---- 定时结束游戏 ----
        # 点「开始」后，下一次到这个钟点就不打了：Esc → 点「退出游戏」→ 有确认框就回车，
        # 回到登录界面后停止并解除窗口锁定。
        "quit_enabled": False,
        "quit_time": "06:00",          # 几点结束（今天过了就是明天）
        # ---- 经验记录 ----
        "exp_enabled": True,           # 读状态栏的 EXP，每分钟一行写到 logs/exp_log/日期.csv
        "exp_read_seconds": 5,         # 多少秒读一次
        "exp_log_minutes": 10,         # 多少分钟在日志里报一次（0 = 不报）
        # ---- 测谎弹窗检测（反外挂）----
        # 游戏弹出"测谎探测仪"时（3 秒后开始"鼠标跟随图形"小游戏，失败会被处罚），
        # 检测到就【立即停机 + 报警】，把真实鼠标交给玩家。
        # 实现见 src/perception/lie_detector.py（模板匹配），模板图片默认
        # assets/templates/lie_detector.png，阈值 0.78 有充足的抗误报余量。
        "lie_detect_enabled": True,
        "lie_detect_template": "assets/templates/lie_detector.png",
        "lie_detect_threshold": 0.78,  # 匹配得分阈值（实测正样本 0.96 / 普通帧最高 0.39）
        # ---- 测谎小游戏自动作答（mxd-lie-logic-code 方案）----
        # 命中弹窗停机后：LieRecorder 录画面+光标（60fps，影子跟踪结果写同目录），
        # LieShadow 用形状匹配跟踪目标并接管鼠标（点一下左键才交还）；
        # OCR 认出「成功通过」对话框后按 Esc 关掉并自动恢复打怪，没通过一直停着。
        # 实现见 src/perception/lie_shadow.py（ShapeFinder）/ lie_detector_watch.py / lie_recorder.py。
        "lie_shadow_enabled": True,
        "lie_shadow_model": "models/lie_target/best.onnx",
        "lie_shadow_conf": 0.1,
        "lie_shadow_takeover": True,   # False=只算位置写日志，不动鼠标
        "lie_shadow_end_frames": 15,   # 画板连续 N 帧不见 → 小游戏结束（15 帧≈1 秒）
        "lie_shadow_max_seconds": 60,  # 录制+跟踪最长时长（秒）
        "lie_auto_resume": True,       # 测谎通过后按 Esc 关对话框并自动恢复打怪
        "lie_record_dir": "logs/lie_detected",  # 录制输出目录（相对路径基于程序目录）
        # ---- 测谎影子跟踪 / 接管（mxd-lie-target 方案）----
        # 画板出现后：找金色画板 → 藏准星 → 两帧对齐相减（+ 形状先验）→ YOLO26n 出框
        # → 运动预测跟踪。图形淡出后靠"跟花纹动得不一样"和倒计时记下的剪影继续跟。
        # 你没动鼠标时程序把鼠标移过去；一动就交还给你，这次测谎只录制。
        # 实现与模型说明见 mxd-lie-target/docs/LIE_TARGET.md。
        "lie_shadow_enabled": True,
        "lie_shadow_model": "models/lie_target/best.onnx",
        "lie_shadow_conf": 0.1,
        "lie_shadow_takeover": True,   # False=只算位置写日志，不动鼠标
        "lie_shadow_end_frames": 15,   # 画板连续 N 帧不见 → 判定小游戏结束（15 帧≈1 秒）
        "lie_shadow_max_seconds": 60,  # 影子跟踪最长时长（秒），超时收手
    }


class Config:
    """配置实体类。

    封装双层配置（默认 + 用户），提供属性访问和坐标缩放功能。

    用法:
        cfg = Config.load()  # 从 config/user.yaml 加载
        cfg = Config(overrides={"hp_region": [100, 200, 50, 10]})  # 覆盖某项

    Attributes:
        所有 YAML 配置字段都作为属性直接访问，如 cfg.hp_threshold, cfg.fps 等。
        属性名与 YAML 字段名一致（下划线命名）。
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """构造配置对象。

        Args:
            overrides: 覆盖项字典，键值对会覆盖默认配置
        """
        # 1. 加载默认配置
        data = _defaults()

        # 2. 用用户配置覆盖（YAML 优先，JSON 回退）
        user = _load_user_config()
        data.update(user)

        # 3. 用运行时覆盖项覆盖
        if overrides:
            data.update(overrides)

        self._data = data

    # =========================================================================
    # 工厂方法
    # =========================================================================

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """从指定路径加载配置。

        Args:
            path: 配置文件路径（.yaml 或 .json），None 时自动查找默认路径

        Returns:
            Config 实例
        """
        if path is None:
            path = config_path()

        data = _defaults()
        if path.endswith((".yaml", ".yml")):
            data.update(_load_yaml(path))
        elif path.endswith(".json"):
            data.update(_load_json(path))
        else:
            # 尝试 YAML 再尝试 JSON
            data.update(_load_yaml(path))
            if not data:
                data.update(_load_json(path))

        return cls(overrides=None)

    def save(self, path: Optional[str] = None):
        """保存当前配置到 YAML 文件。

        Args:
            path: 保存路径，None 时使用默认路径 config/user.yaml
        """
        if path is None:
            path = DEFAULT_YAML_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 清理 None 值，使 YAML 更干净
        clean_data = {}
        for k, v in self._data.items():
            if v is not None:
                clean_data[k] = v
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(clean_data, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False, width=120)

    def merge(self, updates: Dict[str, Any]):
        """合并配置项（运行时修改）。

        Args:
            updates: 要更新的键值对
        """
        self._data.update(updates)

    # =========================================================================
    # 属性访问：让配置项像属性一样访问
    # 例如: cfg.hp_threshold 等价于 cfg._data["hp_threshold"]
    # =========================================================================

    def __getattr__(self, name: str):
        """属性访问降级到 _data 字典。

        如果正常属性找不到（如 __dict__ 中没有），
        就在 _data 字典中查找。这样可以让配置项像属性一样访问。

        例如: cfg.hp_threshold → cfg._data["hp_threshold"]
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config 没有 '{name}' 字段")

    def __setattr__(self, name: str, value):
        """属性赋值：保存在 _data 中。

        非私有属性（不以下划线开头）直接写入 _data 字典。
        私有属性（如 _data）正常走标准属性赋值。
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # =========================================================================
    # 坐标自适应缩放
    # =========================================================================

    def scale_region(self, region: Optional[List],
                     frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """将配置区域转换为当前帧像素坐标。

        支持两种格式:
          1. 百分比 [x%, y%, w%, h%]  每个值 0.0~1.0，相对帧宽高
          2. 像素 [x, y, w, h]  整数，基于 reference_width/reference_height 缩放

        Args:
            region:       区域 [x, y, w, h]，None 返回 None
            frame_width:  当前帧宽度（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            区域 (x, y, w, h) 像素坐标，无效时返回 None
        """
        if region is None:
            return None
        if len(region) < 4:
            return None

        # 判断格式：浮点或 <= 1.0 → 百分比，否则 → 像素缩放
        is_percent = any(isinstance(v, float) for v in region) or max(region) <= 1.0

        if is_percent:
            # 百分比：直接乘以帧尺寸
            sx_i = int(region[0] * frame_width)
            sy_i = int(region[1] * frame_height)
            sw_i = int(region[2] * frame_width)
            sh_i = int(region[3] * frame_height)
        else:
            # 像素：按参考分辨率缩放
            sx = frame_width / self.reference_width
            sy = frame_height / self.reference_height
            sx_i = int(region[0] * sx)
            sy_i = int(region[1] * sy)
            sw_i = int(region[2] * sx)
            sh_i = int(region[3] * sy)

        # 边界保护
        if sx_i < 0:
            sw_i += sx_i
            sx_i = 0
        if sy_i < 0:
            sh_i += sy_i
            sy_i = 0
        if sx_i + sw_i > frame_width:
            sw_i = frame_width - sx_i
        if sy_i + sh_i > frame_height:
            sh_i = frame_height - sy_i
        if sw_i <= 0 or sh_i <= 0:
            return None
        return (sx_i, sy_i, sw_i, sh_i)

    def scale_offset(self, offset: int, frame_height: int) -> int:
        """将参考分辨率下的偏移量缩放到当前帧高度。

        公式: scaled_offset = offset * (frame_height / reference_height)

        Args:
            offset:       参考分辨率下的偏移（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            缩放后的偏移量
        """
        return int(offset * frame_height / self.reference_height)


def load_config(path: Optional[str] = None) -> Config:
    """快捷函数：加载配置。

    Args:
        path: 配置文件路径（.yaml / .json），None 自动查找默认路径

    Returns:
        Config 实例
    """
    return Config.load(path)


def save_config(config: Config, path: str):
    """保存配置到指定路径。

    Args:
        config: Config 实例
        path: 保存路径
    """
    config.save(path)


def save_user_config(config: Config):
    """保存配置到默认用户配置文件 (config/user.yaml)。

    Args:
        config: Config 实例
    """
    config.save(DEFAULT_YAML_PATH)