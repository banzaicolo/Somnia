#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AI 医疗床 —— 边缘主控程序骨架 V2（教学演示版，可直接运行）
=============================================================================

【V2 相比 V1 新增了什么？】

V1 的骨架能跑，但它只能"在一切正常时"工作。真实设备开机三个月不重启，
一定会遇到下面这些事：

  1. 串口线松了 / 传感器死机  →  V1 会永远卡死或一直报错
  2. 模型推理某次抛异常       →  那条线程悄悄死了，程序还在"假装正常运行"
  3. 某个环节卡住不动了       →  屏幕上数字不动，护士以为病人没事

所以 V2 补了两块"保命"的东西：

  ┌─────────────────────────────────────────────────────────┐
  │  ① 断线自动重连（自己治自己）                            │
  │     传感器掉了 → 自己重试连接，间隔越来越长，直到连上     │
  ├─────────────────────────────────────────────────────────┤
  │  ② 看门狗（Watchdog，别人盯着你，你死了它把你拉起来）     │
  │     每条线程每隔一会"喂一次狗"（报平安）                  │
  │     看门狗发现谁超时没报平安 → 判定它出事了 → 重启它      │
  └─────────────────────────────────────────────────────────┘

【整体结构：还是那条流水线，只是多了条"巡逻的狗"】

  [床垫传感器] ──> 采集线程 ──> [原始数据队列] ──> 推理线程 ──> [结果队列] ──> 主线程(显示/报警)
                     │                                │                          │
                     └──────── 定期"喂狗" ────────────┘                          │
                                    │                                            │
                                    v                                            │
                              [看门狗线程] ── 发现异常 ──> [重启请求队列] ─────────>│
                                                                                 │
                                                              主线程看到请求就去重启它

【⚠️ 必须先看懂的几句话】

1. 本文件里的"传感器"和"AI 模型"都是【假的】，用随机数模拟。
   等你拿到真硬件，只改标着 ★★★替换点★★★ 的 4 个函数，其他一行不用动。

2. 文件顶部有一组 DEMO_* 开关，会【故意制造故障】让你亲眼看到自愈过程。
   等你把这套机制看懂了，把 DEMO = True 改成 False，就是干净的生产版行为。

3. 【这一条最重要，很多人不知道】：
   Python 没有办法从外面"杀掉"一条正在运行的线程。
   所以看门狗要分两种情况处理：
     - 线程【已经死了】（抛异常退出了）→ 看门狗能安全重启它 ✅
     - 线程【还活着但卡住了】（死循环/阻塞在某个调用上）→ Python 杀不掉它 ❌
       正确做法是让【整个程序】退出，交给外面的守护程序（systemd）
       把整个进程重新拉起来。这在文件里能看到演示。

【怎么运行】

Mac 打开"终端"（启动台里搜"终端"），粘进去回车：

    python3 src/main_controller.py

想提前结束：按 Ctrl + C
=============================================================================
"""


# ============================================================================
# 第一部分：把要用的"工具箱"拿进来
# ============================================================================
# 全部是 Python 自带的库，不用额外安装任何东西。

import threading  # 多线程：让几件事能"同时干"
import queue      # 队列：线程之间安全传数据的"传送带"
import time       # 时间：计时、故意等待
import random     # 随机：造假传感器数据用
import sys        # 系统：往屏幕输出、控制退出码


# ============================================================================
# 第二部分：全局参数（所有线程都能看到的配置）
# ============================================================================
# 数字集中放在最上面，以后想改采样率、改超时，只改这里，不用翻代码。

# ---- 采集相关 ----
SENSOR_POINTS = 96      # 床垫上有多少个压力感应点（12×8 阵列）
SAMPLE_HZ = 20          # 每秒采集多少次（20 表示每 50 毫秒一帧）
INFER_BATCH = 4         # 攒够几帧再送进模型（攒一批算一次，效率高）

# ---- 队列容量（保护机制，防止内存被撑爆）----
RAW_QUEUE_MAX = 100     # 原始数据队列最多存多少帧
RESULT_QUEUE_MAX = 50   # 结果队列最多存多少条

# ---- 报警 ----
PRESSURE_ALERT = 800    # 压力报警阈值（假数值，真实产品要临床数据定）

# ---- 🆕 断线重连相关 ----
SENSOR_DROP_PROB = 0.01     # 每帧有多大概率"假装断线"（演示用，1%）
RECONNECT_BASE_DELAY = 0.5  # 第一次重连等 0.5 秒
RECONNECT_MAX_DELAY = 8.0   # 重连等待时间最多 8 秒（不能无限涨上去）
RECONNECT_SUCCESS_RATE = 0.6  # 每次重连尝试的成功概率（模拟现实里可能连不上）

# ---- 🆕 看门狗相关 ----
WATCHDOG_CHECK_INTERVAL = 1.0  # 看门狗每隔多久巡查一次（秒）
WATCHDOG_TIMEOUT = 3.0         # 超过多少秒没"喂狗"就算出事了（秒）
# ⚠️ 这个 3 秒怎么定？要大于"正常工作时两次喂狗的最大间隔"。
#    我们采集线程最慢也就几十毫秒喂一次，3 秒绰绰有余，不会误报。
#    真实产品里要按你最慢的那个环节来定，别拍脑袋。

# ---- 🆕 演示开关：故意制造故障，让你看到自愈过程 ----
# 把这行改成 False，程序就变成"永远正常运行"的干净版本。
DEMO = True

DEMO_CRASH_AT_BATCH = 5    # 推理线程算到第 5 批时，故意崩溃一次
                           # （演示：线程死了 → 看门狗发现 → 自动重启）
DEMO_FREEZE_AT_BATCH = 12  # 推理线程算到第 12 批时，故意"卡死不动"
                           # （演示：线程假死 → Python 杀不掉 → 只能重启整个程序）
                           # 想跳过这个演示，把它改成 99999


# ============================================================================
# 第三部分：四个"造假"函数 —— 拿到真硬件后就换掉这四个
# ============================================================================


# ---- 🆕 先定义一个"异常"类型：传感器断线 ----
# 异常（Exception）就是"出事了"的信号。
# 自己定义一个异常类型，好处是：出问题的时候能一眼看出是"断线"还是别的毛病。
class SensorDisconnected(Exception):
    """传感器断线（串口读不到数据、设备掉线等）"""
    pass


def open_sensor_port():
    """
    ★★★替换点 1 of 4★★★  —— 打开传感器连接（现在是假的）

    真实产品里大概是：
        import serial
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        return ser

    现在假装"打开串口"，并且有概率失败（模拟现实里重连不一定一次成功）。

    返回值：一个"连接对象"（这里随便返回个字典意思一下）
    失败时：抛出 SensorDisconnected 异常
    """
    # random() 返回 0~1 之间的小数。小于成功率就算"连上了"
    if random.random() < RECONNECT_SUCCESS_RATE:
        return {"端口": "假串口", "打开时间": time.time()}
    else:
        raise SensorDisconnected("打开串口失败（模拟：设备还没准备好）")


def read_one_frame_from_sensor():
    """
    ★★★替换点 2 of 4★★★  —— 读一帧传感器数据（现在是假的）

    真实产品里：
        data = serial_port.read(192)   # 从串口读 192 个字节
        frame = 解析(data)              # 按协议拆成 96 个数字
        return frame

    现在造假：返回 96 个 0~1000 的随机数，偶尔造一个特别大的好触发报警。
    另外按 SENSOR_DROP_PROB 的概率【假装断线】，用来演示自动重连。

    返回值：长度 96 的列表
    出问题：抛 SensorDisconnected
    """
    # 🆕 模拟断线：扔个骰子，1% 的概率"掉线"
    if random.random() < SENSOR_DROP_PROB:
        raise SensorDisconnected("串口读超时（模拟：线缆松了 / 设备掉电）")

    # 列表推导式：一口气造 96 个随机数
    frame = [random.randint(0, 1000) for _ in range(SENSOR_POINTS)]

    # 模拟"人翻身压到某个点"
    if random.random() < 0.05:
        frame[random.randrange(SENSOR_POINTS)] = random.randint(850, 1000)

    return frame


def preprocess(frames):
    """
    ★★★替换点 3 of 4★★★  —— 数据预处理（现在是假的）

    真实产品里这里要做一堆正经事，而且【很重要】：
      - 去噪（滤掉抖动干扰）
      - 温度补偿（夏天冬天读数不一样，要修正）
      - 零点校准（人下床时归零，修正长期漂移）
      - 归一化（缩放到 0~1，模型才好吃进去）

    现在只算个平均值意思一下。
    """
    all_values = []
    for frame in frames:
        all_values.extend(frame)   # extend：把一个列表"接"到另一个后面
    return sum(all_values) / len(all_values)


def run_model_on_npu(data):
    """
    ★★★替换点 4 of 4★★★  —— 跑 AI 模型（现在是假的）

    真实产品里调 RKNN 的 SDK（瑞芯微给的接口）：
        rknn_outputs = rknn_runtime.inference(...)

    ⚠️ 关键点：这步【不会拖累 Python】。真正算东西的是 NPU 硬件，
    Python 只是"下单"然后"等结果"，等待时会把那把锁（GIL）放掉。

    现在用 sleep(0.03) 假装"在等 NPU 算"。
    """
    time.sleep(0.03)  # 假装在等 NPU，真实产品里删掉

    if data < 300:
        posture = "离床"
    elif data < 500:
        posture = "仰卧"
    elif data < 700:
        posture = "侧卧"
    else:
        posture = "俯卧"

    return {
        "睡姿": posture,
        "平均压力": round(data, 1),
        "置信度": round(random.uniform(0.75, 0.99), 2),
    }


# ============================================================================
# 🆕 第四部分：心跳工具 —— "喂狗"的两个小函数
# ============================================================================
# 【什么是心跳 / 喂狗？】
#
# 每条干活的线程，每隔一小会儿就来"报个到"，把当前时间记进一个小本子。
# 这个动作就叫"喂狗"（feed the dog）。
#
# 为什么要叫"喂狗"？因为现实里的硬件看门狗是这样的：
#   你养了条狗，它有个倒计时。你必须定期喂它，它就不叫。
#   你要是昏过去了没喂，倒计时一到，它就狂吠 —— 然后把你咬醒（重启系统）。
#
# 软件里是同一个道理：线程定期报平安，看门狗盯着时间戳；
# 谁超时没报，就说明谁出事了。


def beat(heartbeats, hb_lock, name):
    """
    喂狗：把"我还活着"这件事记下来。

    参数：
      heartbeats —— 小本子（字典），格式 {"采集线程": 1234567890.1, ...}
                    值是"最后一次喂狗的时间"
      hb_lock    —— 一把锁。因为小本子是所有线程共用的，同时写会打架，
                    所以写之前先锁上，写完解锁。with 语句会自动解锁，
                    不用你记得写 unlock()，这是最推荐的写法。
      name       —— 是谁在喂狗
    """
    with hb_lock:
        heartbeats[name] = time.time()   # time.time() 是"现在"的时间戳（秒）


def last_beat(heartbeats, hb_lock, name):
    """
    查一下某人最后一次喂狗是什么时候。

    返回值：时间戳（秒）。如果从没喂过，返回 None。
    """
    with hb_lock:
        return heartbeats.get(name)      # get 取不到就返回 None，不会报错


# ============================================================================
# 🆕 第五部分：断线自动重连
# ============================================================================


def reconnect_sensor(stop_event, heartbeats, hb_lock, name):
    """
    传感器断线后的重连逻辑。

    【为什么不能直接 while True 一直重试？】
    因为如果设备是真死了（比如断电了），你每微秒重试一次，
    会把 CPU 跑满、日志刷爆，还会把别的活拖慢。这叫"忙等"，是坏毛病。

    【正确做法叫"指数退避"（Exponential Backoff）】
    第 1 次失败 → 等 0.5 秒
    第 2 次失败 → 等 1 秒
    第 3 次失败 → 等 2 秒
    第 4 次失败 → 等 4 秒
    第 5 次及以后 → 最多等 8 秒（封顶，不能无限涨）
    这样既不会放弃，也不会折腾机器。所有正经的联网程序都这么干。

    返回值：True = 连上了；False = 程序要收工了，别再连了
    """
    attempt = 0   # 第几次尝试

    while not stop_event.is_set():   # 只要还没收到"收工"信号，就一直试
        attempt += 1

        # 算这次要等多久：0.5 × 2^(n-1)，但最多 8 秒
        delay = min(RECONNECT_BASE_DELAY * (2 ** (attempt - 1)), RECONNECT_MAX_DELAY)
        print(f"[{name}] 第 {attempt} 次重连，{delay} 秒后尝试...")

        # ---- 分段等待，不要一口气睡死 ----
        # 为什么不用 time.sleep(delay) 一行搞定？
        # 因为睡死期间你就收不到"收工"信号了，程序会卡在这里关不掉。
        # 所以切成 0.1 秒一小段一小段地睡，每小段都看一眼停止旗。
        waited = 0.0
        while waited < delay and not stop_event.is_set():
            time.sleep(0.1)
            waited += 0.1
            # ⚠️ 关键：等待期间也要喂狗！
            # 否则看门狗会以为"这条线程死了"，把你重启掉 —— 冤枉。
            beat(heartbeats, hb_lock, name)

        if stop_event.is_set():   # 等待期间收到了收工信号
            return False

        # ---- 真的去连 ----
        try:
            open_sensor_port()
            return True           # 连上了
        except SensorDisconnected:
            continue              # 没连上，下一轮再试（等待时间会翻倍）


# ============================================================================
# 第六部分：四条线程各自干的事（本文件的核心）
# ============================================================================


def collect_worker(stop_event, raw_queue, heartbeats, hb_lock):
    """
    【采集线程】一刻不停地从传感器读数据，读完塞进队列。
                遇到断线就自己重连。

    ⚠️ 注意本函数的结构，这是"能自愈的线程"的标准写法：
       try:     正常干活
       except:  出事了 → 记日志 → 直接 return（不放哨兵，等看门狗重启我）
       正常退出：走到最后，放哨兵通知下游收工

       为什么要区分这两种退出？
       - 崩溃退出：我马上要被重启了，不能告诉下游"结束了"，否则整条线都停了
       - 正常收工：程序要关了，必须告诉下游，否则下游会永远等下去
    """
    name = "采集线程"
    dropped = 0

    print(f"[{name}] 开工，采样率 {SAMPLE_HZ} Hz")

    try:
        while not stop_event.is_set():

            # ---- 0. 每次干活前先喂狗 ----
            beat(heartbeats, hb_lock, name)

            # ---- 1. 读一帧数据（这里可能抛"断线"异常）----
            try:
                frame = read_one_frame_from_sensor()
            except SensorDisconnected as e:
                # 🆕 断线了，进入重连流程
                print(f"[{name}] ⚠️  传感器断线：{e}")
                ok = reconnect_sensor(stop_event, heartbeats, hb_lock, name)
                if ok:
                    print(f"[{name}] ✅ 已重新连上，继续采集")
                    continue    # 继续下一帧
                else:
                    # 重连途中被叫停 = 程序要收工了。
                    # ⚠️ 这里必须也放一个哨兵！
                    # 否则下游推理线程收不到通知，会一直卡在 get() 等下去，
                    # 最后只能靠 daemon 被强收，打印"没能正常退出"的警告。
                    raw_queue.put(None)
                    return

            # ---- 2. 往队列里放 ----
            try:
                raw_queue.put(frame, timeout=0.5)   # 一定要给 timeout！
            except queue.Full:
                # 队列满了说明下游跟不上。丢掉这一帧，别傻等。
                # 记住：丢几帧数据没事，程序卡死是大事。
                dropped += 1
                if dropped % 50 == 0:
                    print(f"[{name}] 警告：处理跟不上，已丢弃 {dropped} 帧")

            # ---- 3. 歇一下，控制采样率 ----
            time.sleep(1.0 / SAMPLE_HZ)

    except Exception as e:
        # 出了没预料到的毛病（比如内存不够、协议解析崩了）
        # 这里不 raise，只是记下来然后退出，好让看门狗能检测到"我死了"
        print(f"[{name}] ❌ 意外崩溃：{e!r}（等看门狗重启）")
        return    # 【关键】直接返回，不放哨兵

    # ---- 走到这里 = 正常收工（收到了停止旗）----
    # 放一个哨兵 None 通知下游："我这边结束了，你也收工吧"
    raw_queue.put(None)
    print(f"[{name}] 收工（共丢弃 {dropped} 帧）")


def next_batch_no(state, state_lock):
    """
    🆕 取一个"批次号"，每调一次加一。

    ⚠️ 为什么不能直接用一个普通变量 n_batch += 1？
    因为这个计数要【跨线程重启】保持连续。如果用线程内部的局部变量，
    线程一重启它就从 0 重新数了 —— 演示里就会永远卡在同一批反复崩溃。
    真实产品里也一样：重启后你得知道"我已经跑到第几批了"，
    比如做"每 1000 批做一次自检"这种逻辑，计数归零就全乱了。

    所以把计数放在一个【共享的字典】里，并用锁保护它。
    """
    with state_lock:
        state["批次"] += 1
        return state["批次"]


def infer_worker(raw_queue, result_queue, heartbeats, hb_lock, state, state_lock):
    """
    【推理线程】从队列取数据，攒够一批送进模型，结果塞进结果队列。
    """
    name = "推理线程"
    print(f"[{name}] 开工，攒够 {INFER_BATCH} 帧推理一次")

    batch = []        # 攒数据的篮子（线程内部变量，重启就清空，没关系）

    try:
        while True:

            # ---- 1. 取一帧 ----
            # 这里用 get(timeout=0.5) 而不是 get()：
            # 没数据时也能每 0.5 秒醒一次去喂狗，证明"我还活着，只是没活干"。
            # 如果用不带 timeout 的 get()，会一直阻塞，就没机会喂狗了 ——
            # 看门狗会误以为你死了。这是很常见的坑。
            try:
                frame = raw_queue.get(timeout=0.5)
            except queue.Empty:
                beat(heartbeats, hb_lock, name)   # 没活干也要报平安
                continue

            # ---- 2. 是不是"收工暗号" ----
            if frame is None:
                raw_queue.task_done()
                result_queue.put(None)   # 把暗号传给下游
                break

            # ---- 3. 放进篮子 ----
            batch.append(frame)
            raw_queue.task_done()

            # ---- 4. 攒够了就送进模型 ----
            if len(batch) >= INFER_BATCH:
                # 批次号从【共享计数器】取，线程重启后不会归零
                n_batch = next_batch_no(state, state_lock)

                # 🆕 演示 1：故意崩一次，让你看"线程死了→被重启"
                if DEMO and n_batch == DEMO_CRASH_AT_BATCH:
                    raise RuntimeError("模拟：RKNN 推理返回异常码 -5")

                # 🆕 演示 2：故意卡死，让你看"卡住的线程 Python 杀不掉"
                if DEMO and n_batch == DEMO_FREEZE_AT_BATCH:
                    print(f"[{name}] 💀 模拟：NPU 驱动卡死，本线程将不再喂狗也不再工作")
                    time.sleep(300)   # 假装卡死 5 分钟，看门狗会在此期间发现

                data = preprocess(batch)
                result = run_model_on_npu(data)
                result_queue.put(result)
                batch = []

                beat(heartbeats, hb_lock, name)   # 干完活喂狗

    except Exception as e:
        print(f"[{name}] ❌ 意外崩溃：{e!r}（等看门狗重启）")
        return    # 不放哨兵，等被重启

    print(f"[{name}] 收工")


def watchdog_worker(stop_event, threads, heartbeats, hb_lock, restart_queue):
    """
    🆕【看门狗线程】什么都不干，就盯着别人有没有按时喂狗。

    这是整个自愈机制的核心。它每秒巡查一次，检查每条线程的"最后喂狗时间"。

    ⚠️ 它要区分两种情况，处理方式完全不同：

      情况 A：线程【已经死了】（thread.is_alive() 返回 False）
              → 说明它抛异常退出了，内存干净，可以安全重建一条新的 ✅

      情况 B：线程【还活着但超时没喂狗】
              → 说明它卡在某个地方不动了（死循环、阻塞在某个调用上）
              → ❌ Python 没有任何办法从外部强制杀死一条线程！
                 （这是 Python 的设计限制，不是你代码写错了）
              → 唯一正确的做法：让【整个程序】退出，
                 由外面的守护程序（Linux 上叫 systemd）把整个进程重新拉起来。
                 所以要让程序以【非 0 的退出码】结束，systemd 看到非 0 才重启。

    参数：
      threads       —— 一个字典 {"采集线程": 线程对象, "推理线程": 线程对象}
                       传字典而不是传单个对象，是因为线程被重启后会换成新对象，
                       字典是可变的，看门狗每次读都能拿到最新的那个。
      restart_queue —— 发现问题时，把"请求"放进这个队列，由主线程去执行重启
                       （为什么不自己重启？—— 因为重启后要更新 threads 字典，
                         这类"管理动作"集中在主线程做，逻辑更清楚、不容易乱）
    """
    name = "看门狗"
    print(f"[{name}] 开工，每 {WATCHDOG_CHECK_INTERVAL} 秒巡查一次，"
          f"超时 {WATCHDOG_TIMEOUT} 秒判定出事")

    while not stop_event.is_set():
        # 巡查间隔。同样分段睡，保证能及时收到收工信号
        waited = 0.0
        while waited < WATCHDOG_CHECK_INTERVAL and not stop_event.is_set():
            time.sleep(0.2)
            waited += 0.2

        if stop_event.is_set():
            break

        now = time.time()

        # list() 包一层：遍历字典的同时主线程可能在改它，
        # 直接遍历有极小概率出问题，复制一份快照更稳。
        for tname, thread in list(threads.items()):
            lb = last_beat(heartbeats, hb_lock, tname)

            if lb is None:
                continue   # 还没开始喂过（刚启动），不判罚

            silence = now - lb   # 沉默了多少秒

            if silence > WATCHDOG_TIMEOUT:
                if not thread.is_alive():
                    # ---- 情况 A：已经死了，可以安全重启 ----
                    print(f"[{name}] 🚨 发现「{tname}」已停止（{silence:.1f} 秒没喂狗），请求重启")
                    restart_queue.put(("THREAD", tname))
                else:
                    # ---- 情况 B：还活着但卡住了，Python 杀不掉 ----
                    print(f"[{name}] 🚨 发现「{tname}」还活着，但已 {silence:.1f} 秒没喂狗 —— 卡死了")
                    print(f"[{name}] ⚠️  Python 无法强制杀死线程，只能请求【整个程序重启】")
                    restart_queue.put(("PROCESS", tname))
                    return   # 看门狗自己收工，剩下的交给主线程

    print(f"[{name}] 收工")


def main_ui_loop(result_queue, stop_event, restart_queue,
                 threads, builders, heartbeats, hb_lock):
    """
    【主线程】三件事：
      1. 看有没有"重启请求"（看门狗发过来的）
      2. 取推理结果显示到屏幕
      3. 判断要不要报警

    为什么界面必须放主线程？
    因为 Qt、Tkinter 这些界面库都要求只能在主线程里操作界面，放别处会崩。

    返回值：0 = 正常结束；1 = 需要整个程序重启（交给 systemd）
    """
    name = "主线程"
    print(f"[{name}] 开工，开始显示结果\n")

    frame_count = 0

    # 注意这里用 stop_event 控制循环，而不是"收到哨兵才退出"。
    # 为什么？因为如果某条线程崩了，它可能来不及放哨兵，
    # 主线程要是一心等哨兵，就永远等不到了。靠停止旗更可靠。
    while not stop_event.is_set():

        # ---- 第 1 件事：处理重启请求 ----
        # get_nowait() = "有就拿，没有别等，立刻报错"。这里要的就是"别等"。
        try:
            kind, who = restart_queue.get_nowait()
        except queue.Empty:
            kind, who = None, None

        if kind == "THREAD":
            restart_thread(who, threads, builders, heartbeats, hb_lock)
        elif kind == "PROCESS":
            # 有线程卡死了，Python 杀不掉，只能让整个程序退出
            print(f"\n[{name}] ❌ 「{who}」卡死且无法单独重启，程序将退出。")
            print(f"[{name}]    真实产品中，systemd 看到非 0 退出码会自动把整个进程拉起来。\n")
            return 1   # 非 0 退出码 = "我是非正常退出的，请重启我"

        # ---- 第 2 件事：取一条推理结果 ----
        try:
            result = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue    # 没结果，回去再看看有没有重启请求

        if result is None:      # 收到哨兵
            result_queue.task_done()
            break

        result_queue.task_done()
        frame_count += 1

        # ---- 显示 + 报警 ----
        print(f"  第 {frame_count:>4} 批 | "
              f"睡姿：{result['睡姿']:<4} | "
              f"平均压力：{result['平均压力']:>6} | "
              f"置信度：{result['置信度']}")

        # ⚠️ 真实产品里报警逻辑千万别这么草率：
        #    漏报（真出事没报）的代价 >> 误报（没事乱叫）
        #    阈值要往"宁可多叫几次"那一边调。
        if result["平均压力"] > PRESSURE_ALERT:
            print(f"          >>> 报警：局部压力过高（{result['平均压力']}），"
                  f"压疮风险，请协助翻身 <<<")

        sys.stdout.flush()   # 立刻刷到屏幕，不然看着像卡住了

    print(f"\n[{name}] 收工，共处理 {frame_count} 批")
    return 0


def restart_thread(tname, threads, builders, heartbeats, hb_lock):
    """
    🆕 重启一条线程。

    ⚠️ 最容易踩的坑在这里：
       同一个 Thread 对象【只能 start() 一次】，第二次 start() 会报错
       "threads can only be started once"。

       所以重启不是"把老线程再启动一次"，而是【造一条全新的线程】。
       这就是为什么要有一个 builders（建造者）字典：
       里面存的是"造线程的方法"，每次重启就调它造一条新的。

    参数：
      tname     —— 要重启哪条（"采集线程" / "推理线程"）
      threads   —— 线程字典，重启后要把新线程对象写回去
      builders  —— {"采集线程": 造采集线程的函数, "推理线程": 造推理线程的函数}
    """
    name = "主线程"

    old = threads.get(tname)
    if old is not None and old.is_alive():
        # 理论上不该发生（看门狗只在线程死了才请求重启），
        # 但多一道保险没坏处：避免不小心开出两条同名线程，数据就乱了。
        print(f"[{name}] 「{tname}」还在跑，不重复重启")
        return

    new_thread = builders[tname]()      # 造一条全新的线程
    threads[tname] = new_thread         # 更新字典，让看门狗盯新的这条

    # ⚠️ 必须重置心跳！
    # 否则新线程还没来得及喂狗，看门狗一看"时间戳还是老的"，
    # 会立刻又判定它死了，陷入"一直重启"的死循环。
    beat(heartbeats, hb_lock, tname)

    new_thread.start()
    print(f"[{name}] ✅ 「{tname}」已重启")


# ============================================================================
# 第七部分：程序入口 —— 把上面几块拼起来
# ============================================================================


def main():
    """
    主函数：造队列 → 造线程 → 启动 → 监督 → 收尾
    """
    print("=" * 66)
    print(" AI 医疗床 —— 边缘主控程序骨架 V2（带断线重连 + 看门狗）")
    print(" 按 Ctrl + C 结束")
    if DEMO:
        print(" ⚠️ 演示模式：会故意制造故障，让你看到自愈过程")
        print(f"    · 第 {DEMO_CRASH_AT_BATCH} 批：推理线程崩溃 → 看门狗重启它")
        print(f"    · 第 {DEMO_FREEZE_AT_BATCH} 批：推理线程卡死 → 只能重启整个程序")
        print("    · 随机：传感器断线 → 自动重连")
    print("=" * 66)
    print()

    # ---- 1. 停止旗 ----
    stop_event = threading.Event()

    # ---- 2. 三条传送带 ----
    raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAX)        # 采集 → 推理
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)  # 推理 → 主线程
    restart_queue = queue.Queue()                         # 🆕 看门狗 → 主线程

    # ---- 3. 🆕 心跳小本子 + 它的锁 ----
    heartbeats = {}                    # {"采集线程": 时间戳, "推理线程": 时间戳}
    hb_lock = threading.Lock()         # 保护这个小本子的锁

    # ---- 3b. 🆕 共享计数器（跨线程重启保持连续）----
    state = {"批次": 0}
    state_lock = threading.Lock()

    # ---- 4. 🆕 建造者字典：重启时用得着 ----
    # 这里用的是"闭包"：函数里面又定义了函数，里面的函数能记住外面的变量
    # （比如 raw_queue、stop_event），所以造线程时不用再传一堆参数。
    def build_collect():
        return threading.Thread(
            target=collect_worker,
            args=(stop_event, raw_queue, heartbeats, hb_lock),
            daemon=True,
            name="采集线程",
        )

    def build_infer():
        return threading.Thread(
            target=infer_worker,
            args=(raw_queue, result_queue, heartbeats, hb_lock, state, state_lock),
            daemon=True,
            name="推理线程",
        )

    builders = {
        "采集线程": build_collect,
        "推理线程": build_infer,
    }

    # ---- 5. 造线程并启动 ----
    threads = {}
    threads["采集线程"] = build_collect()
    threads["推理线程"] = build_infer()

    # 🆕 看门狗线程（它自己不需要被监督，简化起见）
    t_watchdog = threading.Thread(
        target=watchdog_worker,
        args=(stop_event, threads, heartbeats, hb_lock, restart_queue),
        daemon=True,
        name="看门狗线程",
    )

    threads["采集线程"].start()
    threads["推理线程"].start()
    t_watchdog.start()

    # ---- 6. 主线程开始干活（显示 + 报警 + 处理重启请求）----
    exit_code = 0
    try:
        exit_code = main_ui_loop(
            result_queue, stop_event, restart_queue,
            threads, builders, heartbeats, hb_lock,
        )
    except KeyboardInterrupt:
        print("\n\n[主线程] 收到 Ctrl+C，正在收工...")
    finally:
        # 不管怎么结束，都要举旗通知所有线程收工
        stop_event.set()

        # 等它们收工。timeout 一定要给，否则万一谁卡住了就永远等下去
        for tname, t in threads.items():
            t.join(timeout=2)
            if t.is_alive():
                print(f"[主线程] 警告：{tname} 没能正常退出（可能卡住了）")
        t_watchdog.join(timeout=2)

    print("\n[主线程] 程序已安全退出")
    return exit_code


# ============================================================================
# 第八部分：Python 的固定套路
# ============================================================================
# 只有"直接运行这个文件"时才执行 main()；被 import 时不会自动跑。
if __name__ == "__main__":
    # sys.exit(退出码)：0 表示正常，非 0 表示异常退出。
    # 真实部署时，systemd 配置 Restart=on-failure，
    # 看到非 0 就会自动把整个程序重新拉起来 —— 这就是最后一道防线。
    sys.exit(main())


# ============================================================================
# 【附录】你可能会问的几个问题
# ============================================================================
#
# Q1：为什么用 queue.Queue，不用普通 list？
# A ：因为 Queue 内部【自带锁】，多个线程同时 put/get 不会出错。
#     普通 list 在多线程下可能算错、丢数据。一律用 Queue，别用 list。
#
# Q2：为什么用多线程，不用多进程？
# A ：因为这里的活主要是【等】：等串口数据、等 NPU 算完。
#     等待时 Python 会把那把锁（GIL）放掉，所以多线程是真有效的。
#     什么时候才用多进程？—— 瓶颈是【纯 Python 的密集计算】时。
#
# Q3：为什么不用 asyncio（异步）？
# A ：asyncio 是为【上万个并发连接】设计的（Web 服务器、爬虫）。
#     我们这里并发量是个位数。而且 pyserial、RKNN 的 SDK 都是同步阻塞的，
#     硬塞进 asyncio 会把事件循环卡死，最后还是得绕回线程池。白折腾。
#
# Q4：🆕 看门狗的超时时间（3 秒）该怎么定？
# A ：要【大于】正常工作时两次喂狗的最大间隔，留 3~5 倍余量。
#     我们采集线程最慢几十毫秒喂一次，3 秒绰绰有余。
#     定太短会误报（老重启），定太长会漏报（病人出事了半天才发现）。
#     真实产品里要按你最慢的那个环节实测来定，别拍脑袋。
#
# Q5：🆕 为什么线程"卡住"了不能像"死了"一样重启？
# A ：这是 Python 的硬限制 —— 没有官方 API 能从外部杀死一条正在运行的线程。
#     （网上那些用信号、用 ctypes 强行杀的办法都不安全，会留下烂摊子：
#       锁没释放、资源没回收，反而更糟。）
#     正经做法只有一条：让整个进程退出，由 systemd 重新拉起。
#     所以生产部署一定要配 systemd，配 Restart=on-failure。
#
# Q6：🆕 采集线程重连的时候，为什么不直接 time.sleep(delay)？
# A ：因为睡死期间收不到"收工"信号，程序会卡住关不掉。
#     正确做法是切成一小段一小段地睡，每小段都检查停止旗。
#     这条适用于【所有】等待场景，是个通用习惯。
#
# Q7：🆕 为什么重连的等待时间要"越来越长"（指数退避）？
# A ：防止"忙等"。设备真死了的话，你一秒钟重试一千次只会把 CPU 跑满、
#     日志刷爆，还拖慢别的活。逐渐拉长间隔，既不死心也不折腾机器。
#
# Q8：真实产品里，这个骨架还缺什么？
# A ：至少还缺：
#     - 数据落盘（本地存一份，出事能回溯）
#     - 日志系统（现在用 print 是玩具做法，正式要用 logging 模块，
#                 能自动分文件、分级别、记时间戳）
#     - 资源清理（串口、NPU 句柄要显式关闭，靠 finally）
#     - 硬件看门狗（板子上真正的那个，软件死了它能断电重启）
#     - 掉电保护（突然断电时数据不能丢）
#     这些等骨架跑通了再一层层加，别一上来就想做全套。
#
# ============================================================================
