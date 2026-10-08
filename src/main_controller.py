#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AI medical bed — edge main-controller skeleton V2 (teaching/demo version, runnable)
=============================================================================

What's new in V2 vs V1?

The V1 skeleton ran, but only "when everything is normal". A real device that
runs for three months without a reboot will definitely hit these:

  1. serial cable loose / sensor dead  →  V1 would hang forever or keep erroring
  2. model inference throws once       →  that thread silently dies, while the program keeps "pretending to run"
  3. some step gets stuck              →  the numbers on screen freeze, and the nurse thinks the patient is fine

So V2 adds two "life-saving" pieces:

  ┌─────────────────────────────────────────────────────────┐
  │  1) auto-reconnect on disconnect (heals itself)        │
  │     sensor drops → retries on its own, growing intervals until reconnected │
  ├─────────────────────────────────────────────────────────┤
  │  2) watchdog (someone watching you; if you die, it pulls you back up) │
  │     each thread "feeds the dog" every so often (reports in)          │
  │     the watchdog finds whoever timed out → judges it broken → restarts it │
  └─────────────────────────────────────────────────────────┘

Overall structure: still the same pipeline, plus a "patrolling dog"

  [mattress sensor] ──> collection thread ──> [raw data queue] ──> inference thread ──> [result queue] ──> main thread (display/alarm)
                           │                                │                          │
                           └──────── periodic "feed the dog" ──────────┘              │
                                          │                                            │
                                          v                                            │
                                    [watchdog thread] ── finds trouble ──> [restart request queue] ──────>│
                                                                                       │
                                                        the main thread sees the request and restarts it

A few lines you must read first:

1. The "sensor" and "AI model" in this file are fake, simulated with random
   numbers. Once you have real hardware, only change the 4 functions marked
   ★★★replacement point★★★; not one other line needs touching.

2. At the top of the file there's a group of DEMO_* switches that will
   deliberately inject faults so you can see the self-healing process firsthand.
   Once you understand the mechanism, change DEMO = True to False for the clean
   production behavior.

3. This one matters most, and many people don't know it:
   Python has no way to "kill" a running thread from outside.
   So the watchdog handles two cases separately:
     - the thread is already dead (exited via exception) → the watchdog can safely restart it ✅
     - the thread is alive but stuck (infinite loop / blocked on a call) → Python can't kill it ❌
       the correct move is to let the whole program exit, and let an external
       supervisor (systemd) pull the whole process back up. this is demoed in the file.

How to run:

On Mac, open "Terminal" (search "Terminal" in Launchpad), paste and hit Enter:

    python3 src/main_controller.py

To quit early: press Ctrl + C
=============================================================================
"""


# ============================================================================
# Part 1: bring in the tools we need
# ============================================================================
# All are Python built-ins; nothing extra to install.

import threading  # multi-threading: let several things run "at the same time"
import queue      # queue: a "conveyor belt" that safely passes data between threads
import time       # time: timing and deliberate waits
import random     # random: for faking sensor data
import sys        # system: writing to screen and controlling exit codes


# ============================================================================
# Part 2: global parameters (configuration all threads can see)
# ============================================================================
# Numbers are grouped at the top; to change sample rate or timeouts later,
# only edit here, no need to dig through the code.

# ---- collection-related ----
SENSOR_POINTS = 96      # how many pressure sensing points on the mattress (12×8 array)
SAMPLE_HZ = 20          # samples per second (20 means one frame every 50 ms)
INFER_BATCH = 4         # how many frames to accumulate before feeding the model (batch them, more efficient)

# ---- queue capacity (protection against memory blow-up) ----
RAW_QUEUE_MAX = 100     # max frames the raw data queue can hold
RESULT_QUEUE_MAX = 50   # max entries the result queue can hold

# ---- alarm ----
PRESSURE_ALERT = 800    # pressure alarm threshold (fake value; a real product needs clinical data)

# ---- 🆕 auto-reconnect related ----
SENSOR_DROP_PROB = 0.01     # probability per frame of "pretend disconnect" (demo only, 1%)
RECONNECT_BASE_DELAY = 0.5  # first reconnect waits 0.5 s
RECONNECT_MAX_DELAY = 8.0   # reconnect wait caps at 8 s (can't grow forever)
RECONNECT_SUCCESS_RATE = 0.6  # success probability per reconnect attempt (simulates real-world failures)

# ---- 🆕 watchdog-related ----
WATCHDOG_CHECK_INTERVAL = 1.0  # how often the watchdog patrols (seconds)
WATCHDOG_TIMEOUT = 3.0         # seconds of silence before judging trouble (seconds)
# ⚠️ how to pick this 3 s? it must exceed "the max interval between two feeds during normal work".
#    our collection thread feeds at most every few dozen ms, so 3 s is plenty, no false alarms.
#    in a real product, set it by the slowest component; don't guess.

# ---- 🆕 demo switches: inject faults on purpose so you can see the self-healing process ----
# change this line to False and the program becomes a clean "always runs normally" version.
DEMO = True

DEMO_CRASH_AT_BATCH = 5    # the inference thread crashes on purpose at batch 5
                           # (demo: thread died → watchdog detects → auto-restart)
DEMO_FREEZE_AT_BATCH = 12  # the inference thread "freezes" on purpose at batch 12
                           # (demo: thread faked death → Python can't kill it → only a full program restart)
                           # to skip this demo, change it to 99999


# ============================================================================
# Part 3: four "fake" functions — replace these four once you have real hardware
# ============================================================================


# ---- 🆕 first define an "exception" type: sensor disconnected ----
# An exception is the signal that "something went wrong".
# Defining your own exception type helps you tell at a glance whether it's a
# "disconnect" or some other problem.
class SensorDisconnected(Exception):
    """Sensor disconnected (serial can't read data, device dropped, etc.)"""
    pass


def open_sensor_port():
    """
    ★★★ replacement point 1 of 4 ★★★ — open the sensor connection (fake for now)

    In a real product it's roughly:
        import serial
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        return ser

    For now it pretends to "open the serial port", with a chance of failure
    (simulating that a real reconnect doesn't always succeed on the first try).

    Returns: a "connection object" (here just a dict, for show)
    On failure: raises SensorDisconnected
    """
    # random() returns a float between 0 and 1. below the success rate counts as "connected"
    if random.random() < RECONNECT_SUCCESS_RATE:
        return {"port": "fake_serial", "opened_at": time.time()}
    else:
        raise SensorDisconnected("failed to open serial port (simulated: device not ready yet)")


def read_one_frame_from_sensor():
    """
    ★★★ replacement point 2 of 4 ★★★ — read one sensor frame (fake for now)

    In a real product:
        data = serial_port.read(192)   # read 192 bytes from the serial port
        frame = parse(data)            # split into 96 numbers per the protocol
        return frame

    For now it fakes: return 96 random numbers in 0~1000, occasionally one very
    large to trigger the alarm. Also, with probability SENSOR_DROP_PROB it
    pretends to disconnect, to demo auto-reconnect.

    Returns: a list of length 96
    On trouble: raises SensorDisconnected
    """
    # 🆕 simulated disconnect: roll a die, 1% chance of "dropping"
    if random.random() < SENSOR_DROP_PROB:
        raise SensorDisconnected("serial read timeout (simulated: cable loose / device powered off)")

    # list comprehension: build 96 random numbers in one go
    frame = [random.randint(0, 1000) for _ in range(SENSOR_POINTS)]

    # simulate "a person turning over and pressing on some point"
    if random.random() < 0.05:
        frame[random.randrange(SENSOR_POINTS)] = random.randint(850, 1000)

    return frame


def preprocess(frames):
    """
    ★★★ replacement point 3 of 4 ★★★ — data preprocessing (fake for now)

    In a real product this is where a bunch of serious work happens, and it's important:
      - denoise (filter out jitter/interference)
      - temperature compensation (readings differ between summer and winter, need correction)
      - zero-point calibration (zero out when off the bed, correcting long-term drift)
      - normalization (scale to 0~1, so the model can digest it)

    For now it just computes an average.
    """
    all_values = []
    for frame in frames:
        all_values.extend(frame)   # extend: append one list onto the end of another
    return sum(all_values) / len(all_values)


def run_model_on_npu(data):
    """
    ★★★ replacement point 4 of 4 ★★★ — run the AI model (fake for now)

    In a real product it calls the RKNN SDK (the interface Rockchip provides):
        rknn_outputs = rknn_runtime.inference(...)

    ⚠️ key point: this step does NOT bog down Python. The actual computing is done
    by the NPU hardware; Python just "places the order" then "waits for the result",
    releasing the GIL while waiting.

    For now it uses sleep(0.03) to pretend to be "waiting for the NPU".
    """
    time.sleep(0.03)  # pretend to wait for the NPU; delete in a real product

    if data < 300:
        posture = "off_bed"
    elif data < 500:
        posture = "supine"
    elif data < 700:
        posture = "side"
    else:
        posture = "prone"

    return {
        "posture": posture,
        "avg_pressure": round(data, 1),
        "confidence": round(random.uniform(0.75, 0.99), 2),
    }


# ============================================================================
# 🆕 Part 4: heartbeat utilities — two small "feed the dog" functions
# ============================================================================
# [What is a heartbeat / feeding the dog?]
#
# Each working thread, every so often, "checks in" and records the current time
# into a small notebook. This action is called "feeding the dog".
#
# Why call it "feeding the dog"? Because a real hardware watchdog works like this:
#   you keep a dog with a countdown. you must feed it regularly, or it barks.
#   if you pass out and don't feed it, when the countdown ends it barks wildly —
#   and bites you awake (reboots the system).
#
# In software it's the same idea: threads report in periodically, the watchdog
# watches the timestamps; whoever fails to report on time is in trouble.


def beat(heartbeats, hb_lock, name):
    """
    Feed the dog: record the fact that "I'm still alive".

    Args:
      heartbeats — the notebook (dict), format {"collect_thread": 1234567890.1, ...}
                   the value is "the last feed time"
      hb_lock    — a lock. because the notebook is shared by all threads, writing
                   simultaneously would clash, so lock before writing and unlock
                   after. the with statement unlocks automatically, so you don't
                   have to remember unlock(); this is the recommended style.
      name       — who is feeding the dog
    """
    with hb_lock:
        heartbeats[name] = time.time()   # time.time() is the "now" timestamp (seconds)


def last_beat(heartbeats, hb_lock, name):
    """
    Check when someone last fed the dog.

    Returns: a timestamp (seconds). If never fed, returns None.
    """
    with hb_lock:
        return heartbeats.get(name)      # get returns None if missing, no error


# ============================================================================
# 🆕 Part 5: auto-reconnect on sensor disconnect
# ============================================================================


def reconnect_sensor(stop_event, heartbeats, hb_lock, name):
    """
    Reconnect logic after the sensor disconnects.

    [Why not just while True and retry forever?]
    Because if the device is truly dead (e.g. powered off), retrying every
    microsecond would max the CPU, flood the log, and slow other work. That's
    called "busy waiting", and it's a bad habit.

    [The correct approach is "exponential backoff"]
    1st failure → wait 0.5 s
    2nd failure → wait 1 s
    3rd failure → wait 2 s
    4th failure → wait 4 s
    5th and later → at most 8 s (capped, can't grow forever)
    This neither gives up nor tortures the machine. every serious networked
    program does this.

    Returns: True = connected; False = the program is shutting down, stop trying
    """
    attempt = 0   # which attempt

    while not stop_event.is_set():   # keep trying until the "shutdown" signal
        attempt += 1

        # how long to wait this time: 0.5 × 2^(n-1), capped at 8 s
        delay = min(RECONNECT_BASE_DELAY * (2 ** (attempt - 1)), RECONNECT_MAX_DELAY)
        print(f"[{name}] attempt {attempt} to reconnect, trying in {delay} s...")

        # ---- wait in segments, don't sleep straight through ----
        # why not just time.sleep(delay) in one line?
        # because during that deep sleep you can't receive the "shutdown" signal;
        # the program would hang here and won't close. so sleep in 0.1 s segments,
        # checking the stop flag every segment.
        waited = 0.0
        while waited < delay and not stop_event.is_set():
            time.sleep(0.1)
            waited += 0.1
            # ⚠️ key: keep feeding the dog during the wait!
            # otherwise the watchdog would think "this thread is dead" and restart you — wrongly.
            beat(heartbeats, hb_lock, name)

        if stop_event.is_set():   # received the shutdown signal during the wait
            return False

        # ---- actually connect ----
        try:
            open_sensor_port()
            return True           # connected
        except SensorDisconnected:
            continue              # not connected, try again next round (wait time doubles)


# ============================================================================
# Part 6: what each of the four threads does (the core of this file)
# ============================================================================


def collect_worker(stop_event, raw_queue, heartbeats, hb_lock):
    """
    [Collection thread] reads from the sensor nonstop and pushes frames into the queue.
                 Reconnects on its own when the sensor drops.

    ⚠️ Note this function's structure — the standard shape of a "self-healing thread":
       try:     do the normal work
       except:  something went wrong → log it → return directly (no sentinel; let the watchdog restart me)
       normal exit: reaching the end, put a sentinel to tell downstream to shut down

       Why distinguish these two exits?
       - crash exit: I'm about to be restarted, so I must NOT tell downstream "done", or the whole line stops
       - normal shutdown: the program is closing, must tell downstream, or it will wait forever
    """
    name = "collect_thread"
    dropped = 0

    print(f"[{name}] started, sampling at {SAMPLE_HZ} Hz")

    try:
        while not stop_event.is_set():

            # ---- 0. feed the dog before every work item ----
            beat(heartbeats, hb_lock, name)

            # ---- 1. read one frame (may raise a "disconnected" exception) ----
            try:
                frame = read_one_frame_from_sensor()
            except SensorDisconnected as e:
                # 🆕 disconnected, enter the reconnect flow
                print(f"[{name}] ⚠️  sensor disconnected: {e}")
                ok = reconnect_sensor(stop_event, heartbeats, hb_lock, name)
                if ok:
                    print(f"[{name}] ✅ reconnected, continuing to collect")
                    continue    # continue to the next frame
                else:
                    # interrupted during reconnect = the program is shutting down.
                    # ⚠️ must also put a sentinel here!
                    # otherwise the downstream inference thread never gets notified,
                    # and will hang on get() forever, until it's forcibly killed by
                    # daemon, printing a "did not exit cleanly" warning.
                    raw_queue.put(None)
                    return

            # ---- 2. put into the queue ----
            try:
                raw_queue.put(frame, timeout=0.5)   # always give a timeout!
            except queue.Full:
                # queue full means downstream can't keep up. Drop this frame, don't block.
                # remember: dropping a few frames is fine, a hung program is not.
                dropped += 1
                if dropped % 50 == 0:
                    print(f"[{name}] warning: can't keep up, dropped {dropped} frames")

            # ---- 3. take a breath, control the sample rate ----
            time.sleep(1.0 / SAMPLE_HZ)

    except Exception as e:
        # unexpected trouble (e.g. out of memory, protocol parsing crash)
        # don't re-raise; just log it and exit so the watchdog can detect "I'm dead"
        print(f"[{name}] ❌ unexpected crash: {e!r} (waiting for watchdog restart)")
        return    # [key] return directly, no sentinel

    # ---- reaching here = normal shutdown (got the stop flag) ----
    # put a None sentinel to tell downstream: "I'm done, you can shut down too"
    raw_queue.put(None)
    print(f"[{name}] shut down (dropped {dropped} frames total)")


def next_batch_no(state, state_lock):
    """
    🆕 Fetch a "batch number", incrementing by one each call.

    ⚠️ Why not just use a plain variable n_batch += 1?
    Because this counter must stay continuous across thread restarts. A local
    variable inside the thread resets to 0 whenever the thread restarts — in this
    demo it would keep crashing on the same batch forever. Same in a real product:
    after a restart you need to know "which batch I've reached", e.g. for logic
    like "self-check every 1000 batches" — a reset counter would ruin it.

    So the counter lives in a shared dict, protected by a lock.
    """
    with state_lock:
        state["batch"] += 1
        return state["batch"]


def infer_worker(raw_queue, result_queue, heartbeats, hb_lock, state, state_lock):
    """
    [Inference thread] pulls data from the queue, batches it, feeds the model,
    and pushes results into the result queue.
    """
    name = "infer_thread"
    print(f"[{name}] started, inferring once every {INFER_BATCH} frames")

    batch = []        # basket for collecting data (thread-local; clears on restart, fine)

    try:
        while True:

            # ---- 1. take one frame ----
            # use get(timeout=0.5) instead of get() here:
            # when there's no data it still wakes every 0.5 s to feed the dog,
            # proving "I'm alive, just idle". a timeout-less get() would block
            # forever with no chance to feed the dog — the watchdog would wrongly
            # think you're dead. this is a very common pitfall.
            try:
                frame = raw_queue.get(timeout=0.5)
            except queue.Empty:
                beat(heartbeats, hb_lock, name)   # idle but still report alive
                continue

            # ---- 2. is this the "shutdown signal" ----
            if frame is None:
                raw_queue.task_done()
                result_queue.put(None)   # pass the signal downstream
                break

            # ---- 3. put into the basket ----
            batch.append(frame)
            raw_queue.task_done()

            # ---- 4. when full, feed the model ----
            if len(batch) >= INFER_BATCH:
                # batch number comes from the shared counter, so it survives restarts
                n_batch = next_batch_no(state, state_lock)

                # 🆕 demo 1: crash once on purpose, so you can see "thread died → restarted"
                if DEMO and n_batch == DEMO_CRASH_AT_BATCH:
                    raise RuntimeError("simulated: RKNN inference returned error code -5")

                # 🆕 demo 2: freeze on purpose, so you can see "a frozen thread can't be killed by Python"
                if DEMO and n_batch == DEMO_FREEZE_AT_BATCH:
                    print(f"[{name}] 💀 simulated: NPU driver frozen, this thread will stop feeding the dog and working")
                    time.sleep(300)   # pretend to freeze for 5 minutes; the watchdog will notice meanwhile

                data = preprocess(batch)
                result = run_model_on_npu(data)
                result_queue.put(result)
                batch = []

                beat(heartbeats, hb_lock, name)   # feed the dog after finishing work

    except Exception as e:
        print(f"[{name}] ❌ unexpected crash: {e!r} (waiting for watchdog restart)")
        return    # no sentinel, wait to be restarted

    print(f"[{name}] shut down")


def watchdog_worker(stop_event, threads, heartbeats, hb_lock, restart_queue):
    """
    🆕 [Watchdog thread] does nothing but watch whether others feed the dog on time.

    This is the core of the whole self-healing mechanism. It patrols every second,
    checking each thread's "last feed time".

    ⚠️ It distinguishes two cases, handled completely differently:

      case A: the thread is already dead (thread.is_alive() returns False)
              → it exited via an exception, memory is clean, can safely rebuild a new one ✅

      case B: the thread is alive but timed out on feeding the dog
              → it's stuck somewhere (infinite loop, blocked on a call)
              → ❌ Python has no way to forcibly kill a thread from outside!
                 (a Python design limitation, not a bug in your code)
              → the only correct move: let the whole program exit, and have an
                 external supervisor (systemd on Linux) pull the whole process
                 back up. So the program should end with a non-zero exit code —
                 systemd restarts only on non-zero.

    Args:
      threads       — a dict {"collect_thread": thread object, "infer_thread": thread object}
                      Pass a dict rather than single objects because after a thread
                      is restarted it becomes a new object; the dict is mutable, so
                      the watchdog always reads the latest one.
      restart_queue — when it finds a problem, it puts a "request" into this queue
                      for the main thread to carry out the restart.
                      (why not restart itself? — because after restarting it must
                      update the threads dict; keeping such "management actions" in
                      the main thread is clearer and less error-prone)
    """
    name = "watchdog"
    print(f"[{name}] started, patrolling every {WATCHDOG_CHECK_INTERVAL} s,"
          f" judging trouble after {WATCHDOG_TIMEOUT} s of silence")

    while not stop_event.is_set():
        # patrol interval. also sleep in segments to reliably receive the shutdown signal
        waited = 0.0
        while waited < WATCHDOG_CHECK_INTERVAL and not stop_event.is_set():
            time.sleep(0.2)
            waited += 0.2

        if stop_event.is_set():
            break

        now = time.time()

        # wrap in list(): the main thread may modify the dict while iterating;
        # direct iteration has a tiny chance of trouble, a snapshot copy is safer.
        for tname, thread in list(threads.items()):
            lb = last_beat(heartbeats, hb_lock, tname)

            if lb is None:
                continue   # hasn't fed yet (just started), don't penalize

            silence = now - lb   # how many seconds of silence

            if silence > WATCHDOG_TIMEOUT:
                if not thread.is_alive():
                    # ---- case A: already dead, can safely restart ----
                    print(f"[{name}] 🚨 found [{tname}] stopped ({silence:.1f} s without feeding), requesting restart")
                    restart_queue.put(("THREAD", tname))
                else:
                    # ---- case B: alive but stuck, Python can't kill it ----
                    print(f"[{name}] 🚨 found [{tname}] still alive but {silence:.1f} s without feeding — frozen")
                    print(f"[{name}] ⚠️  Python cannot force-kill a thread, can only request a full program restart")
                    restart_queue.put(("PROCESS", tname))
                    return   # watchdog itself shuts down, leaving the rest to the main thread

    print(f"[{name}] shut down")


def main_ui_loop(result_queue, stop_event, restart_queue,
                 threads, builders, heartbeats, hb_lock):
    """
    [Main thread] three jobs:
      1. check for "restart requests" (sent by the watchdog)
      2. pull inference results and display them on screen
      3. decide whether to alarm

    Why must the UI live on the main thread?
    Because UI libraries like Qt and Tkinter require UI operations on the main
    thread only; doing it elsewhere crashes.

    Returns: 0 = normal exit; 1 = needs a full program restart (hand off to systemd)
    """
    name = "main_thread"
    print(f"[{name}] started, displaying results\n")

    frame_count = 0

    # note the loop is controlled by stop_event, not "exit on receiving a sentinel".
    # why? because if some thread crashes, it may not have time to send a sentinel;
    # if the main thread only waits for sentinels, it would wait forever. the stop
    # flag is more reliable.
    while not stop_event.is_set():

        # ---- job 1: handle restart requests ----
        # get_nowait() = "take it if present, don't wait, raise immediately if empty". here we want "don't wait".
        try:
            kind, who = restart_queue.get_nowait()
        except queue.Empty:
            kind, who = None, None

        if kind == "THREAD":
            restart_thread(who, threads, builders, heartbeats, hb_lock)
        elif kind == "PROCESS":
            # a thread is frozen and can't be killed by Python; the whole program must exit
            print(f"\n[{name}] ❌ [{who}] is frozen and can't be restarted alone; the program will exit.")
            print(f"[{name}]    in a real product, systemd sees the non-zero exit code and pulls the whole process back up.\n")
            return 1   # non-zero exit code = "I exited abnormally, please restart me"

        # ---- job 2: pull one inference result ----
        try:
            result = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue    # no result, loop back to check restart requests again

        if result is None:      # received a sentinel
            result_queue.task_done()
            break

        result_queue.task_done()
        frame_count += 1

        # ---- display + alarm ----
        print(f"  batch {frame_count:>4} | "
              f"posture: {result['posture']:<4} | "
              f"avg_pressure: {result['avg_pressure']:>6} | "
              f"confidence: {result['confidence']}")

        # ⚠️ in a real product the alarm logic must never be this crude:
        #     the cost of a missed alarm (real trouble not reported) >> a false alarm (crying wolf)
        #     the threshold should be tuned toward "better to alarm a bit more often".
        if result["avg_pressure"] > PRESSURE_ALERT:
            print(f"          >>> Alarm: local pressure too high ({result['avg_pressure']}),"
                  f"pressure-ulcer risk, please assist turning <<<")

        sys.stdout.flush()   # flush to screen immediately, otherwise it looks stuck

    print(f"\n[{name}] shut down, processed {frame_count} batches")
    return 0


def restart_thread(tname, threads, builders, heartbeats, hb_lock):
    """
    🆕 Restart a thread.

    ⚠️ The easiest pitfall here:
        the same Thread object can only be start()-ed once; a second start()
        raises "threads can only be started once".

        So restarting is not "starting the old thread again", but creating a
        brand-new thread. That's why there's a builders dict: it stores the
        "thread factory" methods, and each restart calls it to build a new one.

    Args:
      tname     — which thread to restart ("collect_thread" / "infer_thread")
      threads   — the threads dict; after restart, write the new thread object back
      builders  — {"collect_thread": function that builds the collection thread,
                   "infer_thread": function that builds the inference thread}
    """
    name = "main_thread"

    old = threads.get(tname)
    if old is not None and old.is_alive():
        # shouldn't happen in theory (the watchdog only requests restart when dead),
        # but an extra guard doesn't hurt: avoid accidentally starting two threads
        # of the same name, which would corrupt the data.
        print(f"[{name}] [{tname}] is still running, won't restart redundantly")
        return

    new_thread = builders[tname]()      # build a brand-new thread
    threads[tname] = new_thread         # update the dict so the watchdog watches the new one

    # ⚠️ must reset the heartbeat!
    # otherwise the new thread hasn't fed the dog yet, and the watchdog sees "the
    # timestamp is still old" and immediately judges it dead again, falling into a
    # "restart forever" loop.
    beat(heartbeats, hb_lock, tname)

    new_thread.start()
    print(f"[{name}] ✅ [{tname}] restarted")


# ============================================================================
# Part 7: program entry — assemble the pieces above
# ============================================================================


def main():
    """
    Main function: build queues → build threads → start → supervise → wrap up
    """
    print("=" * 66)
    print(" AI medical bed — edge main-controller skeleton V2 (with auto-reconnect + watchdog)")
    print(" Press Ctrl + C to quit")
    if DEMO:
        print(" ⚠️ demo mode: faults are injected on purpose so you can see the self-healing process")
        print(f"    · batch {DEMO_CRASH_AT_BATCH}: inference thread crashes → watchdog restarts it")
        print(f"    · batch {DEMO_FREEZE_AT_BATCH}: inference thread freezes → only a full program restart works")
        print("    · random: sensor disconnects → auto-reconnects")
    print("=" * 66)
    print()

    # ---- 1. stop flag ----
    stop_event = threading.Event()

    # ---- 2. three conveyor belts ----
    raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAX)        # collection → inference
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)  # inference → main thread
    restart_queue = queue.Queue()                         # 🆕 watchdog → main thread

    # ---- 3. 🆕 heartbeat notebook + its lock ----
    heartbeats = {}                    # {"collect_thread": timestamp, "infer_thread": timestamp}
    hb_lock = threading.Lock()         # the lock protecting this notebook

    # ---- 3b. 🆕 shared counter (continuous across thread restarts) ----
    state = {"batch": 0}
    state_lock = threading.Lock()

    # ---- 4. 🆕 builders dict: needed at restart time ----
    # this uses "closures": functions defined inside functions can remember the
    # outer variables (e.g. raw_queue, stop_event), so building threads doesn't
    # need to pass a pile of arguments.
    def build_collect():
        return threading.Thread(
            target=collect_worker,
            args=(stop_event, raw_queue, heartbeats, hb_lock),
            daemon=True,
            name="collect_thread",
        )

    def build_infer():
        return threading.Thread(
            target=infer_worker,
            args=(raw_queue, result_queue, heartbeats, hb_lock, state, state_lock),
            daemon=True,
            name="infer_thread",
        )

    builders = {
        "collect_thread": build_collect,
        "infer_thread": build_infer,
    }

    # ---- 5. build and start threads ----
    threads = {}
    threads["collect_thread"] = build_collect()
    threads["infer_thread"] = build_infer()

    # 🆕 watchdog thread (it doesn't need supervising itself; keep it simple)
    t_watchdog = threading.Thread(
        target=watchdog_worker,
        args=(stop_event, threads, heartbeats, hb_lock, restart_queue),
        daemon=True,
        name="watchdog_thread",
    )

    threads["collect_thread"].start()
    threads["infer_thread"].start()
    t_watchdog.start()

    # ---- 6. the main thread starts working (display + alarm + handle restart requests) ----
    exit_code = 0
    try:
        exit_code = main_ui_loop(
            result_queue, stop_event, restart_queue,
            threads, builders, heartbeats, hb_lock,
        )
    except KeyboardInterrupt:
        print("\n\n[main_thread] received Ctrl+C, shutting down...")
    finally:
        # no matter how it ends, raise the flag to tell all threads to shut down
        stop_event.set()

        # wait for them to shut down. always give a timeout, or if someone is
        # stuck we'd wait forever
        for tname, t in threads.items():
            t.join(timeout=2)
            if t.is_alive():
                print(f"[main_thread] warning: {tname} did not exit cleanly (possibly stuck)")
        t_watchdog.join(timeout=2)

    print("\n[main_thread] program exited safely")
    return exit_code


# ============================================================================
# Part 8: Python's fixed boilerplate
# ============================================================================
# Only runs main() when this file is executed directly; not when imported.
if __name__ == "__main__":
    # sys.exit(code): 0 means normal, non-zero means abnormal exit.
    # in real deployment, systemd config Restart=on-failure,
    # and seeing non-zero will pull the whole program back up — the last line of defense.
    sys.exit(main())


# ============================================================================
# [Appendix] questions you may ask
# ============================================================================
#
# Q1: why use queue.Queue instead of a plain list?
# A: because Queue has a built-in lock; multiple threads put/get without errors.
#    a plain list under multi-threading can miscalculate or drop data. always use Queue, not list.
#
# Q2: why threads instead of processes?
# A: because the work here is mostly waiting: for serial data, for the NPU to finish.
#    while waiting, Python releases the GIL, so multi-threading is genuinely effective.
#    when to use processes? — when the bottleneck is pure-Python intensive computation.
#
# Q3: why not asyncio?
# A: asyncio is designed for tens of thousands of concurrent connections (web servers, crawlers).
#    here concurrency is single-digit. besides, pyserial and the RKNN SDK are synchronous/blocking;
#    cramming them into asyncio would stall the event loop, and you'd end up back at a thread pool anyway. wasted effort.
#
# Q4: 🆕 how to choose the watchdog timeout (3 s)?
# A: it must be longer than the max interval between two feeds during normal work, with 3~5× headroom.
#    our collection thread feeds at most every few dozen ms, so 3 s is plenty.
#    too short → false alarms (constant restarts); too long → missed alarms (trouble found too late).
#    in a real product, measure the slowest component and set it accordingly, don't guess.
#
# Q5: 🆕 why can't a "frozen" thread be restarted like a "dead" one?
# A: it's a hard Python limitation — there is no official API to kill a running thread from outside.
#    (the internet's tricks using signals or ctypes are all unsafe, leaving a mess:
#       unreleased locks, unreclaimed resources — worse, not better.)
#    the only proper way: let the whole process exit and have systemd pull it back up.
#    so production deployment must configure systemd, with Restart=on-failure.
#
# Q6: 🆕 while reconnecting, why not just time.sleep(delay)?
# A: because during that deep sleep you can't receive the "shutdown" signal; the program hangs and won't close.
#    the correct way is to sleep in small segments, checking the stop flag each segment.
#    this applies to all waiting scenarios — a universal habit.
#
# Q7: 🆕 why should the reconnect wait "grow longer" (exponential backoff)?
# A: to avoid busy-waiting. if the device is truly dead, retrying a thousand times a second
#    just maxes the CPU, floods the log, and slows other work. gradually lengthening the
#    interval keeps trying without torturing the machine.
#
# Q8: in a real product, what's still missing from this skeleton?
# A: at least:
#     - data persistence (save a local copy, traceable after an incident)
#     - a logging system (print is a toy; use the logging module, which can
#                  auto-split files, levels, and timestamps)
#     - resource cleanup (serial port, NPU handles must be closed explicitly, via finally)
#     - a hardware watchdog (the real one on the board; if software dies it can power-cycle)
#     - power-loss protection (data must not be lost on sudden power-off)
#     add these layer by layer once the skeleton runs; don't try to build everything upfront.
#
# ============================================================================
