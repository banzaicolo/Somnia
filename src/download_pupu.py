#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PoPu 真实数据下载器 —— 从"模拟数据"迈向"真实数据"的入口
=============================================================================

【PoPu 是什么？】

一个公开发布的科研数据集（CC0 许可，随便用、可商用）：
60 个志愿者 × 28 种姿势，共 5 万多张真实的床垫压力图，
而且分「床垫上 + 床垫下」两层传感器——跟咱们项目场景一模一样。

项目主页：https://github.com/rdionisio1403/PoPu

【这个脚本干嘛？】

把数据从 GitHub 下载下来并解压。两种可选：

  预览版（2.5MB，先小后大验证流程）：  python3 src/download_pupu.py preview
  完整版（82MB，真正做训练用）：      python3 src/download_pupu.py full

数据会下载到项目根目录下的 pupu_data/ 文件夹。

【为什么默认不自动下完整版？】

尊重你的带宽和时间。先用预览版确认网络通、能解压、能读，
再下完整版不迟。
=============================================================================
"""

import os
import sys
import urllib.request
import zipfile

# 数据文件在 GitHub 仓库里的真实地址（raw 直链）
BASE_URL = "https://github.com/rdionisio1403/PoPu/raw/main/"
FILES = {
    "preview": ("preview_data.zip", 2.6 * 1024 * 1024),
    "full": ("PoPu_data.zip", 82 * 1024 * 1024),
}
OUT_DIR = "pupu_data"


def download(which="preview"):
    """下载并解压指定版本的数据。"""
    filename, size_hint = FILES[which]
    url = BASE_URL + filename
    zip_path = os.path.join(OUT_DIR, filename)

    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(zip_path):
        print(f"[跳过下载] {zip_path} 已存在。")
    else:
        print(f"开始下载 {which} 版（约 {size_hint // 1024 // 1024 + 1} MB）……")
        print(f"  地址：{url}")
        # urllib 是 Python 自带的下载工具，不用装任何东西
        urllib.request.urlretrieve(url, zip_path)
        print(f"  已保存到 {zip_path}")

    # 解压
    print("解压中……")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(OUT_DIR)
    print(f"✅ 完成。数据在 {OUT_DIR}/ 文件夹里。")

    # 打印解压出来的内容，让你知道里面有什么
    for name in sorted(os.listdir(OUT_DIR)):
        full = os.path.join(OUT_DIR, name)
        if os.path.isdir(full):
            print(f"  [目录] {name}")
        else:
            print(f"  [文件] {name}  ({os.path.getsize(full) // 1024} KB)")


if __name__ == "__main__":
    # 命令行参数：preview（默认）或 full
    which = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if which not in FILES:
        print(f"参数只能是 {' 或 '.join(FILES)}，收到的是 {which}")
        sys.exit(1)
    download(which)
