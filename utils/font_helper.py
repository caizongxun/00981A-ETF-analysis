#!/usr/bin/env python3
"""
跨平台中文字型設定工具
在所有畫圖腳本 import 後呼叫 setup_font() 即可
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

def setup_font():
    system = platform.system()
    font_found = False

    if system == "Windows":
        candidates = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DFKai-SB"]
    elif system == "Darwin":  # macOS
        candidates = ["PingFang TC", "Heiti TC", "STHeiti"]
    else:  # Linux
        candidates = ["Noto Sans CJK TC", "WenQuanYi Zen Hei", "AR PL UMing TW"]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            font_found = True
            break

    if not font_found:
        # fallback: 登錄系統內建字型路徑
        try:
            if system == "Windows":
                fm.fontManager.addfont("C:/Windows/Fonts/msjh.ttc")   # 微軟正黑體
                prop = fm.FontProperties(fname="C:/Windows/Fonts/msjh.ttc")
                plt.rcParams["font.family"] = prop.get_name()
                font_found = True
        except Exception:
            pass

    # 不管有沒有找到字型，都關閉漏字警告
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    return font_found
