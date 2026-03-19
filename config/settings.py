# config/settings.py
import os
from pathlib import Path

API_KEY = "vnstock_2e9544d83583fddf72c0aa9ec93746bd"
INITIAL_CAPITAL = float(100000000)
TP_PCT = float(0.12) # Mặc định 7% nếu không tìm thấy
SL_PCT = float(0.05) # Mặc định 3% nếu không tìm thấy
EMA_FAST = 34
EMA_SLOW = 89
VOL_MA_PERIOD = 20
# Khoảng thời gian xác định Trading Range (3 tháng)
LOOKBACK_TR = 60
# thị trường tốt thì để 7%, thị trường rủi ro thì bóp chặt lại còn 4-5%
MAX_DIST_TO_SUPPORT = float(os.getenv("MAX_DIST_TO_SUPPORT", 0.07))
# Nâng lên 20 ngày theo khuyến nghị
PRE_TET_FREEZE_DAYS = 20
# Ngưỡng giá để coi là "Thị giá cao"
HIGH_PRICE_THRESHOLD = 100000

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Quản lý dữ liệu
DATA_DIR = BASE_DIR / "data"
INVEST_DIR = DATA_DIR / "invest"
LIVE_DIR = DATA_DIR / "live"

# Tạo thư mục nếu chưa có
for folder in [DATA_DIR, INVEST_DIR, LIVE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)