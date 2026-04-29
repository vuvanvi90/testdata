import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.live import LiveAssistant, DualLogger
from src.darkpool import DarkPoolRadar
from src.reporter import CashFlowReporter
from src.reporter_by_group import GroupCashFlowReporter

def run_trading_system():
    """Hàm điều khiển toàn bộ quy trình chạy Bot"""
    log_dir = Path("data/logs/hose")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC HOSE. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        bot = LiveAssistant(universe='HOSE')
        bot.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT cho HOSE: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_vn30_live():
    """Hàm điều khiển toàn bộ quy trình chạy Bot"""
    log_dir = Path("data/logs/vn30")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC VN30. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        bot = LiveAssistant(universe='VN30')
        bot.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT cho VN30: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_midcap_live():
    """Hàm điều khiển toàn bộ quy trình chạy Bot"""
    log_dir = Path("data/logs/midcap")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC MIDCAP. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        bot = LiveAssistant(universe='VNMidCap')
        bot.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT cho MidCap: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_smallcap_live():
    """Hàm điều khiển toàn bộ quy trình chạy Bot"""
    log_dir = Path("data/logs/smallcap")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC SMALLCAP. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        bot = LiveAssistant(universe='VNSmallCap')
        bot.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT cho SmallCap: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_darkpool_radar():
    log_dir = Path("data/logs/darkpool")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC DARKPOOL. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        radar = DarkPoolRadar()
        radar.run_radar()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT cho DARKPOOL: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_cashflow_report(timeframe='week', df_foreign=None, df_prop=None, target_date=None):
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"cashflow_{timeframe}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)

    print(f"[*] 📊 BẮT ĐẦU PHIÊN LÀM VIỆC. Log được lưu tại: {log_filename}")

    try:
        reporter = CashFlowReporter(df_foreign, df_prop)
        reporter.generate_report(timeframe, target_date)
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

def run_cashflow_group_report(timeframe='week', df_foreign=None, df_prop=None, df_industry=None, target_date=None):
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"cashflow_group_{timeframe}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)

    print(f"[*] 📊 BẮT ĐẦU PHIÊN LÀM VIỆC. Log được lưu tại: {log_filename}")

    try:
        reporter = GroupCashFlowReporter(df_foreign, df_prop, df_industry)
        reporter.generate_report(timeframe, target_date)
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT: {e}")
        print(traceback.format_exc()) # In chi tiết dòng code gây lỗi vào log
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛑 KẾT THÚC PHIÊN LÀM VIỆC.")

if __name__ == "__main__":
    run_trading_system()