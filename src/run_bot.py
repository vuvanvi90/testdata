import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.live import LiveAssistant, DualLogger
from src.reporter import CashFlowReporter
from src.reporter_by_group import GroupCashFlowReporter

def run_trading_system():
    """Hàm điều khiển toàn bộ quy trình chạy Bot"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] 🚀 BẮT ĐẦU PHIÊN LÀM VIỆC. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        bot = LiveAssistant()
        bot.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT: {e}")
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