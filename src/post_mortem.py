import pandas as pd
import os
from datetime import datetime
from pathlib import Path

from src.forecaster import WyckoffForecaster
from src.smart_money import SmartMoneyEngine
from src.shadow_profiler import ShadowProfiler

# Tắt cảnh báo Pandas để output sạch sẽ
import warnings
warnings.filterwarnings('ignore')

class PostMortemAnalyzer:
    """
    Công cụ Khám nghiệm Tử thi Định lượng (Post-mortem Analysis).
    Dùng để truy vết nguyên nhân tại sao Bot lỡ một siêu cổ phiếu, 
    hoặc phân tích lại một kèo cắt lỗ trong quá khứ.
    """
    def __init__(self, data_dir='data/parquet', universe='VN30'):
        self.data_dir = Path(data_dir)
        self.universe = universe
        self.price_path = self.data_dir / 'price/master_price.parquet'
        self.foreign_path = self.data_dir / 'macro/foreign_flow.parquet'
        self.prop_path = self.data_dir / 'macro/prop_flow.parquet'
        self.comp_path = self.data_dir / 'company/master_company.parquet'
        
        print("\n" + "="*80)
        print(" 🔍 KHỞI ĐỘNG CÔNG CỤ KHÁM NGHIỆM ĐỊNH LƯỢNG (POST-MORTEM)")
        print("="*80)
        self._load_data()

    def _load_data(self):
        print("[*] Đang nạp dữ liệu Master Parquet...")

        df_price = self._load_parquet_safe(self.price_path)
        if not df_price.empty:
            df_price['time'] = pd.to_datetime(df_price['time']).dt.normalize()
        self.price_dict = self._load_data_dict(self.price_path)

        foreign_dict = self._load_data_dict(self.foreign_path)
        prop_dict = self._load_data_dict(self.prop_path)

        out_shares_dict = {}
        df_comp = self._load_parquet_safe(self.comp_path)
        if not df_comp.empty and 'ticker' in df_comp.columns and 'issue_share' in df_comp.columns:
            out_shares_dict = df_comp.set_index('ticker')['issue_share'].to_dict()
        
        # Khởi tạo các Động cơ Phân tích
        self.sm_engine = SmartMoneyEngine(
            foreign_dict=foreign_dict, 
            prop_dict=prop_dict, 
            out_shares_dict=out_shares_dict,
            price_dict=self.price_dict,
            universe=self.universe 
        )
        self.profiler = ShadowProfiler(df_price, verbose=False)

    def _load_parquet_safe(self, path):
        if path.exists():
            try: 
                return pd.read_parquet(path)
            except: 
                print(f"Could NOT read {path}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_data_dict(self, path):
        df = self._load_parquet_safe(path)
        if df.empty or 'ticker' not in df.columns:
            return {}

        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.normalize()
            
        data_dict = {}
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('time').tail(130)
            data_dict[ticker] = group
            
        return data_dict

    def analyze(self, ticker, target_date_str, lookback_days=15):
        """Khám nghiệm 1 mã cổ phiếu tại 1 thời điểm cụ thể, lùi về N ngày."""
        print(f"\n[🔬] KHÁM NGHIỆM MÃ: {ticker} | NGÀY SỰ KIỆN: {target_date_str} | QUÉT LÙI: {lookback_days} PHIÊN")
        print("-" * 80)

        df_p = self.price_dict.get(ticker)
        if df_p is None or df_p.empty:
            print(f"[!] Không tìm thấy dữ liệu giá cho mã {ticker}.")
            return

        target_date = pd.to_datetime(target_date_str).normalize()
        df_p_valid = df_p[df_p['time'] <= target_date]
        
        if df_p_valid.empty:
            print(f"[!] Mã {ticker} chưa niêm yết hoặc không có dữ liệu trước ngày {target_date_str}.")
            return

        # Lấy danh sách N phiên giao dịch gần nhất
        trading_dates = df_p_valid['time'].sort_values().unique()
        recent_dates = trading_dates[-lookback_days:]

        print(f"{'NGÀY (T-X)':<15} | {'GIÁ ĐÓNG':<10} | {'TÍN HIỆU WYCKOFF (KỸ THUẬT)':<35} | {'SMART MONEY & LÁI NỘI (DÒNG TIỀN)'}")
        print("-" * 110)

        # Mô phỏng chạy qua từng ngày
        for date_obj in recent_dates:
            date_str = pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            is_target_day = (date_obj == target_date)
            date_label = f"🔥 {date_str} (T0)" if is_target_day else f"{date_str}"
            
            # --- 1. Chạy Kỹ Thuật (Wyckoff) ---
            # Để công bằng, ta phải cắt dữ liệu giá ĐÚNG tại ngày đang xét
            df_hist_cut = df_p[df_p['time'] <= date_obj].copy()
            forecaster = WyckoffForecaster(price_dir=df_hist_cut, verbose=False)
            df_forecast = forecaster.forecast()
            
            signal = "Không"
            close_price = 0
            if df_forecast is not None and not df_forecast.empty:
                last_row = df_forecast.iloc[-1]
                close_price = last_row.get('close', 0)
                signal_raw = last_row.get('Signal', 'Không')
                if signal_raw in ['SOS', 'SPRING', 'TEST_CUNG', 'BU']:
                    signal = f"{signal_raw} (+30đ)"

            # --- 2. Chạy Smart Money ---
            sm_result = self.sm_engine.analyze_ticker(ticker, board_info=None, target_date=date_str)
            sm_score = sm_result.get('total_sm_score', 0)
            sm_danger = sm_result.get('is_danger', False)
            sm_msg = ""
            if sm_danger: sm_msg += "[🚨 BÁO ĐỘNG ĐỎ] "
            if sm_score != 0: sm_msg += f"SM Điểm: {sm_score} "

            # --- 3. Chạy Shadow Profiler ---
            shadow_msg = ""
            alerts = self.profiler.live_shadow_radar([ticker], self.profiler._compile_rules(), target_date=date_str)
            if alerts:
                shadow_msg = f"| Lái nội: {alerts[0]['Status']}"

            # --- 4. Tổng hợp & In ra dòng ---
            flow_info = f"{sm_msg} {shadow_msg}".strip()
            if not flow_info: flow_info = "Bình thường"

            print(f"{date_label:<15} | {close_price:<10,.0f} | {signal:<35} | {flow_info}")

        print("=" * 110)
        self._print_conclusion(ticker, recent_dates[-1], sm_result, signal)

    def _print_conclusion(self, ticker, target_date, final_sm_result, final_signal):
        """Phân tích tổng hợp để đưa ra Lời biện hộ cho Bot"""
        print(f"\n🧠 KẾT LUẬN TỪ TRUNG TÂM PHÂN TÍCH (TẠI NGÀY {pd.to_datetime(target_date).strftime('%Y-%m-%d')}):")
        
        if final_signal != "Không":
            print(f"- [Kỹ thuật]: Tốt. Mã {ticker} ĐÃ có tín hiệu Wyckoff ({final_signal}).")
        else:
            print(f"- [Kỹ thuật]: KHÔNG CÓ TÍN HIỆU WYCKOFF. Bot bỏ qua vì Cấu trúc giá/Khối lượng không chuẩn form tích lũy.")

        if final_sm_result.get('is_danger', False):
            warnings = final_sm_result.get('warnings', [])
            print(f"- [Dòng tiền]: BÁO ĐỘNG ĐỎ. Bot bị chặn mua do Tây/Tự doanh xả rát. Lý do: {', '.join(warnings)}.")
        elif final_sm_result.get('total_sm_score', 0) < 0:
             print(f"- [Dòng tiền]: Xấu (Điểm âm). Bot hạ tỷ trọng hoặc từ chối vì thiếu sự ủng hộ của tay to.")
             
        print("\n=> [LỜI BIỆN HỘ CỦA BOT]: Nếu mã này sau đó vẫn tăng mạnh, đây là lệnh tăng do Yếu tố Ngoại lai (Tin tức M&A, Cờ bạc úp sọt, Giải cứu BCTC...). Bot được lập trình để BỎ QUA các kèo đánh bạc này nhằm bảo vệ vốn!")

if __name__ == "__main__":
    analyzer = PostMortemAnalyzer()
    
    print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
    print("Nhập mã cổ phiếu và ngày nó bắt đầu chạy mạnh mà Bot không báo.")
    print("Gõ 'q' để thoát.")
    
    while True:
        ticker = input("\nNhập Mã Cổ Phiếu (VD: FPT): ").strip().upper()
        if ticker == 'Q': break
        
        date_input = input("Nhập Ngày Nổ Điểm (Định dạng YYYY-MM-DD, VD: 2026-04-10): ").strip()
        if date_input == 'Q': break
        
        try:
            # Test tính hợp lệ của ngày
            pd.to_datetime(date_input)
            analyzer.analyze(ticker=ticker, target_date_str=date_input, lookback_days=15)
        except Exception as e:
            print(f"[!] Lỗi định dạng ngày hoặc dữ liệu: {e}. Vui lòng thử lại.")