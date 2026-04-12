import pandas as pd
import os
import traceback
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
        try:
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
        except Exception as e:
            print(f"[!] Lỗi _load_data: {e}")

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
        print(f"\n[🔬] KHÁM NGHIỆM MÃ: {ticker} | NGÀY SỰ KIỆN: {target_date_str} | QUÉT LÙI: {lookback_days} PHIÊN")
        print("-" * 145)

        df_p = self.price_dict.get(ticker)
        if df_p is None or df_p.empty:
            print(f"[!] Không tìm thấy dữ liệu giá cho mã {ticker}.")
            return

        target_date = pd.to_datetime(target_date_str).normalize()
        df_p_valid = df_p[df_p['time'] <= target_date].copy() # Copy để tránh SettingWithCopyWarning
        
        if df_p_valid.empty:
            print(f"[!] Mã {ticker} chưa niêm yết hoặc không có dữ liệu trước ngày {target_date_str}.")
            return

        # 🚀 TÍNH TOÁN HIỆU SUẤT QUÁ KHỨ (% PnL / Momentum)
        # Tính toán trên toàn bộ df_p_valid trước để đạt tốc độ O(1)
        df_p_valid['pct_1d'] = df_p_valid['close'].pct_change(1) * 100
        df_p_valid['pct_5d'] = df_p_valid['close'].pct_change(5) * 100
        df_p_valid['pct_20d'] = df_p_valid['close'].pct_change(20) * 100

        trading_dates = df_p_valid['time'].sort_values().unique()
        recent_dates = trading_dates[-lookback_days:]

        # Căn chỉnh lại Header mở rộng
        print(f"{'NGÀY (T-X)':<15} | {'C-H-L (ĐÓNG/CAO/THẤP)':<22} | {'KHỐI LƯỢNG':<10} | {'%1D':>6} | {'%5D':>6} | {'%20D':>6} | {'WYCKOFF':<18} | {'DÒNG TIỀN & LÁI NỘI'}")
        print("-" * 145)

        sm_result = {}
        final_signal = "Không"

        for date_obj in recent_dates:
            date_str = pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            is_target_day = (date_obj == target_date)
            date_label = f"🔥 {date_str}" if is_target_day else f"{date_str}"
            
            # --- Trích xuất Dữ liệu Kỹ thuật ---
            curr_row = df_p_valid[df_p_valid['time'] == date_obj].iloc[0]
            close_p = curr_row['close']
            high_p = curr_row['high']
            low_p = curr_row['low']
            vol = curr_row['volume']
            
            # Hàm format % an toàn
            def fmt_pct(val): return f"{val:>5.1f}%" if pd.notna(val) else "   -  "
            
            pct_1d_str = fmt_pct(curr_row['pct_1d'])
            pct_5d_str = fmt_pct(curr_row['pct_5d'])
            pct_20d_str = fmt_pct(curr_row['pct_20d'])
            
            price_str = f"{close_p:,.0f} ({high_p:,.0f}/{low_p:,.0f})"
            
            df_forecast = None
            signal = "Không"
            try:
                df_hist_cut = df_p_valid[df_p_valid['time'] <= date_obj].copy()
                forecaster = WyckoffForecaster(data_input=df_hist_cut, output_dir=None, run_date=date_obj)
                df_forecast = forecaster.run_forecast()
            except Exception:
                signal = "[Lỗi WyckoffForecaster]"
                print(f"   [!] Gợi ý gỡ lỗi Forecaster tại ngày {date_str}: {e}")
                traceback.print_exc()
            
            if df_forecast is not None and isinstance(df_forecast, pd.DataFrame) and not df_forecast.empty:
                try:
                    last_row = df_forecast.iloc[-1]
                    close_price = last_row.get('Price', 0)
                    signal = last_row.get('Signal', 'Không')
                except Exception:
                    pass

            # Cập nhật kết luận cho ngày cuối cùng
            if is_target_day: final_signal = signal

            try:
                sm_result = self.sm_engine.analyze_ticker(ticker, board_info=None, target_date=date_str)
            except Exception as e:
                print(f"[!] Lỗi analyze_ticker: {e}")

            sm_score = sm_result.get('total_sm_score', 0)
            sm_danger = sm_result.get('is_danger', False)
            sm_msg = "[🚨 BÁO ĐỘNG ĐỎ] " if sm_danger else ""
            if sm_score != 0: sm_msg += f"SM: {sm_score} "

            shadow_msg = ""
            try:
                rules = self.profiler.build_criminal_profile([ticker], lookback_days=250)
                alerts = self.profiler.live_shadow_radar([ticker], rules, target_date=date_str)
                
                if isinstance(alerts, pd.DataFrame):
                    if not alerts.empty:
                        shadow_msg = f"| Lái nội: {alerts.iloc[0].get('Status', '')}"
                elif isinstance(alerts, list):
                    if len(alerts) > 0:
                        alert_item = alerts[0]
                        if isinstance(alert_item, dict):
                            shadow_msg = f"| Lái nội: {alert_item.get('Status', '')}"
                        else:
                            shadow_msg = f"| Lái nội: {alert_item}"
            except Exception as e:
                shadow_msg = f"| Lái nội: [Lỗi Radar - {e}]"

            flow_info = f"{sm_msg} {shadow_msg}".strip()
            if not flow_info: flow_info = "Bình thường"

            # In format mở rộng
            print(f"{date_label:<15} | {price_str:<22} | {vol:10,.0f} | {pct_1d_str} | {pct_5d_str} | {pct_20d_str} | {signal:<18} | {flow_info}")

        print("=" * 145)
        self._print_conclusion(ticker, recent_dates[-1], sm_result, final_signal)


    def _print_conclusion(self, ticker, target_date, final_sm_result, final_signal):
        print(f"\n🧠 KẾT LUẬN TỪ TRUNG TÂM PHÂN TÍCH (TẠI NGÀY {pd.to_datetime(target_date).strftime('%Y-%m-%d')}):")
        
        # Sửa lỗi logic: Chỉ khen Tốt nếu thực sự có tín hiệu Mua
        if final_signal not in ["Không", "NEUTRAL", "[Lỗi Wyckoff]"]:
            print(f"- [Kỹ thuật]: Tốt. Mã {ticker} ĐÃ có tín hiệu Wyckoff ({final_signal}).")
        else:
            print(f"- [Kỹ thuật]: {final_signal}. Bot bỏ qua vì Cấu trúc giá/Khối lượng không ở Form Mua (Chưa có SOS/SPRING/TEST_CUNG).")

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