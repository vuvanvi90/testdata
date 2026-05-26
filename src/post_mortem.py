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
        self.price_path = self.data_dir / 'price/master_price_l2.parquet'
        self.foreign_path = self.data_dir / 'macro/foreign_flow.parquet'
        self.prop_path = self.data_dir / 'macro/prop_flow.parquet'
        self.comp_path = self.data_dir / 'company/master_company.parquet'
        self.intra_path = self.data_dir / 'intraday/master_intraday.parquet'
        
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
                if not df_price.empty and 'matched_volume' in df_price.columns and 'volume' not in df_price.columns:
                    df_price = df_price.rename(columns={'matched_volume': 'volume'})

            self.price_dict = self._filterd_dict(df=df_price)
            prop_dict = self._load_data_dict(self.prop_path)

            # Nạp dữ liệu Intraday cho Post-Mortem
            df_intra = self._load_parquet_safe(self.intra_path)
            self.intra_dict = {}
            if not df_intra.empty and 'ticker' in df_intra.columns:
                for ticker, group in df_intra.groupby('ticker'):
                    self.intra_dict[ticker] = group

            out_shares_dict = {}
            df_comp = self._load_parquet_safe(self.comp_path)
            # 1. BASE LEVEL: Lấy từ master_company (Làm nền dự phòng)
            if not df_comp.empty and 'ticker' in df_comp.columns and 'issue_share' in df_comp.columns:
                out_shares_dict = df_comp.set_index('ticker')['issue_share'].to_dict()

             # 2. Ghi đè bằng Dữ liệu Tươi (Live Data) từ master_price_l2
            if not df_price.empty and 'total_shares' in df_price.columns:
                df_price_l2_raw = df_price.copy()
                # Lọc bỏ các dòng bị rỗng
                df_l2_shares = df_price_l2_raw.dropna(subset=['total_shares'])
                if not df_l2_shares.empty:
                    # Lấy record ngày mới nhất của từng mã cổ phiếu
                    latest_shares = df_l2_shares.sort_values('time').groupby('ticker').tail(1)
                    l2_shares_dict = latest_shares.set_index('ticker')['total_shares'].to_dict()
                    # Cập nhật (ghi đè) vào dictionary tổng
                    out_shares_dict.update(l2_shares_dict)
            
            # Khởi tạo các Động cơ Phân tích
            self.sm_engine = SmartMoneyEngine(
                price_l2_dict=self.price_dict, 
                prop_dict=prop_dict, 
                out_shares_dict=out_shares_dict, 
                universe=self.universe
            )
            self.profiler = ShadowProfiler(df_l2=df_price, verbose=False)
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

    def _filterd_dict(self, df):
        if df.empty or 'ticker' not in df.columns:
            return {}
            
        d_dict = {}
        for ticker, group in df.groupby('ticker'):
            group = group.sort_values('time').tail(130)
            d_dict[ticker] = group
            
        return d_dict

    def _prepare_latest_ohlcv(self, ticker, df_p):
        """Đồng bộ dữ liệu: Nặn dòng nến T0 từ dữ liệu khớp lệnh Intraday thực tế"""
        if df_p.empty: return df_p
        df_p = df_p.copy()
        df_p['time'] = pd.to_datetime(df_p['time']).dt.normalize()
        
        df_intra = self.intra_dict.get(ticker, pd.DataFrame())
        if not df_intra.empty:
            df_i = df_intra.copy()
            df_i['time'] = pd.to_datetime(df_i['time'])
            # latest_date = df_i['time'].dt.date.max()
            latest_date = pd.Timestamp.now().date() # chỉ lấy phiên trong ngày
            df_today = df_i[df_i['time'].dt.date == latest_date].sort_values('time')
            
            if not df_today.empty:
                # Ép kiểu an toàn thể tích khối lượng khớp lệnh
                today_candle = pd.DataFrame([{
                    'ticker': ticker,
                    'time': pd.to_datetime(latest_date),
                    'open': float(df_today['price'].iloc[0]),
                    'high': float(df_today['price'].max()),
                    'low': float(df_today['price'].min()),
                    'close': float(df_today['price'].iloc[-1]),
                    'volume': float(df_today['volume'].sum())
                }])
                
                # Đồng bộ cấu trúc cột phòng trường hợp df_p chứa nhiều chỉ báo cấu trúc khác
                for col in df_p.columns:
                    if col not in today_candle.columns:
                        today_candle[col] = None
                        
                today_candle = today_candle[df_p.columns]
                
                # Gộp cây nến T0 sống vào đáy của dữ liệu quá khứ tĩnh
                df_combined = pd.concat([df_p, today_candle], ignore_index=True)
                df_combined = df_combined.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                return df_combined
        return df_p

    def analyze(self, ticker, target_date_str, lookback_days=15):
        print(f"\n[🔬] KHÁM NGHIỆM MÃ: {ticker} | NGÀY SỰ KIỆN: {target_date_str} | QUÉT LÙI: {lookback_days} PHIÊN")
        print("-" * 65)

        df_p = self.price_dict.get(ticker)
        if df_p is None or df_p.empty:
            print(f"[!] Không tìm thấy dữ liệu giá cho mã {ticker}.")
            return

        # Hòa mạng dữ liệu Intraday sống vào dòng thời gian trước khi lọc
        df_p = self._prepare_latest_ohlcv(ticker, df_p)

        target_date = pd.to_datetime(target_date_str).normalize()
        df_p_valid = df_p[df_p['time'] <= target_date].copy() # Copy để tránh SettingWithCopyWarning
        
        if df_p_valid.empty:
            print(f"[!] Mã {ticker} chưa niêm yết hoặc không có dữ liệu trước ngày {target_date_str}.")
            return

        # 🚀 TÍNH TOÁN HIỆU SUẤT QUÁ KHỨ (% PnL / Momentum)
        # Tính toán trên toàn bộ df_p_valid trước để đạt tốc độ O(1)
        df_p_valid['pct_1d'] = df_p_valid['close'].pct_change(1) * 100
        df_p_valid['pct_5d'] = df_p_valid['close'].pct_change(5) * 100
        df_p_valid['pct_10d'] = df_p_valid['close'].pct_change(10) * 100
        df_p_valid['pct_15d'] = df_p_valid['close'].pct_change(15) * 100
        df_p_valid['pct_20d'] = df_p_valid['close'].pct_change(20) * 100

        trading_dates = df_p_valid['time'].sort_values().unique()
        recent_dates = trading_dates[-lookback_days:]

        # Căn chỉnh lại Header mở rộng
        print(f"{'NGÀY (T-X)':<10} | {'C-H-L (ĐÓNG/CAO/THẤP)':<23} | {'KHỐI LƯỢNG':<11} | {'%1D':>6} | {'%5D':>6} | {'%20D':>6} | {'WYCKOFF':<9} | {'DÒNG TIỀN & LÁI NỘI'}")
        print("-" * 65)

        sm_result = {}
        final_signal = "Không"

        for date_obj in recent_dates:
            date_str = pd.to_datetime(date_obj).strftime('%Y-%m-%d')
            is_target_day = (date_obj == target_date)
            date_label = f"{date_str}" if is_target_day else f"{date_str}"
            
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
            pct_10d_str = fmt_pct(curr_row['pct_10d'])
            pct_15d_str = fmt_pct(curr_row['pct_15d'])
            pct_20d_str = fmt_pct(curr_row['pct_20d'])
            
            price_str = f"{close_p:,.0f}/{high_p:,.0f}/{low_p:,.0f}"
            
            df_forecast = None
            signal = "Không"
            try:
                df_hist_cut = df_p_valid[df_p_valid['time'] <= date_obj].copy()
                forecaster = WyckoffForecaster(data_input=df_hist_cut, output_dir=None, run_date=date_obj)
                df_forecast = forecaster.run_forecast()
            except Exception as e:
                signal = "[Lỗi WyckoffForecaster]"
                print(f"   [!] Gợi ý gỡ lỗi Forecaster tại ngày {date_str}: {e}")
                traceback.print_exc()
            
            if df_forecast is not None and isinstance(df_forecast, pd.DataFrame) and not df_forecast.empty:
                try:
                    last_row = df_forecast.iloc[-1]
                    close_price = last_row.get('Price', 0)
                    signal = last_row.get('Signal', 'Không')
                except Exception as e:
                    print(f"   [!] Gợi ý gỡ lỗi Forecaster tại ngày {date_str}: {e}")
                    pass
            
            # Cập nhật kết luận cho ngày cuối cùng
            if is_target_day: final_signal = signal

            # =================================================================
            # SMART MONEY & SHADOW PROFILER
            # =================================================================
            try:
                sm_result = self.sm_engine.analyze_ticker(ticker, target_date=date_str)
            except Exception as e:
                print(f"[!] Lỗi analyze_ticker: {e}")
                sm_result = {}

            # 1. GỌI SHADOW PROFILER TRƯỚC ĐỂ LẤY TRẠNG THÁI LÁI NỘI
            shadow_status = ""
            try:
                rules = self.profiler.build_criminal_profile([ticker], lookback_days=125)
                alerts = self.profiler.live_shadow_radar([ticker], rules, target_date=date_str)
                if isinstance(alerts, pd.DataFrame):
                    if not alerts.empty:
                        shadow_status = alerts.iloc[0].get('Status', '')
                elif isinstance(alerts, list):
                    if len(alerts) > 0:
                        alert_item = alerts[0]
                        if isinstance(alert_item, dict):
                            shadow_status = alert_item.get('Status', '')
                        else:
                            shadow_status = str(alert_item)
            except Exception as e:
                shadow_status = f"[Lỗi Radar - {e}]"

            # 2. TRÍCH XUẤT DỮ LIỆU SMART MONEY
            sm_score = sm_result.get('total_sm_score', 0)
            sm_danger = sm_result.get('is_danger', False)
            sm_warnings = sm_result.get('warnings', [])
            sm_details = sm_result.get('sm_details', [])
            
            if isinstance(sm_warnings, str):
                sm_warnings = [sm_warnings] if sm_warnings else []

            # 3. SHADOW OVERRIDE (Miễn trừ bằng Lái Nội)
            is_shadow_override = False
            if shadow_status and "CHÍN MUỒI" in shadow_status:
                is_shadow_override = True
                if sm_danger:
                    sm_danger = False
                    sm_warnings = []
                    # Ghi đè vào sm_result để hàm _print_conclusion nhận diện sự ân xá
                    sm_result['is_danger'] = False 
                    sm_result['warnings'] = []

            # 4. ACTIVE DEMAND OVERRIDE (Miễn trừ bằng Cầu Đỡ)
            active_override_msg = ""
            for detail in sm_details:
                if "🛡️ ACTIVE OVERRIDE" in detail:
                    active_override_msg = detail
                    sm_danger = False
                    sm_warnings = []
                    # Ghi đè vào sm_result để hàm _print_conclusion nhận diện sự ân xá
                    sm_result['is_danger'] = False 
                    sm_result['warnings'] = []
                    break

            # 5. FORMAT CHUỖI IN RA BẢNG
            sm_str = ""
            if sm_danger and sm_warnings:
                sm_str = f"🚨 BÁO ĐỘNG ĐỎ: {' | '.join(sm_warnings)}"
            else:
                # Nếu không có nguy hiểm, ưu tiên hiển thị Cờ Miễn Trừ
                if is_shadow_override and active_override_msg:
                    sm_str = f"🛡️ SHADOW OVERRIDE | {active_override_msg}"
                elif is_shadow_override:
                    sm_str = "🛡️ SHADOW OVERRIDE: Lái Nội cân Tây"
                elif active_override_msg:
                    sm_str = active_override_msg
                else:
                    sm_str = f"SM: ({sm_score})" if sm_score != 0 else "Bình thường"

            # Đính kèm trạng thái Lái nội (Nếu không nằm trong trạng thái Chín Muồi Override)
            if shadow_status and "CHÍN MUỒI" not in shadow_status:
                sm_str += f" | Lái nội: {shadow_status}"

            flow_info = sm_str.strip()
            if not flow_info: flow_info = "Bình thường"

            # In format mở rộng
            print(f"{date_label:<10} | {price_str:<23} | {vol:11,.0f} | {pct_1d_str} | {pct_5d_str} | {pct_20d_str} | {signal:<9} | {flow_info}")

        print("=" * 65)
        self._print_conclusion(ticker, recent_dates[-1], sm_result, final_signal)


    def _print_conclusion(self, ticker, target_date, final_sm_result, final_signal):
        print(f"\n🧠 KẾT LUẬN TỪ TRUNG TÂM PHÂN TÍCH (TẠI NGÀY {pd.to_datetime(target_date).strftime('%Y-%m-%d')}):")
        
        # 1. KẾT LUẬN KỸ THUẬT
        if final_signal not in ["Không", "NEUTRAL", "[Lỗi Wyckoff]"]:
            print(f"- [Kỹ thuật]: Tốt. Mã {ticker} ĐÃ có tín hiệu Wyckoff ({final_signal}).")
        else:
            print(f"- [Kỹ thuật]: {final_signal}. Bot bỏ qua vì Cấu trúc giá/Khối lượng không ở Form Mua (Chưa có SOS/SPRING/TEST_CUNG).")

        # 2. KẾT LUẬN DÒNG TIỀN (ĐÃ ĐƯỢC ÂN XÁ NẾU CÓ OVERRIDE)
        if final_sm_result.get('is_danger', False):
            warnings = final_sm_result.get('warnings', [])
            warn_str = ', '.join(warnings) if isinstance(warnings, list) else str(warnings)
            print(f"- [Dòng tiền]: BÁO ĐỘNG ĐỎ. Bot bị chặn mua do Tây/Tự doanh xả rát. Lý do: {warn_str}.")
        elif final_sm_result.get('total_sm_score', 0) < 0:
             print(f"- [Dòng tiền]: Xấu (Điểm âm). Bot hạ tỷ trọng hoặc từ chối vì thiếu sự ủng hộ của tay to.")
        else:
             print(f"- [Dòng tiền]: 🟢 ĐỒNG THUẬN. Dòng tiền an toàn, đã vượt qua các lớp kiểm duyệt/phủ quyết.")
             
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