import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from config import settings

# --- DANH SÁCH MÙNG 1 TẾT NGUYÊN ĐÁN (2021-2030) ---
# Dùng để tính toán thời điểm "Tháng Chạp"
LUNAR_NEW_YEARS = [
    '2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10', 
    '2025-01-29', '2026-02-17'
]

class WyckoffForecaster:
    def __init__(self, data_input=None, output_dir=None, run_date=datetime.now(), verbose=False, save_file=False):

        if isinstance(data_input, pd.DataFrame):
            self.price_df = data_input.copy()
        elif isinstance(data_input, (str, Path)):
            price_dir = Path(data_input)
            if price_dir.exists():
                self.price_df = pd.read_parquet(price_dir)
            else:
                print(f"[!] WyckoffForecaster Lỗi: Không tìm thấy file tại {price_dir}")
                self.price_df = pd.DataFrame()
        else:
            raise ValueError("[!] WyckoffForecaster Yêu cầu đầu vào phải là DataFrame hoặc Đường dẫn File (Path/str).")

        self.output_dir = Path(output_dir) if output_dir else Path('data/forecast')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Xử lý an toàn nếu settings chưa có LOOKBACK_TR
        try:
            self.lookback = settings.LOOKBACK_TR
        except:
            self.lookback = 20 # Mặc định 20 phiên nếu config lỗi
        self.run_date = pd.to_datetime(run_date)
        self.verbose = verbose
        self.save_file = save_file
        # Đổi mảng string sang datetime một lần duy nhất
        self.lunar_dates = pd.to_datetime(LUNAR_NEW_YEARS)

    def _is_pre_tet_period(self, current_date):
        """Kiểm tra xem ngày hiện tại có nằm trong 30 ngày trước Tết Âm lịch không"""
        # Trừ trực tiếp chuỗi thời gian bằng Vector siêu nhanh
        days_to_tet = (self.lunar_dates - current_date).days
        return any((days_to_tet > 0) & (days_to_tet <= 30))

    def _calculate_indicators(self, df):
        """Bước 1: Phân tích Khối lượng và Cấu trúc Nến (VPA - Volume Price Analysis)"""
        # 1. Đo lường Khối lượng
        df['vol_ma20'] = df['volume'].rolling(window=settings.VOL_MA_PERIOD).mean()
        df['rel_vol'] = df['volume'] / df['vol_ma20']
        
        # Tính Z-Score Khối lượng
        df['vol_std20'] = df['volume'].rolling(window=settings.VOL_MA_PERIOD).std()
        df['vol_std20'] = df['vol_std20'].replace(0, np.nan) # Tránh lỗi chia cho 0
        df['vol_z_score'] = ((df['volume'] - df['vol_ma20']) / df['vol_std20']).fillna(0)

        # 2. X-Quang Cấu trúc Nến (Spread & Close Position)
        df['spread'] = df['high'] - df['low']
        df['avg_spread'] = df['spread'].rolling(window=20).mean() # Biên độ trung bình 20 phiên
        
        # Vị trí đóng cửa (Từ 0.0 đến 1.0)
        # 1.0 = Đóng cửa cao nhất phiên (Cầu mạnh) | 0.0 = Đóng cửa thấp nhất phiên (Cung mạnh)
        # Thêm 0.0001 để tránh lỗi chia cho 0 khi giá tham chiếu/Doji
        df['close_pos'] = (df['close'] - df['low']) / (df['spread'] + 0.0001) 
        
        # 3. Các đường xu hướng
        df['ema34'] = df['close'].ewm(span=settings.EMA_FAST, adjust=False).mean()
        df['ema89'] = df['close'].ewm(span=settings.EMA_SLOW, adjust=False).mean()
        
        # 4. Tính ATR cho Stoploss
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Phân loại Volume
        df['vol_type'] = np.where(df['rel_vol'] > 2.0, 'Ultra High', 
                         np.where(df['rel_vol'] > 1.5, 'High', 
                         np.where(df['rel_vol'] < 0.6, 'Low', 'Normal')))

        return df

    def _detect_signals(self, df):
        """Bước 2: Nhận diện tín hiệu Wyckoff (2 Chiều & Test Cung)"""
        if len(df) < (self.lookback + 2):
            return "NEUTRAL", "Không đủ dữ liệu.", 0, 0

        # Xác định Vùng đi ngang (Trading Range)
        range_df = df.iloc[-(self.lookback + 2) : -2] 
        sup, res = range_df['low'].min(), range_df['high'].max()
        
        curr, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        
        signal = "NEUTRAL"
        note = "Giá đang biến động trong vùng tích lũy."

        is_pre_tet = self._is_pre_tet_period(self.run_date) if hasattr(self, '_is_pre_tet_period') else False
        sos_vol_req = 2.0 if is_pre_tet else 1.2
        
        # Lấy các thông số X-Quang nến hiện tại
        c_close, c_vol, c_rel_vol = curr['close'], curr['volume'], curr['rel_vol']
        c_high, c_low = curr['high'], curr['low']
        c_pos, c_spread, avg_spread = curr['close_pos'], curr['spread'], curr['avg_spread']
        
        # ==========================================
        # 🟢 CHIỀU MUA (ACCUMULATION & MARKUP)
        # ==========================================
        
        # 1. SOS (Dấu hiệu sức mạnh - Breakout uy tín)
        if c_close > res:
            # SOS thật: Nổ Vol + Biên độ nến mở rộng + Đóng cửa ở 1/3 trên cùng (c_pos > 0.65)
            if c_rel_vol > sos_vol_req and c_spread > avg_spread * 0.8 and c_pos > 0.65:
                signal = "SOS"
                note = f"SOS CHUẨN: Bứt phá kháng cự {res:,.0f}. Lực cầu làm chủ (C.Pos: {c_pos:.2f}, Vol: {c_rel_vol:.1f}x)"
            # SOS rác (Bị bán dội ngược rụt đầu)
            elif c_rel_vol > sos_vol_req and c_pos < 0.5:
                signal = "UTAD" 
                note = f"BẪY FALSE-BREAK: Vượt cản nhưng bị bán rụt đầu (C.Pos: {c_pos:.2f}). Rủi ro Bull-trap!"

        # 2. SPRING (Cú rũ bỏ hoảng loạn)
        elif (prev['low'] < sup or c_low < sup) and (c_close >= sup):
            trend_ok = (c_close > curr['ema89']) or (c_close > curr['ema34'] * 0.95)
            # Spring chuẩn: Rút chân mạnh đóng cửa ở nửa trên nến (c_pos > 0.5)
            if c_pos > 0.5 and trend_ok:
                vol_desc = "Cạn cung" if c_rel_vol < 1.0 else "Có lực bắt đáy"
                signal = "SPRING"
                note = f"SPRING: Rút chân bảo vệ hỗ trợ {sup:,.0f} ({vol_desc})."

        # 3. TEST CUNG (No Supply Bar) - Bắt sớm trong nền
        elif (sup * 1.01) <= c_close <= (res * 0.99): # Đang nằm lơ lửng trong nền giá
            # Cây nến giảm nhẹ, biên độ hẹp, volume cạn kiệt (nhỏ hơn 2 phiên trước)
            if c_close < prev['close'] and c_spread < avg_spread * 0.8:
                if c_vol < prev['volume'] and c_vol < prev2['volume'] and c_rel_vol < 0.8:
                    if c_pos > 0.4: # Không đóng cửa quá thấp (Có cầu đỡ)
                        signal = "TEST_CUNG"
                        note = f"TEST CUNG (No Supply): Lực bán cạn kiệt. Dấu hiệu Lái sắp kéo giá."

        # ==========================================
        # 🔴 CHIỀU BÁN (DISTRIBUTION & MARKDOWN)
        # ==========================================
        
        # 4. UTAD (Bẫy Tăng Giá / Upthrust)
        if signal == "NEUTRAL" and (prev['high'] > res or c_high > res) and (c_close <= res):
            # Nến mọc tóc: Đóng cửa ở 1/3 dưới cùng (c_pos < 0.35)
            if c_rel_vol > 1.0 and c_pos < 0.35: 
                signal = "UTAD"
                note = f"UTAD (Bẫy Bò): Kéo vượt {res:,.0f} nhưng xả rụt đầu (Vol {c_rel_vol:.1f}x). Xác nhận Phân phối."

        # 5. SOW (Dấu hiệu suy yếu / Gãy nền)
        elif signal == "NEUTRAL" and c_close < sup:
            # Gãy nền uy tín: Biên độ lớn, đóng cửa thấp nhất phiên (c_pos < 0.3)
            if c_rel_vol > 1.2 and c_pos < 0.3:
                signal = "SOW"
                note = f"SOW CHUẨN: Gãy hỗ trợ {sup:,.0f}. Lực bán quyết liệt (C.Pos: {c_pos:.2f}, Vol: {c_rel_vol:.1f}x)"
            elif c_rel_vol <= 1.2:
                signal = "MARKDOWN"
                note = f"MARKDOWN: Cưa chân bàn thủng hỗ trợ do cạn Cầu."

        return signal, note, sup, res

    def run_forecast(self):
        results = []
        if self.verbose:
            print(f"--- Bắt đầu dự báo Wyckoff: {self.run_date.strftime('%d/%m/%Y')} ---")

        # if not self.price_dir.exists():
        #     print(f"[!] Không tìm thấy dữ liệu tại: {self.price_dir}")
        #     return pd.DataFrame()

        # Đọc toàn bộ dữ liệu thị trường chỉ với 1 dòng lệnh
        if self.verbose:
            print("Đang tải dữ liệu Parquet...")
        # df_master = pd.read_parquet(self.price_dir)
        df_master = self.price_df
        df_master['time'] = pd.to_datetime(df_master['time']) # Đảm bảo time là datetime
        
        # Lọc dữ liệu tới ngày run_date (Phục vụ Backtest)
        df_master = df_master[df_master['time'] <= self.run_date]

        # Group by mã cổ phiếu để xử lý
        for ticker, df_ticker in df_master.groupby('ticker'):
            try:
                # Sắp xếp lại thời gian cho chuẩn và Chỉ lấy 250 phiên cuối cùng (1 năm)
                df = df_ticker.sort_values('time').tail(250).reset_index(drop=True)

                # Bỏ qua nếu mã mới lên sàn chưa đủ data
                if len(df) < self.lookback + 5: 
                    continue
                
                df = self._calculate_indicators(df)
                signal, note, sup, res = self._detect_signals(df)
                curr = df.iloc[-1]
                high_52w = df['high'].max()
                last_price = curr['close']
                dist_to_sup = ((last_price - sup) / sup) * 100 if sup > 0 else 0

                # Tính khoảng cách tới đỉnh 52 tuần (%)
                dist_to_52w_high = ((last_price - high_52w) / high_52w) * 100

                results.append({
                    'Ticker': ticker,
                    'Price': last_price,
                    'High': curr['high'],                                           # Giá cao nhất phiên
                    'Low': curr['low'],                                             # Giá thấp nhất phiên
                    'Volume': curr['volume'],                                       # Khối lượng phiên
                    'Spread': curr['spread'],
                    'ATR': round(curr['atr'], 2) if not pd.isna(curr['atr']) else 0,# Lưu ATR để Trading dùng
                    'EMA34': round(curr['ema34'], 2),
                    'EMA89': round(curr['ema89'], 2),
                    'Signal': signal,
                    'Support': sup,
                    'Resistance': res,
                    'Dist_to_Support_%': round(dist_to_sup, 4),
                    'Dist_to_52W_High_%': round(dist_to_52w_high, 4),
                    'VPA_Status': curr.get('vol_type', 'Normal'),
                    'Vol_Z_Score': round(curr.get('vol_z_score', 0), 2),
                    'Analysis': note
                })
            except Exception as e:
                continue

        report_df = pd.DataFrame(results)
        if not report_df.empty and getattr(self, 'save_file', False):
            # Xếp hạng ưu tiên
            priority_map = {
                'SPRING': 1,      # Ưu tiên 1: Bắt đáy hoảng loạn (Rủi ro/Lợi nhuận tốt nhất)
                'TEST_CUNG': 2,   # Ưu tiên 2: Mua thăm dò tĩnh lặng trong nền (An toàn nhất)
                'SOS': 3,         # Ưu tiên 3: Mua gia tăng khi bứt phá xác nhận
                'UTAD': 4,        # Ưu tiên 4: Cảnh báo bẫy tăng giá (Canh Bán)
                'SOW': 5,         # Ưu tiên 5: Gãy nền uy tín (Bán cắt lỗ/Chốt lời ngay)
                'MARKDOWN': 6,    # Ưu tiên 6: Cổ phiếu trôi đáy, bò tùng xẻo (Bỏ qua/Bán)
                'NEUTRAL': 7      # Ưu tiên 7: Đi ngang nhàm chán (Không quan tâm)
            }
            report_df['priority'] = report_df['Signal'].map(priority_map)
            # Điền số 8 cho bất kỳ tín hiệu rác nào lọt qua (nếu có) để không bị lỗi NaN
            report_df['priority'] = report_df['priority'].fillna(8)
            report_df = report_df.sort_values('priority').drop(columns=['priority'])
            output_path = self.output_dir / 'wyckoff_forecast.csv'
            report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            if self.verbose:
                print(f"[OK] Đã phân tích xong {len(report_df)} mã. Kết quả lưu tại: {output_path}")
        
        return report_df

if __name__ == "__main__":
    forecaster = WyckoffForecaster()
    report = forecaster.run_forecast()