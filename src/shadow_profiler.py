import pandas as pd
import numpy as np
import os
from datetime import datetime

class ShadowProfiler:
    def __init__(self, price_df):
        """Khởi tạo Hệ thống Nhận diện Đội lái (Shadow Profiler)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Radar Săn Lái Nội (Shadow Profiler)...")
        self.df_price = price_df
        
        if not self.df_price.empty:
            self.df_price['time'] = pd.to_datetime(self.df_price['time']).dt.normalize()
            self.df_price = self.df_price.sort_values(by=['ticker', 'time'])

    def _detect_upthrusts(self, df):
        """Nhận diện Nến Búa ngược / Upthrust (Kéo xả/Nổ xịt)"""
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        
        # Thân nến và Râu trên
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['range'] = df['high'] - df['low']
        
        # Định nghĩa Nổ xịt: Vol to (>1.5 lần MA20), râu trên dài gấp đôi thân nến
        condition_vol = df['volume'] > (df['vol_ma20'] * 1.5)
        condition_tail = (df['upper_shadow'] > (df['body'] * 2)) & (df['range'] > 0)
        
        df['is_upthrust'] = condition_vol & condition_tail
        return df

    def _filter_shadow_candidates(self, tickers):
        """
        Bộ lọc Thanh trừng: Tự động loại bỏ Bluechips và hàng nặng mông.
        Chỉ giữ lại các mã Midcap/Penny có 'Tiền án tiền sự' bơm xả.
        """
        print(f"[*] Đang rà soát tố chất đầu cơ của {len(tickers)} mã...")
        valid_candidates = []
        
        for ticker in tickers:
            df_t = self.df_price[self.df_price['ticker'] == ticker].tail(250).copy() # Quét 1 năm
            if len(df_t) < 100: continue
            
            # 1. TIÊU CHÍ VỐN HÓA & THANH KHOẢN (Loại bỏ hàng quá to, khó lái)
            # Tính thanh khoản trung bình 20 phiên cuối
            avg_vol = df_t['volume'].tail(20).mean()
            avg_price = df_t['close'].tail(20).mean()
            liquidity_bn = (avg_vol * avg_price) / 1_000_000_000
            
            # Lái nội thường không thích/không đủ tiền lái mã có thanh khoản > 150 tỷ/phiên hoặc giá > 60k
            if liquidity_bn > 150.0 or avg_price > 60000 or liquidity_bn < 1.0:
            # if liquidity_bn > 150.0 or avg_price > 60000:
                continue
                
            # 2. TIÊU CHÍ 'TIỀN ÁN BƠM XẢ' (Historical Pump Factor)
            # Mã này trong 1 năm qua đã từng có nhịp nào tăng thốc > 30% trong vòng 15 ngày chưa?
            df_t['future_15d_max'] = df_t['close'].shift(-15).rolling(15).max()
            df_t['max_pump'] = (df_t['future_15d_max'] - df_t['close']) / df_t['close']
            
            if df_t['max_pump'].max() < 0.30: # Nếu chưa từng bơm thốc 30%, nó là hàng "ngoan" -> Bỏ
                continue
                
            valid_candidates.append(ticker)
            
        print(f"   => [OK] Đã tự động chắt lọc được {len(valid_candidates)} mã thuần Đầu Cơ (Penny/Midcap).")
        return valid_candidates

    def build_criminal_profile(self, tickers, lookback_days=250):
        """
        Pha 1: "Học" hành vi của Đội lái trong quá khứ.
        Đã đồng bộ thuật toán Chống nhiễu (Robust Statistics) và Time-in-Zone.
        """
        print("\n" + "="*85)
        print(" 🕵️ GIAI ĐOẠN 1: TRÍCH XUẤT HỒ SƠ LÁI NỘI (CRIMINAL PROFILING)")
        print("="*85)
        
        profiles = []
        
        for ticker in tickers:
            df_t = self.df_price[self.df_price['ticker'] == ticker].tail(lookback_days).copy()
            if len(df_t) < 50: continue
            
            df_t = df_t.sort_values('time').reset_index(drop=True)
            df_t = self._detect_upthrusts(df_t)
            
            # Tìm kiếm các cú KÉO GIÁ (Pump) > 20% trong 10 ngày
            df_t['future_10d_max'] = df_t['close'].shift(-10).rolling(10).max()
            df_t['pump_yield'] = (df_t['future_10d_max'] - df_t['close']) / df_t['close']
            
            breakouts = df_t[df_t['pump_yield'] >= 0.20].index.tolist()
            
            clean_breakouts = []
            for b in breakouts:
                if not clean_breakouts or b - clean_breakouts[-1] > 20:
                    clean_breakouts.append(b)
                    
            for b_idx in clean_breakouts:
                if b_idx < 25: continue 
                
                # ---------------------------------------------------------
                # 1. ĐO LƯỜNG ĐỘ NÉN (20 Ngày trước điểm nổ)
                # ---------------------------------------------------------
                df_base = df_t.iloc[b_idx-20 : b_idx]
                
                # 🛠️ Dùng Quantile để gọt bỏ 1 phiên xé rào cao nhất và 1 phiên đạp thấp nhất
                max_p = df_base['close'].quantile(0.95)
                min_p = df_base['close'].quantile(0.05)
                base_volatility = (max_p - min_p) / min_p * 100 if min_p > 0 else 0
                
                min_vols = df_base['volume'].nsmallest(3).mean()
                avg_vol_ma20 = df_base['vol_ma20'].mean()
                dry_up_ratio = (min_vols / avg_vol_ma20) * 100 if avg_vol_ma20 > 0 else 100
                
                upthrust_count = df_base['is_upthrust'].sum()
                
                # ---------------------------------------------------------
                # 🚀 2. ĐO LƯỜNG THỜI GIAN GOM HÀNG (TIME-IN-ZONE)
                # ---------------------------------------------------------
                # Lấy 60 ngày trước điểm nổ làm khung thời gian khảo sát
                start_idx = max(0, b_idx - 60)
                df_60 = df_t.iloc[start_idx : b_idx]
                
                if not df_60.empty:
                    median_price = df_60['close'].median()
                    # Vùng gom hàng: +/- 15% quanh giá trung vị
                    upper_band = median_price * 1.15
                    lower_band = median_price * 0.85
                    
                    # Đếm tổng số ngày giá ĐÓNG CỬA nằm trong Vùng này
                    acc_days = len(df_60[(df_60['close'] >= lower_band) & (df_60['close'] <= upper_band)])
                else:
                    acc_days = 0
                        
                # ---------------------------------------------------------
                # 🚀 3. ĐO LƯỜNG QUỸ ĐẠO KÉO GIÁ (MARKUP DURATION)
                # ---------------------------------------------------------
                future_15d = df_t.iloc[b_idx : min(b_idx + 15, len(df_t))]
                if not future_15d.empty:
                    peak_idx = future_15d['high'].idxmax()
                    markup_duration = peak_idx - b_idx
                else:
                    markup_duration = 0
                
                profiles.append({
                    'Ticker': ticker,
                    'Pump_Date': df_t.iloc[b_idx]['time'].strftime('%Y-%m-%d'),
                    'Yield_10D': df_t.iloc[b_idx]['pump_yield'] * 100,
                    'Volatility_20D': base_volatility,
                    'DryUp_Ratio': dry_up_ratio,
                    'Upthrust_Count': upthrust_count,
                    'Accumulation_Days': acc_days,
                    'Markup_Duration': markup_duration
                })

        df_profiles = pd.DataFrame(profiles)
        
        if df_profiles.empty:
            print("[!] Không tìm thấy cú kéo giá >20% nào của các mã này trong lịch sử.")
            return None
            
        # TÍNH TOÁN TRUNG VỊ (MEDIAN) CỦA TOÀN BỘ CHỈ SỐ
        median_volatility = df_profiles['Volatility_20D'].median()
        median_dry_up = df_profiles['DryUp_Ratio'].median()
        median_upthrusts = df_profiles['Upthrust_Count'].median()
        median_acc_days = df_profiles['Accumulation_Days'].median()
        median_markup = df_profiles['Markup_Duration'].median()
        
        print(f"[*] Đã mổ xẻ {len(df_profiles)} siêu sóng đầu cơ. Rút ra BỘ LUẬT CHUẨN như sau:")
        print(f"  1. Thời gian Gom hàng (Tích lũy) : Trung bình mất {median_acc_days:.0f} phiên nén giá.")
        print(f"  2. Biên độ Nền nén               : Dao động quanh {median_volatility:.1f}%")
        print(f"  3. Tỷ lệ Vắt kiệt Cung           : Volume tụt chỉ còn {median_dry_up:.1f}% so với MA20")
        print(f"  4. Sức chịu đựng (Nổ xịt)        : Nhẫn nhịn {median_upthrusts:.0f} lần rũ bỏ.")
        print("-" * 85)
        print(f"  => 🚀 QUỸ ĐẠO KÉO GIÁ (MARKUP): Nhịp đánh sẽ đạt đỉnh và kết thúc sau trung bình {median_markup:.0f} phiên kể từ điểm nổ!")
        print("="*85)
        
        # ÁP DỤNG CÁC HỆ SỐ NỚI LỎNG (CHỐNG OVERFITTING)
        return {
            'max_volatility': median_volatility * 1.5,  # Cho phép biên độ lớn hơn 50% so với trung bình
            'max_dry_up': median_dry_up * 1.5,          # Cạn cung không cần quá khắt khe
            'max_upthrusts': 4,                         # Cho phép Lái rũ tối đa 4 lần
            'min_acc_days': median_acc_days * 0.5,      # Chỉ cần nén được 50% thời gian trung bình là đạt
            'markup_duration': median_markup
        }

    def live_shadow_radar(self, tickers, profile_rules, target_date=None):
        """
        Pha 2: Live Radar. 
        Quét bảng điện hôm nay, kết hợp kiểm tra Thời gian Gom hàng (Độ chín muồi)
        và đưa ra Chiến lược chốt lời (Markup Duration).
        Nếu có target_date, radar sẽ lùi về quá khứ để quét.
        """
        date_label = target_date if target_date else "HÔM NAY"
        print("\n" + "="*95)
        print(f" 🎯 GIAI ĐOẠN 2: RADAR CẢNH BÁO SÓNG ĐẦU CƠ - THỜI ĐIỂM QUÉT: [ {date_label} ]")
        print("="*95)
        
        if not profile_rules: return
        
        alerts = []
        for ticker in tickers:
            df_full = self.df_price[self.df_price['ticker'] == ticker]
            # Cắt đứt tương lai nếu có target_date
            if target_date:
                df_full = df_full[df_full['time'] <= pd.to_datetime(target_date)]

            # Lấy 120 phiên tính từ thời điểm quét lùi về trước
            df_t = df_full.tail(120).copy()
            if len(df_t) < 25: continue
            
            df_t = df_t.sort_values('time').reset_index(drop=True)
            df_t = self._detect_upthrusts(df_t)
            
            # --- 1. ĐO LƯỜNG ĐỘ TÍCH LŨY HIỆN TẠI (TIME-IN-ZONE METHOD) ---
            # Lấy 60 ngày làm khung thời gian khảo sát
            df_60 = df_t.tail(60)
            if not df_60.empty:
                median_price = df_60['close'].median()
                # Định nghĩa "Vùng gom hàng" là dao động +/- 15% quanh giá trung vị
                upper_band = median_price * 1.15
                lower_band = median_price * 0.85
                
                # Đếm tổng số ngày giá nằm trong Vùng này (bỏ qua các ngày bị đạp rũ thủng vùng)
                live_acc_days = len(df_60[(df_60['close'] >= lower_band) & (df_60['close'] <= upper_band)])
            else:
                live_acc_days = 0

            # Lấy 20 phiên gần nhất để tính độ nén ngắn hạn
            df_live = df_t.tail(20)
            
            # Dùng Quantile để gọt bỏ 1 phiên cao nhất và 1 phiên thấp nhất (Chống nhiễu rũ hàng)
            max_p = df_live['close'].quantile(0.95)
            min_p = df_live['close'].quantile(0.05)
            live_volatility = (max_p - min_p) / min_p * 100 if min_p > 0 else 0
            
            recent_5d = df_live.tail(5)
            min_vols = recent_5d['volume'].nsmallest(3).mean()
            current_ma20 = df_live.iloc[-1]['vol_ma20']
            live_dry_up = (min_vols / current_ma20) * 100 if current_ma20 > 0 else 100
            
            live_upthrusts = df_live['is_upthrust'].sum()
            
            # --- X-QUANG DEBUG (CHỈ HIỂN THỊ KHI QUÉT TRÚNG MÃ HRC) ---
            if ticker == 'HRC':
                print(f"\n[🔬 DEBUG HRC - NGÀY {target_date if target_date else 'HIỆN TẠI'}]")
                print(f"  + Độ nén nền (Volat): {live_volatility:.1f}% (Luật cho phép <= {profile_rules['max_volatility']:.1f}%) -> Pass: {live_volatility <= profile_rules['max_volatility']}")
                print(f"  + Độ cạn cung (Dry) : {live_dry_up:.1f}% (Luật cho phép <= {profile_rules['max_dry_up']:.1f}%) -> Pass: {live_dry_up <= profile_rules['max_dry_up']}")
                print(f"  + Số ngày gom (Acc) : {live_acc_days} ngày (Luật yêu cầu >= {profile_rules['min_acc_days']:.0f} ngày) -> Pass: {live_acc_days >= profile_rules['min_acc_days']}")
                print(f"  + Số lần nổ xịt     : {live_upthrusts} lần (Luật cho phép <= {profile_rules['max_upthrusts']} lần)")
            
            # --- 3. CHẨN ĐOÁN SO VỚI BỘ LUẬT ---
            is_coiling = live_volatility <= profile_rules['max_volatility']
            is_dry = live_dry_up <= profile_rules['max_dry_up']
            is_ripe = live_acc_days >= profile_rules['min_acc_days']
            
            if is_coiling and is_dry:
                if live_upthrusts >= profile_rules['max_upthrusts']:
                    status = "🩸 NGUY HIỂM (Cạn kiên nhẫn)"
                    note = f"Đã nổ xịt {live_upthrusts} lần. Đừng mua nền, lái sắp đạp gãy rũ!"
                elif not is_ripe:
                    status = "⏳ CHỜ ĐỢI (Nén chưa đủ)"
                    note = f"Mới gom {live_acc_days}/{profile_rules['min_acc_days']:.0f} phiên. Nổ bây giờ 90% là bẫy sớm!"
                else:
                    status = "🌟 CHÍN MUỒI (Sẵn sàng nổ)"
                    markup_t = profile_rules['markup_duration']
                    note = f"Đã gom {live_acc_days} phiên. Cạn cung {live_dry_up:.1f}%. Nếu nổ -> Mục tiêu chốt T+{markup_t:.0f}"
                    
                alerts.append({
                    'Ticker': ticker,
                    'Status': status,
                    'Note': note
                })
                
        # --- 4. IN BÁO CÁO ---
        if not alerts:
            print("[*] Hiện tại không có mã nào lọt vào form nén sóng của Đội lái.")
        else:
            print(f"{'MÃ CP':<8} | {'TRẠNG THÁI':<26} | {'CHIẾN LƯỢC & HÀNH VI':<60}")
            print("-" * 95)
            for a in alerts:
                print(f"{a['Ticker']:<8} | {a['Status']:<26} | {a['Note']:<60}")
        print("="*95)

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    PRICE_PATH = 'data/parquet/price/master_price.parquet'
    profiler = ShadowProfiler(PRICE_PATH)
    
    # 1. Lấy toàn bộ danh sách mã có trên thị trường (3 ký tự)
    all_tickers = profiler.df_price['ticker'].unique().tolist()
    market_tickers = [t for t in all_tickers if len(str(t)) == 3]
    
    print("\n" + "="*85)
    print(" 🤖 BƯỚC 0: TỰ ĐỘNG LỌC TỆP HUẤN LUYỆN (AUTO-PURGE)")
    print("="*85)
    
    # 2. Chạy qua Trạm Thanh trừng để lấy Tệp Training "Thuần chủng"
    training_tickers = profiler._filter_shadow_candidates(market_tickers)
    
    # Pha 1: Học từ Tệp Training chuẩn xác nhất
    if training_tickers:
        rules = profiler.build_criminal_profile(training_tickers, lookback_days=250)
        
        # Pha 2: Mang bộ luật đi quét lại toàn bộ bảng điện
        if rules:
            profiler.live_shadow_radar(market_tickers, rules)

