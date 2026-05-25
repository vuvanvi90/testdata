import pandas as pd
import numpy as np
import os
from datetime import datetime

class ShadowProfiler:
    def __init__(self, df_l2, df_prop=None, verbose=True):
        """Khởi tạo Hệ thống Nhận diện Đội lái (Shadow Profiler V3.0)"""
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Radar Săn Lái & Thâu tóm ngầm (Shadow Profiler V3)...")

        self.verbose = verbose
        
        # 1. NẠP VÀ CHUẨN HÓA L2 DATA
        if df_l2 is not None and not df_l2.empty:
            # Lọc các trường cần thiết cho cả Kỹ thuật và Phân tách Thỏa thuận
            filterd_cols = [
                'time', 'open', 'high', 'low', 'close', 'volume', 'matched_volume', 'ticker',
                'deal_value', 'deal_volume', 'fr_buy_value_deal', 'fr_sell_value_deal', 
                'fr_buy_volume_deal', 'fr_sell_volume_deal', 'fr_available_percentage'
            ]
            self.df_price = df_l2[[c for c in filterd_cols if c in df_l2.columns]].copy()
        else: 
            self.df_price = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ticker'])

        # Ghi đè Matched Volume cho Phân tích Hành vi Giá
        if not self.df_price.empty:
            if 'matched_volume' in self.df_price.columns and 'volume' in self.df_price.columns:
                self.df_price = self.df_price.drop(columns=['volume'])
                self.df_price = self.df_price.rename(columns={'matched_volume': 'volume'})
            elif 'matched_volume' in self.df_price.columns and 'volume' not in self.df_price.columns:
                self.df_price = self.df_price.rename(columns={'matched_volume': 'volume'})

        if not self.df_price.empty and 'time' in self.df_price.columns:
            self.df_price['time'] = pd.to_datetime(self.df_price['time']).dt.normalize()
            if getattr(self.df_price['time'].dt, 'tz', None) is not None:
                self.df_price['time'] = self.df_price['time'].dt.tz_localize(None)
            self.df_price = self.df_price.sort_values(by=['ticker', 'time'])

        # 2. NẠP DỮ LIỆU TỰ DOANH (Nếu có)
        self.df_prop = pd.DataFrame()
        if df_prop is not None and not df_prop.empty:
            self.df_prop = df_prop.copy()
            if 'time' in self.df_prop.columns:
                self.df_prop['time'] = pd.to_datetime(self.df_prop['time']).dt.normalize()
                if getattr(self.df_prop['time'].dt, 'tz', None) is not None:
                    self.df_prop['time'] = self.df_prop['time'].dt.tz_localize(None)

        # 3. KÍCH HOẠT ĐỘNG CƠ PHÂN TÁCH GIAO DỊCH NGẦM
        self.shadow_matrix = pd.DataFrame()
        self._build_shadow_matrix()

    def _build_shadow_matrix(self):
        """Phương trình Kế toán Kép bóc tách Shadow Buy và Shadow Sell"""
        if self.df_price.empty: return
        
        self.shadow_matrix = self.df_price.copy()
        
        # Merge dữ liệu Deal của Tự doanh vào Ma trận
        if not self.df_prop.empty:
            prop_cols = [c for c in self.df_prop.columns if 'deal' in c or c in ['ticker', 'time']]
            if len(prop_cols) > 2:
                self.shadow_matrix = pd.merge(self.shadow_matrix, self.df_prop[prop_cols], on=['ticker', 'time'], how='left')
        
        self.shadow_matrix.fillna(0, inplace=True)
        
        # Hàm lấy cột an toàn hỗ trợ đa định dạng tên
        def get_col(df, names):
            for n in names:
                if n in df.columns: return df[n]
            return pd.Series(0, index=df.index)

        # Trích xuất dữ liệu gốc
        deal_val = get_col(self.shadow_matrix, ['deal_value'])
        deal_vol = get_col(self.shadow_matrix, ['deal_volume'])
        
        f_buy_deal_val = get_col(self.shadow_matrix, ['fr_buy_value_deal'])
        f_sell_deal_val = get_col(self.shadow_matrix, ['fr_sell_value_deal'])
        
        # Tự doanh có thể dùng tên từ JSON gốc hoặc tên đã gọt của Collector
        p_buy_deal_val = get_col(self.shadow_matrix, ['prop_buy_val_deal', 'total_deal_buy_trade_value'])
        p_sell_deal_val = get_col(self.shadow_matrix, ['prop_sell_val_deal', 'total_deal_sell_trade_value'])
        
        # 🚀 PHƯƠNG TRÌNH KHẤU TRỪ: ĐỊNH DANH Ý ĐỒ LÁI NỘI
        self.shadow_matrix['shadow_buy_val'] = deal_val - f_buy_deal_val - p_buy_deal_val
        self.shadow_matrix['shadow_sell_val'] = deal_val - f_sell_deal_val - p_sell_deal_val
        
        for col in ['shadow_buy_val', 'shadow_sell_val']:
            self.shadow_matrix.loc[self.shadow_matrix[col] < 0, col] = 0 # Khử sai số

        # 💎 RADAR THÂU TÓM PREMIUM (TÂY CẠN ROOM)
        self.shadow_matrix['is_premium_deal'] = False
        
        if 'fr_available_percentage' in self.shadow_matrix.columns:
            f_buy_deal_vol = get_col(self.shadow_matrix, ['fr_buy_volume_deal'])
            
            # Điều kiện: Room < 5%, Ngoại Mua Thỏa Thuận, và Giá Deal > Giá Đóng Cửa 5%
            cond_room = self.shadow_matrix['fr_available_percentage'] < 0.05
            cond_deal = f_buy_deal_vol > 0
            
            # np.divide có xử lý chia cho 0
            deal_price = np.where(f_buy_deal_vol > 0, f_buy_deal_val / f_buy_deal_vol.replace(0, 1), 0)
            cond_premium = deal_price > (self.shadow_matrix['close'] * 1.05)
            
            self.shadow_matrix['is_premium_deal'] = cond_room & cond_deal & cond_premium

    def scan_dark_pool_deals(self, tickers, lookback_days=15):
        """Quét bảng điện để tìm các thương vụ Thâu tóm ngầm & Sang tay Mờ ám"""
        if self.verbose:
            print("\n" + "="*95)
            print(f" 🥷 KHỞI ĐỘNG RADAR ĐỌC VỊ DARK POOL (15 PHIÊN GẦN NHẤT)")
            print("="*95)
            
        alerts = []
        if self.shadow_matrix.empty: return alerts
        
        for ticker in tickers:
            df_t = self.shadow_matrix[self.shadow_matrix['ticker'] == ticker].tail(lookback_days)
            if df_t.empty: continue
            
            # 1. BẮT BẠCH TUỘC: THÂU TÓM PREMIUM
            premium_deals = df_t[df_t['is_premium_deal']]
            if not premium_deals.empty:
                total_premium_val = premium_deals['fr_buy_value_deal'].sum() / 1_000_000_000
                alerts.append({
                    'Ticker': ticker,
                    'Type': '💎 PREMIUM DEAL',
                    'Note': f"Tây cạn room, múc thỏa thuận {total_premium_val:.1f} Tỷ với giá đắt hơn sàn >5%!"
                })
                
            # 2. BẮT LÁI NỘI GOM/XẢ NGẦM
            total_s_buy = df_t['shadow_buy_val'].sum() / 1_000_000_000
            total_s_sell = df_t['shadow_sell_val'].sum() / 1_000_000_000
            
            # Tiêu chí: Deal đủ lớn (> 50 Tỷ) và Bất đối xứng (Mua áp đảo Bán hoặc ngược lại)
            if total_s_buy > 50 and total_s_buy > (total_s_sell * 2):
                alerts.append({
                    'Ticker': ticker,
                    'Type': '🥷 LÁI GOM NGẦM',
                    'Note': f"Lái nội mua gom thỏa thuận {total_s_buy:.1f} Tỷ (Áp đảo chiều Bán)."
                })
            elif total_s_sell > 50 and total_s_sell > (total_s_buy * 2):
                alerts.append({
                    'Ticker': ticker,
                    'Type': '🩸 LÁI XẢ NGẦM',
                    'Note': f"Lái nội xả thỏa thuận {total_s_sell:.1f} Tỷ trao kho cho F0/Tổ chức khác."
                })
                
        # In báo cáo
        if self.verbose:
            if not alerts:
                print("[*] Không phát hiện Giao dịch ngầm đáng ngờ nào.")
            else:
                print(f"{'MÃ CP':<8} | {'LOẠI HÌNH ĐỘT BIẾN':<20} | {'CHI TIẾT Ý ĐỒ':<60}")
                print("-" * 95)
                for a in alerts:
                    print(f"{a['Ticker']:<8} | {a['Type']:<20} | {a['Note']:<60}")
        return alerts

    def _detect_upthrusts(self, df):
        """Nhận diện Nến Búa ngược / Upthrust (Kéo xả/Nổ xịt)"""
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        
        # Thân nến và Râu trên
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['range'] = df['high'] - df['low']
        
        # (VPA): Định nghĩa Upthrust Uy tín
        # Vol to (>1.2 lần MA20), Râu trên chiếm >50% nến, Thân nến hẹp (<30% nến)
        condition_vol = df['volume'] > (df['vol_ma20'] * 1.2)
        condition_tail = (df['upper_shadow'] > (df['range'] * 0.5)) & (df['body'] < (df['range'] * 0.3)) & (df['range'] > 0)
        
        df['is_upthrust'] = condition_vol & condition_tail
        return df

    def _filter_shadow_candidates(self, tickers):
        """
        Bộ lọc Thanh trừng: Tự động loại bỏ Bluechips và hàng nặng mông.
        Chỉ giữ lại các mã Midcap/Penny có 'Tiền án tiền sự' bơm xả.
        """
        if self.verbose:
            print(f"[*] Đang rà soát tố chất đầu cơ của {len(tickers)} mã...")
        valid_candidates = []
        
        for ticker in tickers:
            df_t = self.df_price[self.df_price['ticker'] == ticker].tail(130).copy() # Quét nửa năm
            if len(df_t) < 100: continue
            
            # 1. TIÊU CHÍ VỐN HÓA & THANH KHOẢN (Loại bỏ hàng quá to, khó lái)
            # Tính thanh khoản trung bình 20 phiên cuối
            avg_vol = df_t['volume'].tail(20).mean()
            avg_price = df_t['close'].tail(20).mean()
            liquidity_bn = (avg_vol * avg_price) / 1_000_000_000
            
            # Lái nội thường không thích/không đủ tiền lái mã có thanh khoản > 150 tỷ/phiên hoặc giá > 60k
            if liquidity_bn > 150.0 or avg_price > 60000 or liquidity_bn < 15.0:
                continue
                
            # 2. TIÊU CHÍ 'TIỀN ÁN BƠM XẢ' (Historical Pump Factor)
            # Mã này trong 1 năm qua đã từng có nhịp nào tăng thốc > 30% trong vòng 15 ngày chưa?
            # (Pandas Rolling): Tính chính xác max 15 ngày tương lai
            df_t['future_15d_max'] = df_t['close'].rolling(window=15, min_periods=1).max().shift(-14)
            df_t['max_pump'] = (df_t['future_15d_max'] - df_t['close']) / df_t['close']
            
            if df_t['max_pump'].max() < 0.30: # Nếu chưa từng bơm thốc 30%, nó là hàng "ngoan" -> Bỏ
                continue
                
            valid_candidates.append(ticker)
            
        if self.verbose:
            print(f"   => [OK] Đã tự động chắt lọc được {len(valid_candidates)} mã thuần Đầu Cơ (Penny/Midcap).")
        return valid_candidates

    def build_criminal_profile(self, tickers, lookback_days=130):
        """
        Pha 1: "Học" hành vi của Đội lái trong quá khứ.
        Đã đồng bộ thuật toán Chống nhiễu (Robust Statistics) và Time-in-Zone.
        """
        if self.verbose:
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
            df_t['future_10d_max'] = df_t['close'].rolling(window=10, min_periods=1).max().shift(-9)
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
                    markup_duration = future_15d['high'].idxmax() - b_idx
                else: markup_duration = 0
                
                profiles.append({
                    'Volatility_20D': base_volatility, 'DryUp_Ratio': dry_up_ratio,
                    'Upthrust_Count': upthrust_count, 'Accumulation_Days': acc_days, 'Markup_Duration': markup_duration
                })

        df_profiles = pd.DataFrame(profiles)
        if df_profiles.empty: return None
            
        return {
            'max_volatility': df_profiles['Volatility_20D'].median() * 1.5,  
            'max_dry_up': df_profiles['DryUp_Ratio'].median() * 1.5,          
            'max_upthrusts': 4,                         
            'min_acc_days': df_profiles['Accumulation_Days'].median() * 0.5,      
            'markup_duration': df_profiles['Markup_Duration'].median()
        }

    def live_shadow_radar(self, tickers, profile_rules, target_date=None):
        """
        Pha 2: Live Radar. 
        Quét bảng điện hôm nay, kết hợp kiểm tra Thời gian Gom hàng (Độ chín muồi)
        và đưa ra Chiến lược chốt lời (Markup Duration).
        Nếu có target_date, radar sẽ lùi về quá khứ để quét.
        """
        if self.verbose:
            date_label = target_date if target_date else "HÔM NAY"
            print("\n" + "="*95)
            print(f" 🎯 GIAI ĐOẠN 3: RADAR CẢNH BÁO SÓNG ĐẦU CƠ - WYCKOFF NÉN NỀN")
            print("="*95)
        
        try:
            alerts = []
            if not profile_rules: return alerts
            
            for ticker in tickers:
                df_full = self.df_price[self.df_price['ticker'] == ticker]
                # Cắt đứt tương lai nếu có target_date
                if target_date:
                    df_full = df_full[df_full['time'] <= pd.to_datetime(target_date)]

                # Lấy 120 phiên tính từ thời điểm quét lùi về trước
                df_t = df_full.tail(130).copy()
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
                
                is_coiling = live_volatility <= profile_rules['max_volatility']
                is_dry = live_dry_up <= profile_rules['max_dry_up']
                is_ripe = live_acc_days >= profile_rules['min_acc_days']
                if is_coiling and is_dry:
                    if live_upthrusts >= profile_rules['max_upthrusts']:
                        status = "🩸 NGUY HIỂM"
                        note = f"Nổ xịt {live_upthrusts} lần. Đừng mua nền, rủi ro gãy rũ cao!"
                    elif not is_ripe:
                        status = "⏳ CHỜ ĐỢI"
                        note = f"Mới gom {live_acc_days}/{profile_rules['min_acc_days']:.0f} phiên. Nổ sớm dễ gặp bẫy!"
                    else:
                        status = "🌟 CHÍN MUỒI"
                        note = f"Đã gom {live_acc_days} phiên. Cạn cung {live_dry_up:.1f}%. Sẵn sàng chờ nổ!"
                        
                    alerts.append({
                        'Ticker': ticker,
                        'Status': status,
                        'Note': note
                    })
                    
            # --- 4. IN BÁO CÁO ---
            if self.verbose:
                if not alerts:
                    print("[*] Hiện tại không có mã nào lọt vào form nén sóng của Đội lái.")
                else:
                    print(f"{'MÃ CP':<8} | {'TRẠNG THÁI':<15} | {'CHIẾN LƯỢC WYCKOFF':<60}")
                    print("-" * 95)
                    for a in alerts:
                        print(f"{a['Ticker']:<8} | {a['Status']:<15} | {a['Note']:<60}")
            return alerts
        except Exception as e:
            print(f"[!] live_shadow_radar Error: {e}")

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    from pathlib import Path
    
    # 1. Load File Parquet Chuẩn Mới
    price_path = Path('data/parquet/price/master_price_l2.parquet')
    prop_path = Path('data/parquet/macro/prop_flow.parquet')
    
    df_l2 = pd.read_parquet(price_path) if price_path.exists() else pd.DataFrame()
    df_prop = pd.read_parquet(prop_path) if prop_path.exists() else pd.DataFrame()
    
    # 2. Khởi tạo Profiler
    profiler = ShadowProfiler(df_l2, df_prop)
    
    # Lấy danh sách mã 
    all_tickers = profiler.df_price['ticker'].unique().tolist()
    market_tickers = [t for t in all_tickers if len(str(t)) == 3]
    
    # 3. QUÉT DARK POOL (Chức năng MỚI ĐỈNH CAO)
    dp_alerts = profiler.scan_dark_pool_deals(market_tickers, lookback_days=15)
    
    # 4. CHẠY RADAR ĐẦU CƠ
    print("\n" + "="*85)
    print(" 🤖 BƯỚC 0: LỌC TỆP ĐẦU CƠ TRAINING (AUTO-PURGE)")
    print("="*85)
    
    # 2. Chạy qua Trạm Thanh trừng để lấy Tệp Training "Thuần chủng"
    training_tickers = profiler._filter_shadow_candidates(market_tickers)
    
    # Pha 1: Học từ Tệp Training chuẩn xác nhất
    if training_tickers:
        rules = profiler.build_criminal_profile(training_tickers, lookback_days=250)
        
        # Pha 2: Mang bộ luật đi quét lại toàn bộ bảng điện
        if rules:
            profiler.live_shadow_radar(market_tickers, rules)
