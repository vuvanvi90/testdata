import pandas as pd
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

class OmniFlowMatrix:
    """
    HỆ THỐNG DATA CUBE ĐA CHIỀU (OMNI-FLOW MATRIX) VERSION 3.0
    🚀 KIẾN TRÚC KÉP (LAMBDA): 
       - Luồng EOD: Xây dựng Chân lý Tuyệt đối từ price_l2 + prop_flow.
       - Luồng T0 (Real-time): Thuật toán Khấu trừ Thỏa thuận Ngoại + Bắt bẫy Sổ lệnh.
    """
    def __init__(self, data_frames: dict, lookback_days=30):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Lò phản ứng OmniFlowMatrix...")
        
        # 1. Nạp các DataFrame từ RAM
        self.df_price_l2 = data_frames.get('price_l2', pd.DataFrame())
        self.df_prop = data_frames.get('prop', pd.DataFrame())
        self.df_comp = data_frames.get('comp', pd.DataFrame())
        self.df_idx = data_frames.get('idx', pd.DataFrame())
        self.df_board = data_frames.get('board', pd.DataFrame())
        self.df_intra = data_frames.get('intra', pd.DataFrame())
        self.df_pt = data_frames.get('put_through', pd.DataFrame())
        
        self.lookback_days = lookback_days
        self.DIVISOR = 1_000_000_000 # Quy đổi ra Tỷ VNĐ
        
        # 2. XÂY DỰNG TRỤC THỜI GIAN TUYỆT ĐỐI (ABSOLUTE TIMELINE)
        self.calendar = self._build_trading_calendar()
        self._set_absolute_timeline()

        # 3. Xây dựng Bản đồ (Mapping) O(1)
        self.ticker_to_universe = {}
        self.ticker_to_sector = {}
        self._build_mappings()
        
        # 4. Trích xuất Snapshot T0 từ Bảng điện
        self.t0_snapshot = self._extract_t0_snapshot()
        
        # 5. Xây dựng Khối dữ liệu Lịch sử (Historical Data Cube)
        self.history_cube = pd.DataFrame()
        self._build_historical_cube()

    def _build_trading_calendar(self):
        """Trích xuất lịch giao dịch thực tế từ dữ liệu Master Price"""
        dates = set()
        if not self.df_price_l2.empty and 'time' in self.df_price_l2.columns:
            time_col = pd.to_datetime(self.df_price_l2['time'], unit='ms' if pd.api.types.is_numeric_dtype(self.df_price_l2['time']) else None)
            dates.update(time_col.dt.date.unique())
        return sorted(list(dates))

    def _set_absolute_timeline(self):
        """
        ĐỘNG CƠ THỜI GIAN TUYỆT ĐỐI (POST-MARKET ROLLOVER)
        Tự động nhận diện Đang trong phiên, Đã đóng cửa, hay Ngày nghỉ.
        """
        from datetime import timedelta
        now = datetime.now()
        current_sys_date = now.date()
        current_time = now.time()
        
        # Tìm ngày EOD mới nhất đã cào được
        latest_l2_date = None
        if not self.df_price_l2.empty and 'time' in self.df_price_l2.columns:
            time_col = pd.to_datetime(self.df_price_l2['time'], unit='ms' if pd.api.types.is_numeric_dtype(self.df_price_l2['time']) else None)
            latest_l2_date = time_col.dt.date.max()
            
        # Một phiên được coi là ĐÃ KẾT THÚC (Post-Market) khi:
        # 1. Đã qua 15h00 chiều.
        # 2. VÀ Dữ liệu CAO CẤP LEVEL-2 của ngày hôm đó đã được tải về thành công.
        is_post_market = False
        if current_time.hour >= 15:
            if latest_l2_date == current_sys_date:
                is_post_market = True

        if is_post_market:
            # Nếu chạy lúc 20h00 tối nay -> Cỗ máy sẽ chuẩn bị cho ngày mai (T0 = Ngày mai)
            # T-1 chính là ngày hôm nay (vừa chốt sổ xong)
            self.t0_date = current_sys_date + timedelta(days=1)
            self.t1_date = current_sys_date
        else:
            # Đang trong phiên (Live), hoặc ngày nghỉ cuối tuần (chưa có data hôm nay)
            self.t0_date = current_sys_date
            # T-1 sẽ là ngày giao dịch gần nhất CÓ TRƯỚC T0
            past_l2 = [d for d in self.calendar if d < self.t0_date]
            self.t1_date = past_l2[-1] if past_l2 else latest_l2_date

        # Xây dựng danh sách quá khứ cho explain_past_movement (Chỉ lấy đến T-1)
        self.past_dates = [d for d in self.calendar if d <= self.t1_date] if self.t1_date else []
        self.t2_date = self.past_dates[-2] if len(self.past_dates) >= 2 else None
        
        status_msg = "POST-MARKET" if is_post_market else "LIVE/WEEKEND"
        print(f"[*] Trục Thời Gian ({status_msg}): T0 (Phiên ngắm bắn) = {self.t0_date.strftime('%d/%m/%Y')} | T-1 (Hậu kiểm L2) = {self.t1_date.strftime('%d/%m/%Y') if self.t1_date else 'N/A'}")

    def _build_mappings(self):
        """Tạo từ điển tra cứu nhanh Universe và Sector cho từng mã"""
        # Mapping Universe (VN30, VNMid, VNSmall)
        if not self.df_idx.empty:
            for _, row in self.df_idx.iterrows():
                ticker, idx_code = row['ticker'], row['index_code']
                if ticker not in self.ticker_to_universe or idx_code in ['VN30', 'VNMidCap', 'VNSmallCap']:
                    self.ticker_to_universe[ticker] = idx_code

        # Mapping Sector (Ngành cấp 2 hoặc 3)
        if not self.df_comp.empty and 'symbol' in self.df_comp.columns:
            sector_col = 'icb_name2' if 'icb_name2' in self.df_comp.columns else 'icb_name'
            for _, row in self.df_comp.dropna(subset=[sector_col]).iterrows():
                self.ticker_to_sector[row['symbol']] = row[sector_col]

        # Lọc lại dữ liệu của price_l2 (chỉ lấy các field cần và rename open/high/low/close)
        filterd_cols = [
            'time', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'matched_volume',
            'fr_net_value_matched', 'fr_net_value_deal', 'average_buy_trade_volume', 'average_sell_trade_volume',
            'total_buy_unmatched_volume', 'total_sell_unmatched_volume', 'total_net_trade_volume', 'fr_available_percentage',
            'fr_owned_percentage', 'fr_buy_value_deal', 'fr_sell_value_deal', 'market_cap'
        ]
        df_price_l2_v = self.df_price_l2[[c for c in filterd_cols if c in self.df_price_l2.columns]].copy()
        if 'matched_volume' in df_price_l2_v.columns and 'volume' not in df_price_l2_v.columns:
            df_price_l2_v = df_price_l2_v.rename(columns={'matched_volume': 'volume'})
        self.df_price_l2 = df_price_l2_v

    def _extract_t0_snapshot(self):
        """
        🚀 BÓC TÁCH DỮ LIỆU T0 TỪ BẢNG ĐIỆN (REAL-TIME SNAPSHOT V3.3)
        Khai thác 100% dữ liệu Vi cấu trúc: Order Book Depth, Limit Locks, Spread Vacuum.
        """
        if self.df_board.empty: return {}
        
        t0_dict = {}
        col_ticker = 'symbol' if 'symbol' in self.df_board.columns else 'ticker'
        has_time_col = 'time' in self.df_board.columns
        
        for _, row in self.df_board.iterrows():
            if has_time_col:
                row_date = pd.to_datetime(row['time'], unit='ms' if isinstance(row['time'], int) else None).date()
                if row_date != self.t0_date: continue

            ticker = row[col_ticker]
            
            def safe_fl(val): 
                try: return float(val) if pd.notna(val) and val != "" else 0.0
                except: return 0.0

            # ---------------------------------------------------------
            # 1. DÒNG TIỀN NGOẠI VÀ ROOM
            # ---------------------------------------------------------
            f_buy_vol = safe_fl(row.get('foreign_buy_volume', 0))
            f_sell_vol = safe_fl(row.get('foreign_sell_volume', 0))
            f_room = safe_fl(row.get('foreign_room', 0))
            
            # Lấy thẳng VWAP của bảng điện (average_price)
            avg_price = safe_fl(row.get('average_price', row.get('close_price', 0)))
            f_net_val_bn = ((f_buy_vol - f_sell_vol) * avg_price) / self.DIVISOR

            # ---------------------------------------------------------
            # 2. HÌNH THÁI NẾN T0 & KIỂM TRA TRẦN/SÀN
            # ---------------------------------------------------------
            c_price = safe_fl(row.get('close_price', 0))
            h_price = safe_fl(row.get('high_price', c_price))
            l_price = safe_fl(row.get('low_price', c_price))
            ceil_p = safe_fl(row.get('ceiling_price', 0))
            floor_p = safe_fl(row.get('floor_price', 0))
            
            # Tính Vị trí Bóng nến (Wick Ratio): > 0.7 là Cầu áp đảo, < 0.3 là Cung ép
            range_p = h_price - l_price
            wick_ratio = (c_price - l_price) / range_p if range_p > 0 else 0.5
            
            # Cờ báo Nhốt Trần / Nhốt Sàn
            limit_lock = "NONE"
            if ceil_p > 0 and c_price >= ceil_p: limit_lock = "CEILING"
            elif floor_p > 0 and c_price <= floor_p: limit_lock = "FLOOR"

            # ---------------------------------------------------------
            # 3. ĐỘ SÂU SỔ LỆNH (DEPTH OF BOOK) VÀ BẮT BẪY
            # ---------------------------------------------------------
            b_p1, b_p2, b_p3 = safe_fl(row.get('bid_price_1',0)), safe_fl(row.get('bid_price_2',0)), safe_fl(row.get('bid_price_3',0))
            a_p1, a_p2, a_p3 = safe_fl(row.get('ask_price_1',0)), safe_fl(row.get('ask_price_2',0)), safe_fl(row.get('ask_price_3',0))
            
            b_v1, b_v2, b_v3 = safe_fl(row.get('bid_vol_1',0)), safe_fl(row.get('bid_vol_2',0)), safe_fl(row.get('bid_vol_3',0))
            a_v1, a_v2, a_v3 = safe_fl(row.get('ask_vol_1',0)), safe_fl(row.get('ask_vol_2',0)), safe_fl(row.get('ask_vol_3',0))
            
            bid_vol_total = b_v1 + b_v2 + b_v3
            ask_vol_total = a_v1 + a_v2 + a_v3
            imbalance = (bid_vol_total - ask_vol_total) / (bid_vol_total + ask_vol_total) if (bid_vol_total + ask_vol_total) > 0 else 0

            # Phân tích Book Shape (Hình thái Kê Lệnh)
            book_shape = "NORMAL"
            if limit_lock == "CEILING":
                book_shape = "CEILING_LOCKED"
            elif limit_lock == "FLOOR":
                book_shape = "FLOOR_LOCKED"
            else:
                # BẪY 1: Kê ảo dưới sâu (Hơn 60% Dư mua nằm tít ở Bid 3)
                if bid_vol_total > 0 and (b_v3 / bid_vol_total) > 0.6:
                    book_shape = "SPOOFING_BID"
                # BẪY 2: Tường chặn bán tàn nhẫn (Hơn 60% Dư bán đè ngay sát Ask 1)
                elif ask_vol_total > 0 and (a_v1 / ask_vol_total) > 0.6:
                    book_shape = "ASK_WALL_SUPPRESSION"
            
            # Cảnh báo Lỗ hổng Spread (Thanh khoản rỗng)
            spread_vacuum = False
            if a_p1 > 0 and b_p1 > 0:
                spread_gap = (a_p1 - b_p1) / c_price
                if spread_gap > 0.015: # Chênh lệch > 1.5% là báo động rỗng thanh khoản
                    spread_vacuum = True

            # ---------------------------------------------------------
            # 4. TRÍCH XUẤT THỎA THUẬN T0 CHO THUẬT TOÁN KHẤU TRỪ
            # ---------------------------------------------------------
            pt_val_bn = 0.0
            if not self.df_pt.empty:
                col_pt = 'symbol' if 'symbol' in self.df_pt.columns else 'ticker'
                df_pt_ticker = self.df_pt[self.df_pt[col_pt] == ticker]
                if not df_pt_ticker.empty:
                    pt_dates = pd.to_datetime(df_pt_ticker['time']).dt.date
                    df_pt_t0 = df_pt_ticker[pt_dates == self.t0_date]
                    if not df_pt_t0.empty:
                        pt_val_bn = df_pt_t0['match_value'].sum() / self.DIVISOR

            # ---------------------------------------------------------
            # ĐÓNG GÓI HỒ SƠ VI CẤU TRÚC
            # ---------------------------------------------------------
            t0_dict[ticker] = {
                't0_date': self.t0_date,
                't0_foreign_net_bn': f_net_val_bn,
                't0_f_room': f_room,
                't0_pt_val_bn': pt_val_bn,
                't0_imbalance': imbalance,
                't0_last_price': c_price,
                't0_vwap': avg_price,          # Đã lấy thẳng từ Board
                't0_wick_ratio': wick_ratio,   # Vị trí nến T0
                't0_limit_lock': limit_lock,   # Cờ Trần/Sàn
                't0_book_shape': book_shape,   # Hình thái Đội lái Kê lệnh
                't0_spread_vacuum': spread_vacuum, # Báo động Rỗng thanh khoản
                'is_fresh': True
            }
        return t0_dict

    def _build_historical_cube(self):
        """
        KHỐI 1 & 2: ÉP TOÀN BỘ DỮ LIỆU VÀO MỘT MA TRẬN DUY NHẤT
        """
        print("[*] Đang xây dựng Ma trận Lịch sử (Historical Cube)...")
        if self.df_price_l2.empty: return

        # 1. Chuẩn hóa & Lọc 30 phiên gần nhất để nhẹ RAM
        df = self.df_price_l2.copy()
        df['time'] = pd.to_datetime(df['time']).dt.normalize()
        
        if self.past_dates:
            cutoff_date = self.past_dates[-self.lookback_days] if len(self.past_dates) > self.lookback_days else self.past_dates[0]
            df = df[df['time'].dt.date >= cutoff_date]

        # 2. Xử lý Prop EOD
        if not self.df_prop.empty:
            prop = self.df_prop.copy()
            prop['time'] = pd.to_datetime(prop['time']).dt.normalize()
            # Lấy các cột bóc tách của Tự Doanh
            prop_cols = ['ticker', 'time', 'prop_net_val_matched', 'prop_net_val_deal', 'prop_net_value']
            prop_merge = prop[[c for c in prop_cols if c in prop.columns]]
            df = pd.merge(df, prop_merge, on=['ticker', 'time'], how='left')

        # Điền 0 cho các ngày không có giao dịch để tránh lỗi NaN
        df.fillna(0, inplace=True)

        # 3. TÍNH TOÁN CÁC METRICS ĐỊNH LƯỢNG (Vectorized)
        df['total_val_bn'] = (df['close'] * df['volume']) / self.DIVISOR

        # --- KHỐI NGOẠI ---
        # Ưu tiên lấy từ L2, nếu không có thì lấy Fallback
        if 'fr_net_value_matched' in df.columns:
            df['f_net_bn'] = df['fr_net_value_matched'] / self.DIVISOR
            df['f_deal_bn'] = df.get('fr_net_value_deal', 0) / self.DIVISOR
        else:
            df['f_net_bn'] = 0
            df['f_deal_bn'] = 0

        # --- TỰ DOANH ---
        # Lấy trực tiếp từ file Prop Premium
        if 'prop_net_val_matched' in df.columns:
            df['p_net_bn'] = df['prop_net_val_matched'] / self.DIVISOR
            df['p_deal_bn'] = df.get('prop_net_val_deal', 0) / self.DIVISOR
        else:
            df['p_net_bn'] = 0
            df['p_deal_bn'] = 0

        # --- DÒNG TIỀN TỔNG HỢP & DẤU CHÂN ---
        df['sm_net_adj'] = df['f_net_bn'] + df['p_net_bn']
        df['deal_val_bn'] = df.get('deal_value', 0) / self.DIVISOR

        # Shadow Flow: Dòng tiền ngầm của Cổ đông lớn/Lái nội = Tổng Thỏa Thuận - (Tây Deal + Tự Doanh Deal)
        df['shadow_flow_bn'] = df['deal_val_bn'] - df['f_deal_bn'].abs() - df['p_deal_bn'].abs()
        df.loc[df['shadow_flow_bn'] < 0, 'shadow_flow_bn'] = 0 # Fix sai số nhỏ

        # Đánh cờ (Flags) báo hiệu Sang tay cho Sniper.py in ra báo cáo
        df['is_pt_f'] = df['f_deal_bn'].abs() > 0
        df['is_pt_p'] = df['p_deal_bn'].abs() > 0

        # 4. GẮN MÁC RỔ & NGÀNH
        df['universe'] = df['ticker'].map(lambda x: self.ticker_to_universe.get(x, 'HOSE'))
        df['sector'] = df['ticker'].map(lambda x: self.ticker_to_sector.get(x, 'Unknown'))

        # Chuẩn hóa Date để dễ tra cứu sau này
        df['date'] = df['time'].dt.date

        # LƯU TRỮ TỶ LỆ HỞ ROOM NGOẠI VÀO CUBE
        if 'fr_available_percentage' in df.columns:
            df['fr_avail_pct'] = df['fr_available_percentage']
        else:
            df['fr_avail_pct'] = 1.0 

        self.history_cube = df.sort_values(['ticker', 'time']).reset_index(drop=True)
        print(f"[OK] Ma trận hoàn tất: {len(self.history_cube)} records, sẵn sàng cho Inference Engine.")

    def _get_intraday_t0_metrics(self, ticker):
        """Hàm nội bộ: Trích xuất siêu tốc Lực Mua/Bán chủ động T0 từ df_intra"""
        if self.df_intra.empty: return 0, 0, 0
        
        # 1. Lọc đúng mã
        df_i = self.df_intra[self.df_intra['ticker'] == ticker]
        if df_i.empty: return 0, 0, 0

        # Lọc đúng mã và đúng ngày mỏ neo
        df_today = df_i[df_i['time'].dt.date == self.t0_date]

        if df_today.empty: return 0, 0, 0

        # 3. Phân loại lệnh & Tính tiền
        is_bu = df_today['match_type'].isin(['Buy', 'BU', 'B'])
        is_sd = df_today['match_type'].isin(['Sell', 'SD', 'S'])

        trade_val = (df_today['price'] * df_today['volume']) / self.DIVISOR
        active_buy_bn = trade_val[is_bu].sum()
        active_sell_bn = trade_val[is_sd].sum()

        net_active_bn = active_buy_bn - active_sell_bn
        # Tính VWAP Intraday
        total_vol = df_today['volume'].sum()
        vwap_t0 = (df_today['price'] * df_today['volume']).sum() / total_vol if total_vol > 0 else 0

        return net_active_bn, trade_val.sum(), vwap_t0  # Trả về thêm vwap_t0

    def _analyze_level_2_microstructure(self, ticker):
        """ĐỘNG CƠ KIỂM TOÁN L2 (ĐỌC TỪ MASTER_PRICE_L2.PARQUET)"""
        if self.df_price_l2.empty or 'ticker' not in self.df_price_l2.columns:
            return None

        if self.t1_date is None:
            return None # Không xác định được ngày T-1 hệ thống
            
        df_ticker_l2 = self.df_price_l2[self.df_price_l2['ticker'] == ticker].copy()
        if df_ticker_l2.empty:
            return None
            
        # Ép kiểu an toàn cột time
        df_ticker_l2['date'] = pd.to_datetime(
            df_ticker_l2['time'], 
            unit='ms' if pd.api.types.is_numeric_dtype(df_ticker_l2['time']) else None
        ).dt.date
        
        # 🛡️ LỌC ĐÚNG NGÀY T-1 (Xóa sổ bẫy iloc[-1])
        df_t1 = df_ticker_l2[df_ticker_l2['date'] == self.t1_date]
        
        if df_t1.empty:
            return None # Dữ liệu L2 của ngày hôm qua bị Miss
            
        t1_row = df_t1.iloc[0]
        t1_date_str = self.t1_date.strftime('%d/%m/%Y')
        
        def safe_num(val): return float(val) if pd.notna(val) else 0.0
        
        # 1. WHALE FOOTPRINT RATIO
        avg_buy = safe_num(t1_row.get('average_buy_trade_volume', 0))
        avg_sell = safe_num(t1_row.get('average_sell_trade_volume', 0))
        whale_ratio = (avg_buy / avg_sell) if avg_sell > 0 else 1.0
        
        # 2. SPOOFING RATIO 
        buy_unmatched = safe_num(t1_row.get('total_buy_unmatched_volume', 0))
        sell_unmatched = safe_num(t1_row.get('total_sell_unmatched_volume', 0))
        spoofing_ratio = (buy_unmatched / sell_unmatched) if sell_unmatched > 0 else 1.0
        
        # 3. ABSORPTION MOMENTUM
        net_trade_vol = safe_num(t1_row.get('total_net_trade_volume', 0))
        
        # 4. FOREIGN INTENT & OWNERSHIP
        fr_room_avail = safe_num(t1_row.get('fr_available_percentage', 1.0))
        fr_owned_pct = safe_num(t1_row.get('fr_owned_percentage', 0))
        fr_deal_net_bn = (safe_num(t1_row.get('fr_buy_value_deal', 0)) - safe_num(t1_row.get('fr_sell_value_deal', 0))) / self.DIVISOR

        # 5. MARKET CAP & SUPPLY
        market_cap = safe_num(t1_row.get('market_cap', 0)) / self.DIVISOR # Tỷ VNĐ
        
        return {
            "t1_date": t1_date_str,
            "whale_ratio": whale_ratio,
            "spoofing_ratio": spoofing_ratio,
            "net_trade_vol": net_trade_vol,
            "fr_room_avail": fr_room_avail,
            "fr_owned_pct": fr_owned_pct,
            "fr_deal_net_bn": fr_deal_net_bn,
            "avg_buy_vol": avg_buy,
            "avg_sell_vol": avg_sell,
            "market_cap_bn": market_cap,
            "is_trap": (whale_ratio < 0.6 and spoofing_ratio > 2.0),
            "is_whale_buy": (whale_ratio > 1.5),
            "is_free_float_squeeze": (fr_room_avail < 0.05) # Hở room dưới 5% (Độ cạn kiệt cổ phiếu của Khối Ngoại)
        }

    def explain_past_movement(self, ticker, lookback_days=10):
        """
        ĐỘNG CƠ GIẢI THÍCH QUÁ KHỨ (RETROSPECTIVE ENGINE)
        Phân tích đa chiều: Giá vs Dòng tiền Thể chế vs Dòng tiền Lái nội
        """
        if self.history_cube.empty or not self.past_dates:
            return {"error": "Không có dữ liệu lịch sử"}

        # Xác định khung thời gian từ T-N đến T-1
        target_past_dates = self.past_dates[-lookback_days:]
        if not target_past_dates:
            return {"error": "Dữ liệu quá khứ không đủ"}
            
        start_date = target_past_dates[0]
        end_date = target_past_dates[-1] # Chính là t1_date

        df_t = self.history_cube[
            (self.history_cube['ticker'] == ticker) &
            (self.history_cube['date'] >= start_date) &
            (self.history_cube['date'] <= end_date)
        ].sort_values('time')
        
        if df_t.empty: 
            return {"error": "Không có dữ liệu lịch sử trong khoảng thời gian này"}

        # 1. Tính toán Biến động Giá
        start_price = df_t['close'].iloc[0]
        end_price = df_t['close'].iloc[-1]
        price_change_pct = (end_price - start_price) / start_price * 100 if start_price > 0 else 0

        # 2. TỔNG HỢP DÒNG TIỀN TRONG KỲ (SỬ DỤNG DỮ LIỆU KHỚP LỆNH L2)
        total_sm_net = df_t['sm_net_adj'].sum()
        total_f_net = df_t['f_net_bn'].sum()
        total_shadow = df_t['shadow_flow_bn'].sum()
        total_val = df_t['total_val_bn'].sum()

        # 3. QUÉT CÁC NGÀY SANG TAY ĐỘC LẬP (NGOẠI VÀ NỘI)
        pt_f_days = df_t[df_t['is_pt_f']]['time'].dt.strftime('%d/%m').tolist()
        pt_p_days = df_t[df_t['is_pt_p']]['time'].dt.strftime('%d/%m').tolist()

        # 4. CHẨN ĐOÁN (HEURISTIC RULES)
        diagnosis = []
        if price_change_pct > 3.0: # XU HƯỚNG TĂNG
            trend = "TĂNG"
            if total_sm_net > (total_val * 0.15):
                diagnosis.append(f"Sóng Thể chế: Ngoại/Tự doanh dẫn dắt (Gom {total_sm_net:.1f} Tỷ).")
            elif total_shadow > (total_val * 0.7) and total_sm_net < -10.0:
                diagnosis.append(f"Kéo Xả ảo: Lái nội kéo giá (Ẩn {total_shadow:.1f} Tỷ) để Tay to thoát hàng (Xả {total_sm_net:.1f} Tỷ). ⚠️ Rủi ro Bull-Trap!")
            else:
                diagnosis.append(f"Đồng thuận: Dòng tiền lan tỏa đều, Lái nội chi phối chính ({total_shadow:.1f} Tỷ).")

        elif price_change_pct < -3.0: # XU HƯỚNG GIẢM
            trend = "GIẢM"
            if total_sm_net < -(total_val * 0.15):
                diagnosis.append(f"Tắm Máu Thể chế: Áp lực bán tháo dữ dội (Xả {total_sm_net:.1f} Tỷ).")
            elif total_shadow > (total_val * 0.7) and total_sm_net > 10.0:
                diagnosis.append(f"Đạp Gom (Washout): Nhỏ lẻ hoảng loạn, Lái nội đè giá nhưng Tay to đang âm thầm gom ({total_sm_net:.1f} Tỷ). 🌟 Tín hiệu Rũ bỏ!")
            else:
                diagnosis.append("Phân phối diện rộng, cung cầu suy yếu tự nhiên.")

        else: # TÍCH LŨY (ĐI NGANG)
            trend = "ĐI NGANG"
            if total_sm_net > 20.0:
                diagnosis.append(f"Gom Ngầm: Neo giá để Tay to gom hàng ({total_sm_net:.1f} Tỷ). Chờ nổ!")
            elif total_sm_net < -20.0:
                diagnosis.append(f"Phân phối Ngầm: Kéo xả trong biên độ hẹp để Tay to thoát hàng ({total_sm_net:.1f} Tỷ).")
            else:
                diagnosis.append("Siết nền: Thanh khoản cạn, chờ gió đông.")

        # GHI CHÚ SANG TAY CHI TIẾT
        if pt_f_days or pt_p_days:
            msgs = []
            if pt_f_days: msgs.append(f"Ngoại ({', '.join(pt_f_days)})")
            if pt_p_days: msgs.append(f"Nội ({', '.join(pt_p_days)})")
            diagnosis.append(f"⚠️ Sang tay ngầm (Trao kho) vào: {' & '.join(msgs)}.")

        # NHÚNG KẾT QUẢ KIỂM TOÁN L2 VÀO HỒ SƠ QUÁ KHỨ
        l2_data = self._analyze_level_2_microstructure(ticker)
        has_l2_trap = False
        
        if l2_data:
            if l2_data['is_trap']:
                diagnosis.append(f"🚨 BẪY L2 ({l2_data['t1_date']}): Dư mua ảo gấp {l2_data['spoofing_ratio']:.1f}x nhưng lệnh táng thực tế to gấp {1/l2_data['whale_ratio']:.1f}x lệnh mua!")
                has_l2_trap = True
            elif l2_data['is_whale_buy']:
                diagnosis.append(f"🐋 Cá voi ({l2_data['t1_date']}): Lệnh mua to gấp {l2_data['whale_ratio']:.1f}x lệnh bán.")
                
            if l2_data['is_free_float_squeeze']:
                diagnosis.append(f"💎 Cạn Cung Trôi Nổi: Ngoại đã gom {l2_data['fr_owned_pct']*100:.1f}% cty. Hở room < 5%!")

        return {
            "trend": trend,
            "change_pct": price_change_pct,
            "sm_net": total_sm_net,
            "f_net": total_f_net,
            "shadow": total_shadow,
            "verdict": " | ".join(diagnosis),
            "has_l2_trap": has_l2_trap,
            "l2_data": l2_data
        }

    def predict_t0_action(self, ticker, past_context=None):
        """
        ĐỘNG CƠ DỰ BÁO T0 V3.3 (ULTIMATE MICROSTRUCTURE)
        🚀 Tích hợp: Book Depth, Wick Ratio, Limit Lock và Chống Bẫy Kê Lệnh Ảo.
        """
        t0_data = self.t0_snapshot.get(ticker, {})
        is_offline = not t0_data or (t0_data.get('t0_last_price') == 0)

        if is_offline:
            return {
                "verdict": "OFFLINE (Chờ Phiên Mới)", "last_price": 0, "net_active_bn": 0,
                "driver_msg": "Thị trường đóng cửa", "details": "Chưa có dữ liệu T0", "is_offline": True
            }

        # 1. TRÍCH XUẤT HỒ SƠ VI CẤU TRÚC (SỨC MẠNH V3.3)
        f_net_t0 = t0_data.get('t0_foreign_net_bn', 0)
        pt_t0 = t0_data.get('t0_pt_val_bn', 0)
        imbalance = t0_data.get('t0_imbalance', 0)
        last_price = t0_data.get('t0_last_price', 0)
        vwap_t0 = t0_data.get('t0_vwap', 0)
        wick_ratio = t0_data.get('t0_wick_ratio', 0.5)
        limit_lock = t0_data.get('t0_limit_lock', "NONE")
        book_shape = t0_data.get('t0_book_shape', "NORMAL")
        spread_vacuum = t0_data.get('t0_spread_vacuum', False)
        
        net_active_bn, total_intra_val, _ = self._get_intraday_t0_metrics(ticker)

        # 🚀 THUẬT TOÁN KHẤU TRỪ NGOẠI T0
        f_matched_net_t0 = f_net_t0
        if pt_t0 > 0:
            if f_net_t0 > 0: f_matched_net_t0 = max(0, f_net_t0 - pt_t0)
            else: f_matched_net_t0 = min(0, f_net_t0 + pt_t0)

        # 🚀 THUẬT TOÁN ĐỊNH DANH TÁC NHÂN (T0 DRIVER)
        driver_msg = "Chưa rõ tác nhân"
        f_impact_pct = (f_matched_net_t0 / total_intra_val * 100) if total_intra_val > 0 else 0
        price_is_up = last_price > vwap_t0 if vwap_t0 > 0 else net_active_bn > 0
        
        if price_is_up:
            if f_impact_pct > 15.0: driver_msg = "🌍 NGOẠI DẪN SÓNG (Tây chủ động đẩy giá)"
            elif f_impact_pct < -10.0 and net_active_bn > 0: driver_msg = "🔥 NỘI CÂN TÂY (Lái Nội hấp thụ lực xả để kéo)"
            elif net_active_bn > (total_intra_val * 0.15): driver_msg = "🇻🇳 SÓNG THUẦN NỘI (Dòng tiền nội đẩy giá)"
            else: driver_msg = "🕊️ NHỎ LẺ FOMO (Thiếu dấu chân cá mập)"
        else:
            if net_active_bn > (total_intra_val * 0.1): driver_msg = "🛡️ LÁI ĐÈ GOM (Giá đỏ nhưng Cầu chủ động ăn vã hàng)"
            elif f_impact_pct < -15.0: driver_msg = "🩸 TÂY ÚP SỌT (Ngoại chủ động xả gãy giá)"
            elif f_impact_pct > 10.0 and net_active_bn < 0: driver_msg = "🛡️ NGOẠI ĐỠ GIÁ (Nội xả hoảng loạn, Tây nhặt hàng)"
            elif net_active_bn < -(total_intra_val * 0.15): driver_msg = "📉 NỘI TỰ DẪM ĐẠP (Dòng tiền nội tháo chạy)"
            else: driver_msg = "🧊 CẠN CẦU (Rơi tự do do thiếu lực đỡ)"

        # ⚖️ HỆ THỐNG CHẤM ĐIỂM LƯỢNG TỬ (SCORING ENGINE)
        score, signals = 0, []
        verdict = "NEUTRAL (Giằng co)"

        # A. Xử lý trạng thái Đặc biệt (Limit Lock)
        if limit_lock == "CEILING":
            score += 10; signals.append("🔥 TÍM TRẦN: Dòng tiền cực đại quét sạch dư bán"); verdict = "BULLISH (Tím Trần)"
        elif limit_lock == "FLOOR":
            score -= 10; signals.append("🩸 NHỐT SÀN: Trạng thái trắng bên mua"); verdict = "BEARISH (Nhốt Sàn)"
        
        if limit_lock == "NONE":
            # B. Đánh giá Hình thái Kê Lệnh (Book Shape)
            if book_shape == "SPOOFING_BID":
                score -= 2; signals.append("⚠️ BẪY KÊ LỆNH: Dư mua ảo nằm tuốt ở dưới sâu (Bid 3)")
            elif book_shape == "ASK_WALL_SUPPRESSION":
                if last_price < vwap_t0:
                    # Phân biệt Đè Gom và Tường Chết Chóc
                    if net_active_bn < -20.0 or f_matched_net_t0 < -20.0:
                        score -= 2; signals.append("🩸 TƯỜNG CHẾT CHÓC: Lái chặn trên đầu + táng thẳng xuống dưới (Không lối thoát!)")
                    else:
                        score += 1; signals.append("🛡️ ĐÈ GOM: Lái chặn tường bán dày đặc để ép nhỏ lẻ nhả hàng")
                else:
                    score -= 1; signals.append("⚠️ CHẶN TRÊN: Tường bán dày đặc cản trở đà tăng giá")
            
            # C. Đánh giá Động lượng (Wick Ratio & VWAP)
            if wick_ratio > 0.8: score += 2; signals.append("🚀 ĐÓNG NẾN QUYẾT LIỆT: Giá sát đỉnh phiên")
            elif wick_ratio < 0.2: score -= 2; signals.append("📉 ÁP LỰC ĐÈ CUỐI PHIÊN: Giá sát đáy phiên")
            
            if last_price > vwap_t0: score += 1; signals.append("🟢 GIÁ TRÊN VWAP: Phe Bò đang kiểm soát")
            else: score -= 1; signals.append("🔴 GIÁ DƯỚI VWAP: Phe Gấu đang đè giá")

            # D. Đánh giá Khớp lệnh chủ động & Ngoại
            if net_active_bn > (total_intra_val * 0.15): score += 2; signals.append(f"🌊 Cầu chủ động mạnh (+{net_active_bn:.1f} Tỷ)")
            elif net_active_bn < -(total_intra_val * 0.15): score -= 2; signals.append(f"🩸 Cung chủ động xả ({net_active_bn:.1f} Tỷ)")

            if f_matched_net_t0 > 10.0: score += 2; signals.append(f"🐋 Tây múc ròng (+{f_matched_net_t0:.1f} Tỷ)")
            elif f_matched_net_t0 < -10.0: score -= 2; signals.append(f"🦈 Tây xả ròng ({f_matched_net_t0:.1f} Tỷ)")

        # E. Cảnh báo Rỗng thanh khoản
        if spread_vacuum:
            score -= 1; signals.append("🚨 THANH KHOẢN RỖNG: Chênh lệch Bid/Ask > 1.5%. Cẩn thận trượt giá!")

        # 🛡️ KIỂM TOÁN NGỮ CẢNH QUÁ KHỨ (BẢN VÁ TRAP L2)
        if past_context and "error" not in past_context:
            if past_context.get('has_l2_trap'):
                score -= 50; signals.insert(0, "🚨 TỬ HUYỆT L2: Kê ảo dụ FOMO để xả thật"); verdict = "BEARISH (Bẫy Kéo Xả Ảo)"

        # QUYỀN PHỦ QUYẾT CỦA CÁ VOI (WHALE OVERRIDE)
        # Tự động xác định ngưỡng Tiền Tỷ theo rổ Vốn hóa (VN30, MidCap, SmallCap)
        whale_thresh = 50.0 # Mặc định cho MidCap/HOSE
        if hasattr(self, 'df_idx') and not self.df_idx.empty:
            match = self.df_idx[self.df_idx['ticker'] == ticker]
            if not match.empty:
                idx_codes = match['index_code'].tolist()
                is_vn30 = any('VN30' in str(code).upper() for code in idx_codes)
                is_small = any('VNSMALLCAP' in str(code).upper() for code in idx_codes)

                if is_vn30: whale_thresh = 100.0
                elif is_small: whale_thresh = 15.0

        # Nếu đang là Bẫy L2 thì cấm không cho Override
        if "Bẫy Kéo Xả Ảo" not in verdict:
            if net_active_bn > whale_thresh or f_matched_net_t0 > whale_thresh:
                score = max(score, 5) # Ép thẳng lên mức BULLISH tối đa
                signals.append(f"🌟 WHALE OVERRIDE: Lực mua càn quét (>{whale_thresh} Tỷ), xóa bỏ mọi rủi ro nhiễu!")
            elif net_active_bn < -whale_thresh or f_matched_net_t0 < -whale_thresh:
                score = min(score, -5) # Ép thẳng xuống mức BEARISH tột độ
                signals.append(f"🩸 WHALE OVERRIDE: Lực xả tàn bạo (<-{whale_thresh} Tỷ), chặn đứng mọi nỗ lực đỡ giá!")

        # 🎯 PHÁN QUYẾT CUỐI CÙNG
        if verdict == "NEUTRAL (Giằng co)": 
            if score >= 4: verdict = "🔥 BULLISH (Dòng tiền T0 bùng nổ mạnh)"
            elif score <= -4: verdict = "🩸 BEARISH (Áp lực xả T0 cực lớn)"
            elif score > 0: verdict = "🟢 TILT BULL (Ưu thế Mua)"
            elif score < 0: verdict = "🔴 TILT BEAR (Ưu thế Bán)"
            else: verdict = "⚖️ NEUTRAL (Cân bằng/Chờ xác nhận)"

        return {
            "verdict": verdict, "last_price": last_price, "net_active_bn": net_active_bn,
            "t0_f_matched_net_bn": f_matched_net_t0, "t0_f_impact_pct": f_impact_pct,
            "driver_msg": driver_msg, "imbalance": imbalance,
            "details": " | ".join(signals) if signals else "Dòng tiền giằng co quanh tham chiếu.",
            "l2_data": past_context.get('l2_data', None) if past_context else None,
            "is_offline": is_offline
        }

if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = 'data/parquet'
    
    def load_pq(sub_path):
        p = Path(DATA_DIR) / sub_path
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    data_frames = {
        'price_l2': load_pq('price/master_price_l2.parquet'),
        'prop': load_pq('macro/prop_flow.parquet'),
        'comp': load_pq('company/master_company.parquet'),
        'idx': load_pq('macro/index_components.parquet'),
        'board': load_pq('board/master_board.parquet'),
        'intra': load_pq('intraday/master_intraday.parquet'),
        'put_through': load_pq('intraday/master_put_through.parquet') # Đã nạp Radar Thỏa thuận T0
    }

    omni = OmniFlowMatrix(data_frames, lookback_days=30)
    
    test_tickers = ['GMD'] # Thử với VN30, MidCap và SmallCap
    
    for ticker in test_tickers:
        print("\n" + "="*95)
        print(f" 🎯 OMNI-MATRIX REPORT: MÃ [ {ticker} ] - RỔ: {omni.ticker_to_universe.get(ticker, 'N/A')}")
        print("="*95)
        
        # 1. HỒ SƠ QUÁ KHỨ (10 Ngày qua)
        past = omni.explain_past_movement(ticker, lookback_days=10)
        if "error" not in past:
            print(f" 🕰️ GIẢI MÃ QUÁ KHỨ (10 Phiên & Bẫy T-1):")
            print(f"    - Xu hướng     : {past['trend']} ({past['change_pct']:+.2f}%)")
            print(f"    - Dòng tiền 10D: Thể chế {past['sm_net']:+.1f} Tỷ | Ẩn/Lái {past['shadow']:+.1f} Tỷ")
            print(f"    - Chẩn đoán    : 🧠 {past['verdict']}")
        else:
            print(f"    [!] {past['error']}")

        print("-" * 95)
        
        # 2. BỨC TRANH T0 (NGAY LÚC NÀY)
        now = omni.predict_t0_action(ticker, past_context=past)
        if "error" not in now:
            print(f" ⚡ DỰ BÁO HIỆN TẠI (PHIÊN T0): Giá {now['last_price']:,.0f}đ")
            print(f"    - Mua/Bán C.Động : {now['net_active_bn']:+.1f} Tỷ (Từ lệnh Khớp Intraday)")
            print(f"    - Ngoại T0       : {now['t0_f_matched_net_bn']:+.1f} Tỷ (Đã Khấu trừ Thỏa Thuận)")
            print(f"    - Sổ lệnh (Bid)  : Mất cân bằng {now['imbalance']:+.2f} (Dương = Kê mua, Âm = Chặn bán)")
            print(f"    - KẾT LUẬN       : 🎯 {now['verdict']}")
            print(f"    - Tín hiệu phụ   : {now['details']}")
        else:
            print(f"    [!] {now['error']}")