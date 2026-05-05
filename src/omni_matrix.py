import pandas as pd
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

class OmniFlowMatrix:
    """
    HỆ THỐNG DATA CUBE ĐA CHIỀU (OMNI-FLOW MATRIX)
    Kiến trúc phân rã Dòng tiền Thể chế & Lái nội xuyên suốt Rổ Vốn Hóa và Ngành Nghề.
    """
    def __init__(self, data_frames: dict, lookback_days=30):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Lò phản ứng OmniFlowMatrix...")
        
        # 1. Nạp các DataFrame từ RAM
        self.df_price = data_frames.get('price', pd.DataFrame())
        self.df_price_l2 = data_frames.get('price_l2', pd.DataFrame())
        self.df_foreign = data_frames.get('foreign', pd.DataFrame())
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

        # 2. Xây dựng Bản đồ (Mapping) O(1)
        self.ticker_to_universe = {}
        self.ticker_to_sector = {}
        self._build_mappings()
        
        # 3. Trích xuất Snapshot T0 từ Bảng điện
        self.t0_snapshot = self._extract_t0_snapshot()
        
        # 4. Xây dựng Khối dữ liệu Lịch sử (Historical Data Cube)
        self.history_cube = pd.DataFrame()
        self._build_historical_cube()

    def _build_trading_calendar(self):
        """Trích xuất lịch giao dịch thực tế từ dữ liệu Master Price"""
        dates = set()
        if not self.df_price.empty and 'time' in self.df_price.columns:
            time_col = pd.to_datetime(self.df_price['time'], unit='ms' if pd.api.types.is_numeric_dtype(self.df_price['time']) else None)
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

        latest_price_date = None
        if not self.df_price.empty and 'time' in self.df_price.columns:
            time_col = pd.to_datetime(self.df_price['time'], unit='ms' if pd.api.types.is_numeric_dtype(self.df_price['time']) else None)
            latest_price_date = time_col.dt.date.max()
            
        # Một phiên được coi là ĐÃ KẾT THÚC (Post-Market) khi:
        # 1. Đã qua 15h00 chiều.
        # 2. VÀ Dữ liệu CAO CẤP LEVEL-2 của ngày hôm đó đã được tải về thành công.
        is_post_market = False
        if current_time.hour >= 15:
            # BẢN VÁ: Bỏ điều kiện "latest_price_date", chỉ dùng "latest_l2_date" làm mỏ neo chốt sổ.
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

    def _extract_t0_snapshot(self):
        """BÓC TÁCH DỮ LIỆU T0 TỪ BẢNG ĐIỆN (REAL-TIME SNAPSHOT)"""
        if self.df_board.empty: return {}
        
        t0_dict = {}
        col_ticker = 'symbol' if 'symbol' in self.df_board.columns else 'ticker'

        # Kiểm tra nếu df_board có cột time để mapping
        has_time_col = 'time' in self.df_board.columns
        
        for _, row in self.df_board.iterrows():
            # Nếu board data có time, nó phải là ngày t0_date
            if has_time_col:
                row_date = pd.to_datetime(row['time'], unit='ms' if isinstance(row['time'], int) else None).date()
                if row_date != self.t0_date:
                    continue # Bỏ qua dữ liệu cũ của ngày khác

            ticker = row[col_ticker]
            
            # Hàm ép kiểu an toàn
            def safe_fl(val): 
                try: return float(val) if pd.notna(val) and val != "" else 0.0
                except: return 0.0

            # Lấy giá trị giao dịch T0 của Khối ngoại
            f_buy_vol = safe_fl(row.get('foreign_buy_volume', 0))
            f_sell_vol = safe_fl(row.get('foreign_sell_volume', 0))
            avg_price = safe_fl(row.get('average_price', row.get('close_price', 0)))
            
            # Quy đổi ra Tỷ VNĐ
            f_net_val_bn = ((f_buy_vol - f_sell_vol) * avg_price) / self.DIVISOR
            
            # Tính toán Cán cân Sổ lệnh (Bid/Ask Imbalance)
            bid_vol = sum([safe_fl(row.get(f'bid_vol_{i}', 0)) for i in range(1, 4)])
            ask_vol = sum([safe_fl(row.get(f'ask_vol_{i}', 0)) for i in range(1, 4)])
            
            t0_dict[ticker] = {
                't0_date': self.t0_date,
                't0_foreign_net_bn': f_net_val_bn,
                't0_bid_vol': bid_vol,
                't0_ask_vol': ask_vol,
                't0_imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0,
                't0_last_price': safe_fl(row.get('close_price', 0)),
                'is_fresh': True
            }
        return t0_dict

    def _build_historical_cube(self):
        """
        KHỐI 1 & 2: ÉP TOÀN BỘ DỮ LIỆU VÀO MỘT MA TRẬN DUY NHẤT
        """
        print("[*] Đang xây dựng Ma trận Lịch sử (Historical Cube)...")
        if self.df_price.empty: return

        # 1. Chuẩn hóa & Lọc 30 phiên gần nhất để nhẹ RAM
        df_p = self.df_price.copy()
        df_p['time'] = pd.to_datetime(df_p['time']).dt.normalize()
        
        if self.past_dates:
            cutoff_date = self.past_dates[-self.lookback_days] if len(self.past_dates) > self.lookback_days else self.past_dates[0]
            df_p = df_p[df_p['time'].dt.date >= cutoff_date]

        # 2. Xử lý Foreign & Prop EOD
        df_f = self.df_foreign.copy() if not self.df_foreign.empty else pd.DataFrame(columns=['ticker', 'time', 'foreign_net_value'])
        df_pr = self.df_prop.copy() if not self.df_prop.empty else pd.DataFrame(columns=['ticker', 'time', 'prop_net_value'])
        
        for df in [df_f, df_pr]:
            if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.normalize()

        # 3. MERGE KHỔNG LỒ (THE GREAT MERGE)
        cube = pd.merge(df_p[['ticker', 'time', 'open', 'high', 'low', 'close', 'volume']], 
                        df_f[['ticker', 'time', 'foreign_net_value']], 
                        on=['ticker', 'time'], how='left')
        cube = pd.merge(cube, 
                        df_pr[['ticker', 'time', 'prop_net_value']], 
                        on=['ticker', 'time'], how='left').fillna(0)

        # 4. TÍNH TOÁN CÁC METRICS ĐỊNH LƯỢNG (Vectorized)
        cube['total_val_bn'] = (cube['close'] * cube['volume']) / self.DIVISOR
        cube['f_net_bn'] = cube['foreign_net_value'] / self.DIVISOR
        cube['p_net_bn'] = cube['prop_net_value'] / self.DIVISOR
        cube['sm_net_bn'] = cube['f_net_bn'] + cube['p_net_bn']

        # 5. NHẬN DIỆN SANG TAY BẰNG DỮ LIỆU SỰ THẬT (GROUND TRUTH)
        cube['vol_ma20'] = cube.groupby('ticker')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        cube['price_spread_pct'] = (cube['high'] - cube['low']) / cube['low'] * 100
        cond_on_book = (cube['volume'] > cube['vol_ma20'] * 3) & (cube['price_spread_pct'] < 2.0)
        
        # Đọc dữ liệu Thỏa thuận truyền vào từ data_frames
        if not self.df_pt.empty:
            # Nhóm tổng giá trị thỏa thuận theo Ngày và Mã
            df_pt_agg = self.df_pt.groupby(['symbol', 'time'])['match_value'].sum().reset_index()
            df_pt_agg = df_pt_agg.rename(columns={'symbol': 'ticker', 'match_value': 'pt_val_total'})
            df_pt_agg['pt_val_bn'] = df_pt_agg['pt_val_total'] / self.DIVISOR
            # Merge vào Ma trận Lịch sử
            cube = pd.merge(cube, df_pt_agg, on=['ticker', 'time'], how='left').fillna({'pt_val_bn': 0})
        else:
            cube['pt_val_bn'] = 0

        # LỌC KÉP: Dùng cả Công thức Heuristic VÀ Dữ liệu Sự thật
        # Nếu Dòng tiền Thể chế > 80% Thanh khoản HOẶC Dòng tiền Thể chế trùng khớp với giá trị Thỏa thuận thật
        cond_off_book_f = (cube['f_net_bn'].abs() > (cube['total_val_bn'] * 0.8)) | ((cube['pt_val_bn'] > 0) & (cube['f_net_bn'].abs() >= cube['pt_val_bn'] * 0.4))
        cond_off_book_p = (cube['p_net_bn'].abs() > (cube['total_val_bn'] * 0.8)) | ((cube['pt_val_bn'] > 0) & (cube['p_net_bn'].abs() >= cube['pt_val_bn'] * 0.4))
        
        cube['is_pt_f'] = np.where(cond_on_book | cond_off_book_f, True, False)
        cube['is_pt_p'] = np.where(cond_on_book | cond_off_book_p, True, False)
        
        # Làm sạch Dòng tiền
        cube['f_net_adj'] = np.where(cube['is_pt_f'], 0, cube['f_net_bn'])
        cube['p_net_adj'] = np.where(cube['is_pt_p'], 0, cube['p_net_bn'])
        cube['sm_net_adj'] = cube['f_net_adj'] + cube['p_net_adj']

        cube['shadow_flow_bn'] = (cube['total_val_bn'] - cube['f_net_adj'].abs() - cube['p_net_adj'].abs()).clip(lower=0)

        # 6. GẮN MÁC RỔ & NGÀNH
        cube['universe'] = cube['ticker'].map(lambda x: self.ticker_to_universe.get(x, 'HOSE'))
        cube['sector'] = cube['ticker'].map(lambda x: self.ticker_to_sector.get(x, 'Unknown'))

        # Chuẩn hóa Date để dễ tra cứu sau này
        cube['date'] = cube['time'].dt.date

        self.history_cube = cube.sort_values(['ticker', 'time']).reset_index(drop=True)
        print(f"[OK] Ma trận hoàn tất: {len(self.history_cube)} records, sẵn sàng cho Inference Engine.")

    def _get_intraday_t0_metrics(self, ticker):
        """Hàm nội bộ: Trích xuất siêu tốc Lực Mua/Bán chủ động T0 từ df_intra"""
        if self.df_intra.empty: return 0, 0
        
        # 1. Lọc đúng mã
        df_i = self.df_intra[self.df_intra['ticker'] == ticker]
        if df_i.empty: return 0, 0

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
        fr_deal_net_bn = safe_num(t1_row.get('fr_buy_volume_deal', 0) - t1_row.get('fr_sell_volume_deal', 0)) * safe_num(t1_row.get('close_price_adjusted', 0)) / self.DIVISOR
        
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
        price_change_pct = (end_price - start_price) / start_price * 100

        # 2. TỔNG HỢP DÒNG TIỀN TRONG KỲ (SỬ DỤNG DỮ LIỆU ĐÃ LÀM SẠCH)
        # Thay thế _bn bằng _adj để không bị nhiễu bởi các cục sang tay ngàn tỷ
        total_sm_net = df_t['sm_net_adj'].sum()
        total_f_net = df_t['f_net_adj'].sum()
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
                diagnosis.append(f"Sóng Thể chế: Khối ngoại & Tự doanh dẫn dắt (Gom {total_sm_net:.1f} Tỷ).")
            elif total_shadow > (total_val * 0.7) and total_sm_net < -10.0:
                diagnosis.append(f"Kéo Xả ảo: Lái nội kéo giá (Ẩn {total_shadow:.1f} Tỷ) để Tay to thoát hàng (Xả {total_sm_net:.1f} Tỷ). ⚠️ Rủi ro Bull-Trap!")
            else:
                diagnosis.append(f"Đồng thuận: Dòng tiền lan tỏa đều, Lái nội chi phối chính ({total_shadow:.1f} Tỷ).")

        elif price_change_pct < -3.0: # XU HƯỚNG GIẢM
            trend = "GIẢM"
            if total_sm_net < -(total_val * 0.15):
                diagnosis.append(f"Tắm Máu Thể chế: Áp lực bán tháo dữ dội từ Tay to (Xả {total_sm_net:.1f} Tỷ).")
            elif total_shadow > (total_val * 0.7) and total_sm_net > 10.0:
                diagnosis.append(f"Đạp Gom (Washout): Nhỏ lẻ hoảng loạn, Lái nội đè giá nhưng Tay to đang âm thầm gom ({total_sm_net:.1f} Tỷ). 🌟 Tín hiệu Rũ bỏ!")
            else:
                diagnosis.append("Phân phối diện rộng, cung cầu suy yếu tự nhiên.")

        else: # TÍCH LŨY (ĐI NGANG)
            trend = "ĐI NGANG"
            if total_sm_net > 20.0:
                diagnosis.append(f"Gom Ngầm (Stealth Accumulation): Neo giá để Tay to gom hàng ({total_sm_net:.1f} Tỷ). Chờ nổ!")
            elif total_sm_net < -20.0:
                diagnosis.append(f"Phân phối Ngầm (Stealth Distribution): Kéo xả trong biên độ hẹp để Tay to thoát hàng ({total_sm_net:.1f} Tỷ).")
            else:
                diagnosis.append("Siết nền (Base Building): Thanh khoản cạn, chờ gió đông.")

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
        ĐỘNG CƠ DỰ BÁO T0 (T0 PREDICTIVE ENGINE)
        Kết hợp: Ngoại T0 (Bảng điện) + Sổ lệnh Imbalance + Lực mua chủ động Intraday
        """
        t0_data = self.t0_snapshot.get(ticker, {})

        # 1. FALLBACK GIÁ TRỊ KHI OFFLINE (Cuối tuần/Chưa mở cửa)
        f_net_t0 = t0_data.get('t0_foreign_net_bn', 0)
        imbalance = t0_data.get('t0_imbalance', 0)
        
        # Lấy giá gần nhất từ Bảng điện, nếu rỗng thì mượn tạm giá EOD từ Lịch sử
        last_price = t0_data.get('t0_last_price', 0)
        if last_price == 0 and not self.df_price.empty:
            df_p_ticker = self.df_price[self.df_price['ticker'] == ticker]
            if not df_p_ticker.empty:
                last_price = df_p_ticker.iloc[-1]['close']
        
        # Bóc tách lệnh chủ động từ Intraday
        net_active_bn, total_intra_val, vwap_t0 = self._get_intraday_t0_metrics(ticker)

        score = 0
        signals = []
        verdict = "NEUTRAL (Giằng co)"

        # Cờ nhận diện trạng thái Thị trường đóng cửa
        is_offline = not t0_data and net_active_bn == 0

        # --- PHẦN 1: ĐÁNH GIÁ VI CẤU TRÚC (MICROSTRUCTURE) ---
        if is_offline:
            signals.append("Thị trường Đóng cửa/Chưa có GD T0")
        else:
            # Khớp lệnh chủ động lớn (Thực tế)
            if net_active_bn > (total_intra_val * 0.1): 
                if imbalance < -0.3:
                    # Kịch bản Vàng: Lái kê bán ảo để đè giá, nhưng lệnh khớp thật lại là MUA XUYÊN TƯỜNG! (Đè Gom / Hấp thụ)
                    signals.append(f"Cầu chủ động ăn vã Bức tường Bán ảo (+{net_active_bn:.1f} Tỷ)")
                    score += 3 # Thưởng điểm cực cao
                else:
                    signals.append(f"Cầu chủ động áp đảo (+{net_active_bn:.1f} Tỷ)")
                    score += 2
            # Áp lực bán chủ động lớn (Thực tế)
            elif net_active_bn < -(total_intra_val * 0.1): 
                if imbalance > 0.3:
                    # Kịch bản Rủi ro: Lái kê mua ảo dụ Nhỏ lẻ, nhưng lại lén XẢ THẲNG VÀO ĐẦU (Phân phối)
                    signals.append(f"Kê Mua ảo dụ cầu, Bán chủ động táng rát ({net_active_bn:.1f} Tỷ)")
                    score -= 3
                else:
                    signals.append(f"Cung chủ động áp đảo ({net_active_bn:.1f} Tỷ)")
                    score -= 2
            # Nếu chưa có Khớp lệnh đáng kể
            else:
                # Thiếu thanh khoản xác nhận
                if imbalance > 0.3:
                    signals.append("Lái kê lệnh Mua (Bid) dày đặc chặn dưới (Thiếu Vol)")
                    score += 1
                elif imbalance < -0.3:
                    signals.append("Lái chặn lệnh Bán (Ask) dày đặc ép giá (Thiếu Vol)")
                    score -= 1

            # Đánh giá Khối Ngoại Real-time (Snapshot)
            if f_net_t0 > 5.0:
                signals.append(f"Tây lông tiếp sức (+{f_net_t0:.1f} Tỷ)")
                score += 2
            elif f_net_t0 < -5.0:
                signals.append(f"Tây lông đang xả rát ({f_net_t0:.1f} Tỷ)")
                score -= 2

            # ĐIỀU KIỆN VWAP
            if vwap_t0 > 0:
                if last_price >= vwap_t0 and net_active_bn > 0:
                    signals.append("Giá neo vững trên VWAP T0")
                    score += 1
                elif last_price < vwap_t0 and net_active_bn < 0:
                    signals.append("Giá thủng VWAP T0, cầu đuối")
                    score -= 1

        # --- PHẦN 2: ĐỐI CHIẾU NGỮ CẢNH T-1 (GHI ĐÈ BẰNG KILL-SWITCH LEVEL 2) ---
        is_shaking_out = False
        l2_trap = False

        if past_context and "error" not in past_context:
            past_verdict = past_context.get('verdict', "")
            l2_trap = past_context.get('has_l2_trap', False)
            l2_data = past_context.get('l2_data', None)
            
            universe = self.ticker_to_universe.get(ticker, 'HOSE')
            if universe == 'VN30': weak_sell_threshold = -15.0
            elif universe == 'VNMidCap': weak_sell_threshold = -3.0
            elif universe == 'VNSmallCap': weak_sell_threshold = -0.5
            else: weak_sell_threshold = -3.0

            is_weak_selling = (net_active_bn >= weak_sell_threshold) or (net_active_bn > -(total_intra_val * 0.05))
            
            if any(keyword in past_verdict for keyword in ["Gom", "Sóng", "Đồng thuận", "Siết nền", "Cá voi"]):
                if score < 0 and is_weak_selling and not l2_trap and not is_offline:
                    is_shaking_out = True
                    
            # KÍCH HOẠT MÁY CHÉM: Nếu T-1 là Bẫy Kéo Xả Ảo
            if l2_trap and l2_data:
                score -= 50 # Ép điểm rớt đài
                signals.insert(0, f"BẪY L2: Lệnh bán thực tế to gấp {1/l2_data['whale_ratio']:.1f}x. Lái kê lệnh ảo dụ Fomo!")
                verdict = "BEARISH (Bẫy Kéo Xả Ảo)"

        # --- PHẦN 3: CHỐT KẾT LUẬN ---
        if verdict == "NEUTRAL (Giằng co)": 
            if is_offline: 
                verdict = "OFFLINE (Chờ Phiên Mới)"
            elif score >= 3: verdict = "🔥 BULLISH (Tích cực - Lái đang kéo giá, Canh Mua)"
            elif score <= -3: verdict = "🩸 BEARISH (Tiêu cực - Áp lực xả lớn, Đứng ngoài/Bán)"
            elif is_shaking_out: verdict = "🟡 NEUTRAL - RŨ BỎ (Bối cảnh tốt, Lái đang đè để gom nốt)"
            elif score > 0: verdict = "🟢 TILT BULL (Nghiêng Mua)"
            elif score < 0: verdict = "🔴 TILT BEAR (Nghiêng Bán)"
            else: verdict = "⚖️ NEUTRAL (Giằng co - Phân phối đều)"

        return {
            "verdict": verdict,
            "last_price": last_price,
            "net_active_bn": net_active_bn,
            "f_net_t0": f_net_t0,
            "imbalance": imbalance,
            "details": " | ".join(signals) if signals else "Chưa có dòng tiền đột biến.",
            # Trả ngược Level 2 data cho live.py/sniper.py đọc để hiển thị chi tiết (nếu cần)
            "l2_data": past_context.get('l2_data', None) if past_context else None, # dữ liệu t-1
            "is_offline": is_offline
        }

if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = 'data/parquet'
    
    def load_pq(sub_path):
        p = Path(DATA_DIR) / sub_path
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    data_frames = {
        'price': load_pq('price/master_price.parquet'),
        'price_l2': load_pq('price/master_price_l2.parquet'),
        'foreign': load_pq('macro/foreign_flow.parquet'),
        'prop': load_pq('macro/prop_flow.parquet'),
        'comp': load_pq('company/master_company.parquet'),
        'idx': load_pq('macro/index_components.parquet'),
        'board': load_pq('board/master_board.parquet'),
        'intra': load_pq('intraday/master_intraday.parquet')
    }

    omni = OmniFlowMatrix(data_frames, lookback_days=30)
    
    test_tickers = ['HPG', 'DIG', 'GMD'] # Thử với VN30, MidCap và SmallCap
    
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
            print(f"    - Ngoại T0       : {now['f_net_t0']:+.1f} Tỷ (Từ Bảng điện Real-time)")
            print(f"    - Sổ lệnh (Bid)  : Mất cân bằng {now['imbalance']:+.2f} (Dương = Kê mua, Âm = Chặn bán)")
            print(f"    - KẾT LUẬN       : 🎯 {now['verdict']}")
            print(f"    - Tín hiệu phụ   : {now['details']}")
        else:
            print(f"    [!] {now['error']}")