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
        self.df_foreign = data_frames.get('foreign', pd.DataFrame())
        self.df_prop = data_frames.get('prop', pd.DataFrame())
        self.df_comp = data_frames.get('comp', pd.DataFrame())
        self.df_idx = data_frames.get('idx', pd.DataFrame())
        self.df_board = data_frames.get('board', pd.DataFrame())
        self.df_intra = data_frames.get('intra', pd.DataFrame())
        
        self.lookback_days = lookback_days
        self.DIVISOR = 1_000_000_000 # Quy đổi ra Tỷ VNĐ
        
        self.t0_date = self._identify_system_t0()

        # 2. Xây dựng Bản đồ (Mapping) O(1)
        self.ticker_to_universe = {}
        self.ticker_to_sector = {}
        self._build_mappings()
        
        # 3. Trích xuất Snapshot T0 từ Bảng điện (Ý TƯỞNG CỦA ANH VÀO ĐÂY)
        self.t0_snapshot = self._extract_t0_snapshot()
        
        # 4. Xây dựng Khối dữ liệu Lịch sử (Historical Data Cube)
        self.history_cube = pd.DataFrame()
        self._build_historical_cube()

    def _identify_system_t0(self):
        """Xác định ngày giao dịch hiện tại thực tế nhất từ Intraday hoặc Price"""
        if not self.df_intra.empty:
            return self.df_intra['time'].dt.date.max()
        if not self.df_price.empty:
            return self.df_price['time'].dt.date.max()
        return datetime.now().date()

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
        
        # Cắt lấy danh sách ngày (lấy 30 ngày cuối)
        trading_days = sorted(df_p['time'].unique())
        if len(trading_days) > self.lookback_days:
            cutoff_date = trading_days[-self.lookback_days]
            df_p = df_p[df_p['time'] >= cutoff_date]

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
        
        # Shadow Flow (Dòng tiền Ẩn/Lái nội) = Tổng thanh khoản - |Ngoại| - |Nội|
        # (Lấy trị tuyệt đối vì Lái có thể tham gia vào cả bên bán của Tây hoặc bên mua của Tây)
        cube['shadow_flow_bn'] = (cube['total_val_bn'] - cube['f_net_bn'].abs() - cube['p_net_bn'].abs()).clip(lower=0)
        
        # 5. NHẬN DIỆN SANG TAY (PUT-THROUGH ANOMALY)
        # Tính MA20 của Volume (Cần groupby)
        cube['vol_ma20'] = cube.groupby('ticker')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        cube['price_spread_pct'] = (cube['high'] - cube['low']) / cube['low'] * 100
        
        # Điều kiện Sang tay: Vol > 300% MA20 NHƯNG Biên độ giá < 2%
        cube['is_put_through'] = np.where(
            (cube['volume'] > cube['vol_ma20'] * 3) & (cube['price_spread_pct'] < 2.0),
            True, False
        )

        # 6. GẮN MÁC RỔ & NGÀNH
        cube['universe'] = cube['ticker'].map(lambda x: self.ticker_to_universe.get(x, 'HOSE'))
        cube['sector'] = cube['ticker'].map(lambda x: self.ticker_to_sector.get(x, 'Unknown'))

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

        if df_today.empty: return 0, 0

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

    def explain_past_movement(self, ticker, lookback_days=10):
        """
        ĐỘNG CƠ GIẢI THÍCH QUÁ KHỨ (RETROSPECTIVE ENGINE)
        Phân tích đa chiều: Giá vs Dòng tiền Thể chế vs Dòng tiền Lái nội
        """
        df_t = self.history_cube[self.history_cube['ticker'] == ticker].tail(lookback_days)
        if df_t.empty: return {"error": "Không có dữ liệu lịch sử"}

        # 1. Tính toán Biến động Giá
        start_price = df_t['close'].iloc[0]
        end_price = df_t['close'].iloc[-1]
        price_change_pct = (end_price - start_price) / start_price * 100

        # 2. Tổng hợp Dòng tiền trong kỳ
        total_sm_net = df_t['sm_net_bn'].sum()
        total_f_net = df_t['f_net_bn'].sum()
        total_shadow = df_t['shadow_flow_bn'].sum()
        total_val = df_t['total_val_bn'].sum()

        # 3. Quét các ngày Sang tay (Put-through)
        put_through_days = df_t[df_t['is_put_through']]['time'].dt.strftime('%d/%m').tolist()

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

        if put_through_days:
            diagnosis.append(f"⚠️ Sang tay ngầm (Trao kho) vào: {', '.join(put_through_days)}.")

        return {
            "trend": trend,
            "change_pct": price_change_pct,
            "sm_net": total_sm_net,
            "f_net": total_f_net,
            "shadow": total_shadow,
            "verdict": " | ".join(diagnosis)
        }

    def predict_t0_action(self, ticker, past_context=None):
        """
        ĐỘNG CƠ DỰ BÁO T0 (T0 PREDICTIVE ENGINE)
        Kết hợp: Ngoại T0 (Bảng điện) + Sổ lệnh Imbalance + Lực mua chủ động Intraday
        """
        t0_data = self.t0_snapshot.get(ticker, {})
        if not t0_data: return {"error": "Không có dữ liệu Bảng điện T0."}

        # Lấy dữ liệu T0
        f_net_t0 = t0_data.get('t0_foreign_net_bn', 0)
        imbalance = t0_data.get('t0_imbalance', 0)
        last_price = t0_data.get('t0_last_price', 0)
        
        # Bóc tách lệnh chủ động từ Intraday
        net_active_bn, total_intra_val, vwap_t0 = self._get_intraday_t0_metrics(ticker)

        score = 0
        signals = []

        # --- PHẦN 1: ĐÁNH GIÁ VI CẤU TRÚC (MICROSTRUCTURE) ---
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

        # --- PHẦN 2: ĐỐI CHIẾU NGỮ CẢNH QUÁ KHỨ (CONTEXT OVERRIDE) ---
        is_shaking_out = False
        if past_context:
            past_verdict = past_context.get('verdict', "")
            
            # NGƯỠNG ĐỘNG THEO RỔ VỐN HÓA
            universe = self.ticker_to_universe.get(ticker, 'HOSE')
            
            if universe == 'VN30':
                weak_sell_threshold = -15.0  # VN30: Xả dưới 15 Tỷ vẫn là nhiễu nhỏ lẻ
            elif universe == 'VNMidCap':
                weak_sell_threshold = -3.0   # MidCap: Xả dưới 3 Tỷ là nhiễu
            elif universe == 'VNSmallCap':
                weak_sell_threshold = -0.5   # Penny: Xả dưới 500 triệu (0.5 Tỷ) là nhiễu. Lớn hơn là CÓ BIẾN!
            else:
                weak_sell_threshold = -3.0   # Mặc định

            # Lực xả được coi là "YẾU" (Không đáng ngại) nếu nó thỏa mãn 1 trong 2 điều kiện:
            # 1. Nhỏ hơn mức chịu đựng tuyệt đối của Rổ (Tuyệt đối)
            # 2. Hoặc chiếm chưa tới 5% tổng thanh khoản hiện tại (Tương đối)
            is_weak_selling = (net_active_bn >= weak_sell_threshold) or (net_active_bn > -(total_intra_val * 0.05))
            
            # Nếu quá khứ đang là Sóng Gom HOẶC đang Siết nền cạn kiệt, và lực xả hiện tại chỉ là "Xả Yếu"
            if any(keyword in past_verdict for keyword in ["Gom Ngầm", "Sóng Thể chế", "Đồng thuận", "Siết nền"]):
                if score < 0 and is_weak_selling:
                    is_shaking_out = True
                    score = 0 # Đưa về Neutral để không báo bán sai

        # ĐIỀU KIỆN VWAP
        if vwap_t0 > 0:
            if last_price >= vwap_t0 and net_active_bn > 0:
                signals.append("Giá neo vững trên VWAP T0")
                score += 1
            elif last_price < vwap_t0 and net_active_bn < 0:
                signals.append("Giá thủng VWAP T0, cầu đuối")
                score -= 1

        # --- PHẦN 3: CHỐT KẾT LUẬN ---
        if score >= 3: verdict = "🔥 BULLISH (Tích cực - Lái đang kéo giá, Canh Mua)"
        elif score <= -3: verdict = "🩸 BEARISH (Tiêu cực - Áp lực xả lớn, Đứng ngoài/Bán)"
        elif is_shaking_out: verdict = "🟡 NEUTRAL - RŨ BỎ (Bối cảnh tốt, Lái đang đè lệnh ảo để gom nốt)"
        elif score > 0: verdict = "🟢 TILT BULL (Hơi nghiêng về chiều Mua)"
        elif score < 0: verdict = "🔴 TILT BEAR (Hơi nghiêng về chiều Bán)"
        else: verdict = "⚖️ NEUTRAL (Giằng co - Phân phối đều)"

        return {
            "verdict": verdict,
            "last_price": last_price,
            "net_active_bn": net_active_bn,
            "f_net_t0": f_net_t0,
            "imbalance": imbalance,
            "details": " | ".join(signals) if signals else "Chưa có dòng tiền đột biến."
        }

if __name__ == "__main__":
    from pathlib import Path
    DATA_DIR = 'data/parquet'
    
    def load_pq(sub_path):
        p = Path(DATA_DIR) / sub_path
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    data_frames = {
        'price': load_pq('price/master_price.parquet'),
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
            print(f" 🕰️ GIẢI MÃ QUÁ KHỨ (10 Phiên):")
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