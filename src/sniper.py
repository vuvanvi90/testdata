import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path

# Import Hệ sinh thái Động cơ
from src.forecaster import WyckoffForecaster
from src.smart_money import SmartMoneyEngine
from src.market_flow import MarketFlowAnalyzer
from src.shadow_profiler import ShadowProfiler
from src.omni_matrix import OmniFlowMatrix

warnings.filterwarnings('ignore')

class TargetSniper:
    def __init__(self, ticker, data_dir='data/parquet'):
        self.ticker = ticker.upper()
        self.data_dir = Path(data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Kích hoạt Chế độ Bắn tỉa. Đang nạp đạn cho mục tiêu: [ {self.ticker} ]...")
        
        self._load_and_filter_data()
        self._init_engines()

    def _load_parquet_safe(self, path):
        if path.exists():
            try: 
                df = pd.read_parquet(path)
                # LỘT BỎ TIMEZONE (ÉP VỀ NAIVE) NGAY KHI ĐỌC LÊN RAM
                # Ép kiểu và Lột múi giờ an toàn
                if 'time' in df.columns:
                    # 1. Kiểm tra và ép về datetime nếu cột time chưa chuẩn
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        if pd.api.types.is_numeric_dtype(df['time']):
                            # Nếu là số nguyên (mili-giây từ epoch)
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                        else:
                            # Nếu là chuỗi string
                            df['time'] = pd.to_datetime(df['time'])
                    
                    # 2. Lột bỏ Timezone (Ép về Naive)
                    if getattr(df['time'].dt, 'tz', None) is not None:
                        df['time'] = df['time'].dt.tz_localize(None)
                return df
            except Exception as e: 
                print(f"[!] Lỗi đọc file cũ {path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_and_filter_data(self):
        """Chỉ nạp và giữ lại dữ liệu của ĐÚNG MÃ CẦN SOI để chạy với tốc độ ánh sáng"""
        # Đọc Master Data
        df_price_raw = self._load_parquet_safe(self.data_dir / 'price/master_price.parquet')
        df_foreign_raw = self._load_parquet_safe(self.data_dir / 'macro/foreign_flow.parquet')
        df_prop_raw = self._load_parquet_safe(self.data_dir / 'macro/prop_flow.parquet')
        df_intra_raw = self._load_parquet_safe(self.data_dir / 'intraday/master_intraday.parquet')
        df_board_raw = self._load_parquet_safe(self.data_dir / 'board/master_board.parquet')
        df_pt_raw = self._load_parquet_safe(self.data_dir / 'intraday/master_put_through.parquet')
        
        self.df_comp = self._load_parquet_safe(self.data_dir / 'company/master_company.parquet')
        self.df_idx = self._load_parquet_safe(self.data_dir / 'macro/index_components.parquet')

        # Lọc siêu tốc (Chỉ lấy ticker)
        self.df_price = df_price_raw[df_price_raw['ticker'] == self.ticker].copy() if not df_price_raw.empty else pd.DataFrame()
        self.df_foreign = df_foreign_raw[df_foreign_raw['ticker'] == self.ticker].copy() if not df_foreign_raw.empty else pd.DataFrame()
        self.df_prop = df_prop_raw[df_prop_raw['ticker'] == self.ticker].copy() if not df_prop_raw.empty else pd.DataFrame()
        self.df_intra = df_intra_raw[df_intra_raw['ticker'] == self.ticker].copy() if not df_intra_raw.empty else pd.DataFrame()
        
        if not df_board_raw.empty:
            col_name = 'symbol' if 'symbol' in df_board_raw.columns else 'ticker'
            self.df_board = df_board_raw[df_board_raw[col_name] == self.ticker].copy()
        else:
            self.df_board = pd.DataFrame()

        if not df_pt_raw.empty:
            col_name = 'symbol' if 'symbol' in df_pt_raw.columns else 'ticker'
            self.df_pt = df_pt_raw[df_pt_raw[col_name] == self.ticker].copy()
        else:
            self.df_pt = pd.DataFrame()

        # Build Dict (Để tương thích với các Engine)
        self.price_dict = {self.ticker: self.df_price}
        self.foreign_dict = {self.ticker: self.df_foreign}
        self.prop_dict = {self.ticker: self.df_prop}
        self.out_shares_dict = {}
        if not self.df_comp.empty and 'ticker' in self.df_comp.columns and 'issue_share' in self.df_comp.columns:
            self.out_shares_dict = self.df_comp[self.df_comp['ticker'] == self.ticker].set_index('ticker')['issue_share'].to_dict()

        # Trích xuất rổ chỉ số (Universe)
        self.universe = "HOSE"
        if not self.df_idx.empty:
            match = self.df_idx[self.df_idx['ticker'] == self.ticker]
            if not match.empty:
                # Ưu tiên lấy VN30/Mid/Small thay vì HOSE chung chung
                valid_idx = [i for i in match['index_code'].tolist() if i in ['VN30', 'VNMidCap', 'VNSmallCap']]
                if valid_idx: self.universe = valid_idx[0]

    def _init_engines(self):
        """Khởi động toàn bộ vũ khí hạng nặng"""
        self.sm_engine = SmartMoneyEngine(
            foreign_dict=self.foreign_dict, prop_dict=self.prop_dict, 
            out_shares_dict=self.out_shares_dict, price_dict=self.price_dict, universe=self.universe
        )
        self.mf_analyzer = MarketFlowAnalyzer()
        self.shadow_profiler = ShadowProfiler(price_df=self.df_price, verbose=False)
        
        # Băm OmniMatrix
        data_frames = {
            'price': self.df_price, 'foreign': self.df_foreign, 'prop': self.df_prop,
            'comp': self.df_comp, 'idx': self.df_idx, 'board': self.df_board, 
            'intra': self.df_intra, 'put_through': self.df_pt
        }
        self.omni_matrix = OmniFlowMatrix(data_frames, lookback_days=30)

    def _prepare_latest_ohlcv(self):
        """Gộp nến Intraday vào EOD để ra nến cập nhật nhất"""
        df_p = self.df_price.copy()
        if df_p.empty: return df_p
        
        df_p['time'] = pd.to_datetime(df_p['time']).dt.normalize()
        
        if not self.df_intra.empty:
            df_i = self.df_intra.copy()
            df_i['time'] = pd.to_datetime(df_i['time'])
            latest_date = df_i['time'].dt.date.max()
            df_today = df_i[df_i['time'].dt.date == latest_date].sort_values('time')
            
            if not df_today.empty:
                today_candle = pd.DataFrame([{
                    'ticker': self.ticker,
                    'time': pd.to_datetime(latest_date),
                    'open': df_today['price'].iloc[0],
                    'high': df_today['price'].max(),
                    'low': df_today['price'].min(),
                    'close': df_today['price'].iloc[-1],
                    'volume': df_today['volume'].sum()
                }])
                df_combined = pd.concat([df_p, today_candle], ignore_index=True)
                df_combined = df_combined.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                return df_combined
        return df_p

    def _calculate_volume_profile(self, df_ticker_price, lookback_days=130):
        if df_ticker_price.empty: return 0.0, 0.0, 0.0
        df_recent = df_ticker_price.sort_values('time').tail(lookback_days).copy()
        
        current_p = df_recent['close'].iloc[-1]
        bin_size = max(0.1, current_p * 0.005) 
        df_recent['price_bin'] = (df_recent['close'] / bin_size).round() * bin_size
        
        vol_profile = df_recent.groupby('price_bin')['volume'].sum().sort_index()
        if vol_profile.empty: return 0.0, 0.0, 0.0
            
        poc_price = float(vol_profile.idxmax())
        target_vol = vol_profile.sum() * 0.70
        cum_vol = vol_profile.sort_values(ascending=False).cumsum()
        value_area_bins = cum_vol[cum_vol <= target_vol].index
        
        if len(value_area_bins) > 0:
            return poc_price, float(min(value_area_bins)), float(max(value_area_bins))
        return poc_price, poc_price, poc_price

    def _analyze_micro_flow(self):
        """
        X-QUANG CHUYÊN SÂU T-5 ĐẾN T-1: Tích hợp Shadow Flow & Lọc Giao dịch Sang tay
        """
        # LẤY TRỰC TIẾP DATA CUBE TỪ OMNI-MATRIX (ĐÃ TÍNH SẴN MỌI THỨ)
        df_cube = self.omni_matrix.history_cube
        if df_cube.empty: return None

        # Lấy 6 phiên gần nhất của mã này
        df_recent = df_cube[df_cube['ticker'] == self.ticker].tail(6).copy()
        if len(df_recent) < 2: return None

        # Bỏ phiên T0 (dòng cuối cùng) ra khỏi phân tích quá khứ
        df_past = df_recent.iloc[:-1].copy()
        
        # ---------------------------------------------------------
        # TÍNH TOÁN BỐI CẢNH TUẦN DỰA TRÊN DÒNG TIỀN ĐÃ LỌC
        # ---------------------------------------------------------
        weekly_net = df_past['sm_net_adj'].sum()
        
        # Lấy 2 ngày gần nhất (T-2, T-1) để tính gia tốc
        net_t2 = df_past.iloc[-2]['sm_net_adj'] if len(df_past) >= 2 else 0
        net_t1 = df_past.iloc[-1]['sm_net_adj'] if len(df_past) >= 1 else 0

        # Lấy giá trị ĐÃ LỌC để in báo cáo cho logic dễ hiểu
        adj_t1, adj_t2, adj_t3, adj_t4, adj_t5 = 0, 0, 0, 0, 0
        vals_adj = df_past['sm_net_adj'].tolist()
        if len(vals_adj) >= 5: adj_t5, adj_t4, adj_t3, adj_t2, adj_t1 = vals_adj[-5:]

        # ---------------------------------------------------------
        # PHÂN TÍCH SÂU (SHADOW FLOW & INSTITUTIONAL CONFLICT)
        # ---------------------------------------------------------
        avg_dominance = (df_past['f_net_bn'].abs() + df_past['p_net_bn'].abs()).mean() / df_past['total_val_bn'].mean() * 100 if df_past['total_val_bn'].mean() > 0 else 0
        
        # Đọc vị Hành vi T-1 (Ngày hôm qua)
        t1_row = df_past.iloc[-1]
        t1_f = t1_row['f_net_bn']
        t1_p = t1_row['p_net_bn']
        t1_shadow = t1_row['shadow_flow_bn']
        
        pattern = "GIẰNG CO (Dòng tiền tuần không rõ ràng)"
        thesis = "NEUTRAL"
        special_warning = ""

        # 🚨 BẮT BẪY 1: XUNG ĐỘT THỂ CHẾ (Khối ngoại xả, Tự doanh đỡ hoặc ngược lại)
        if (t1_f * t1_p < 0) and abs(t1_f) > 15 and abs(t1_p) > 15:
            special_warning = f"⚠️ XUNG ĐỘT TỶ ĐÔ: Ngoại ({t1_f:+.1f}) đánh nhau với Nội ({t1_p:+.1f}). Rủi ro Volatility cực cao!"

        # 🚨 BẮT BẪY 2: SÓNG LÁI NỘI ĐẦU CƠ (Tay to đứng ngoài, Shadow Flow áp đảo)
        elif avg_dominance < 5.0 and t1_shadow > 50:
            special_warning = f"🎭 SÓNG ĐẦU CƠ: Tay to vắng mặt (Chi phối {avg_dominance:.1f}%). Lái nội tự quay tay ({t1_shadow:.1f} Tỷ)!"
            thesis = "NEUTRAL (Cẩn thận bẫy thanh khoản)"

        # 🚨 BẮT BẪY 3: CẢNH BÁO SANG TAY ĐỘC LẬP
        pt_f_days = df_past[df_past['is_pt_f']]['time'].dt.strftime('%d/%m').tolist()
        pt_p_days = df_past[df_past['is_pt_p']]['time'].dt.strftime('%d/%m').tolist()
        
        if pt_f_days or pt_p_days:
            msgs = []
            if pt_f_days: msgs.append(f"Ngoại ({', '.join(pt_f_days)})")
            if pt_p_days: msgs.append(f"Nội ({', '.join(pt_p_days)})")
            special_warning = f"⚠️ LỌC SANG TAY: Đã loại bỏ nhiễu của {' & '.join(msgs)}."

        # LOGIC ĐỌC VỊ NHƯ CŨ (Kết hợp với Dữ liệu đã làm sạch)
        if weekly_net > 5.0: 
            if net_t2 > 0 and net_t1 > 0:
                pattern, thesis = "GOM QUYẾT LIỆT", "BULLISH"
            elif net_t2 < 0 and net_t1 > 0:
                pattern, thesis = "RŨ BỎ THÀNH CÔNG", "BULLISH"
            elif net_t2 < 0 and net_t1 <= 0 and abs(net_t1) < abs(net_t2):
                pattern, thesis = "RŨ BỎ CẠN CUNG", "BOTTOMING"
            elif net_t2 < 0 and net_t1 < 0 and abs(net_t1) > abs(net_t2):
                pattern, thesis = "PHÂN PHỐI LƯỚT SÓNG", "BEARISH"
            else:
                pattern, thesis = f"GOM NHẶT ẨN MÌNH (+{weekly_net:.1f})", "TILT BULL"

        elif weekly_net < -5.0:
            if net_t2 < 0 and net_t1 < 0:
                pattern, thesis = "TẮM MÁU/THÁO CỐNG", "BEARISH"
            elif net_t2 > 0 and net_t1 >= 0 and net_t1 > net_t2:
                pattern, thesis = "CẦU BẮT ĐÁY", "BOTTOMING"
            elif net_t2 > 0 and net_t1 < 0:
                pattern, thesis = "BULL-TRAP NGẮN HẠN", "BEARISH"
            else:
                pattern, thesis = f"XẢ RẢI RÁC ({weekly_net:.1f})", "TILT BEAR"
                
        # Nối cảnh báo đặc biệt vào Pattern nếu có
        if special_warning:
            pattern = f"{special_warning} | {pattern}"

        return {
            "t5": adj_t5, "t4": adj_t4, "t3": adj_t3, "t2": adj_t2, "t1": adj_t1,
            "weekly_net": weekly_net,
            "pattern": pattern, "thesis": thesis
        }

    def analyze(self):
        """Khớp nối dữ liệu và Báo cáo X-Quang"""
        df_full = self._prepare_latest_ohlcv()
        if df_full.empty:
            print(f"[!] Không có dữ liệu giá cho {self.ticker}.")
            return

        # 1. KỸ THUẬT (Wyckoff)
        forecaster = WyckoffForecaster(data_input=df_full, output_dir=None, run_date=datetime.now(), verbose=False)
        report = forecaster.run_forecast()
        if report.empty:
            print(f"[!] Không thể vẽ đồ thị Wyckoff cho {self.ticker}.")
            return
            
        wyckoff_row = report.iloc[-1]
        price = wyckoff_row['Price']
        signal = wyckoff_row['Signal']
        vpa = wyckoff_row.get('VPA_Status', 'N/A')
        poc, val, vah = self._calculate_volume_profile(df_full)

        def safe_float(val, default=0.0):
            if pd.isna(val): return float(default)
            val_str = str(val).strip().upper()
            if val_str in ["", "ATO", "ATC", "NULL", "NONE"]:
                return float(default)
            try:
                return float(val)
            except ValueError:
                return float(default)

        # 2. DÒNG TIỀN (Smart Money & Market Flow)
        board_info = None
        if not self.df_board.empty:
            row = self.df_board.iloc[0]
            board_info = {
                'best_bid': safe_float(row.get('bid_price_1', row.get('close_price', price))),
                'best_ask': safe_float(row.get('ask_price_1', row.get('close_price', price)))
            }

        sm_result = self.sm_engine.analyze_ticker(self.ticker, board_info)
        mf_result = self.mf_analyzer.analyze_flow(self.ticker, df_full, self.df_foreign, self.df_prop, self.df_pt)
        sm_vwap = mf_result.get('sm_vwap', 0)
        dtl = mf_result.get('dtl_days', 0)

        # 3. LÁI NỘI & QUÁ KHỨ (Shadow Profiler & Omni Past)
        past_ctx = self.omni_matrix.explain_past_movement(self.ticker, lookback_days=10)
        rules = self.shadow_profiler.build_criminal_profile([self.ticker], lookback_days=250)
        alerts = self.shadow_profiler.live_shadow_radar([self.ticker], rules)
        shadow_msg = alerts[0]['Note'] if alerts else "Không phát hiện dấu vết ủ mưu."

        # 4. DỰ BÁO T0 (Omni Now)
        omni_now = self.omni_matrix.predict_t0_action(self.ticker, past_context=past_ctx)

        # --- IN BÁO CÁO SNIPER DASHBOARD ---
        print("\n" + "═"*90)
        print(f" 🎯 BÁO CÁO X-QUANG ĐỘC LẬP (SNIPER MODE) | MÃ: [ {self.ticker} ] - RỔ: {self.universe}")
        print("═"*90)
        
        # PHẦN 1
        print(f" 📈 1. KỸ THUẬT & CẤU TRÚC GIÁ (WYCKOFF)")
        print(f"    - Thị giá hiện tại : {price:,.0f} đ")
        print(f"    - Tín hiệu Wyckoff : {signal} (Khối lượng: {vpa})")
        print(f"    - Khung Giá Trị    : VAL {val:,.0f} -> Cân bằng POC {poc:,.0f} -> VAH {vah:,.0f}")
        
        if price < val: print("      => Đánh giá: Đang ở Vùng Giá Rẻ (Oversold). Ưu tiên tìm đáy/SPRING.")
        elif price > vah: print("      => Đánh giá: Đã Bứt phá vùng giá trị (Blue Sky). Ưu tiên đà tăng/SOS.")
        else: print("      => Đánh giá: Đang Tích lũy trong hộp. Ưu tiên test cung cạn kiệt.")

        print("-" * 90)
        
        # PHẦN 2
        print(f" 🐋 2. DÒNG TIỀN THỂ CHẾ (SMART MONEY & MARKET FLOW)")
        print(f"    - Điểm Đồng thuận  : {sm_result.get('total_sm_score', 0)}/100")
        print(f"    - Giá vốn Tay to   : ~{sm_vwap:,.0f} đ (VWAP)")
        print(f"    - Sức ép Xả (DTL)  : {dtl:.1f} ngày")
        
        if sm_result.get("is_danger", False):
            print(f"    - Báo động Đỏ      : 🚨 {' | '.join(sm_result['warnings'])}")
        elif sm_result.get("sm_details"):
            print(f"    - Hành vi Nổi bật  : {' | '.join(sm_result['sm_details'])}")
            
        print(f"    - Radar Lái nội    : {shadow_msg}")
        print("-" * 90)

        # GỌI MODULE MICRO-FLOW T-3 ĐẾN T-1
        micro_flow = self._analyze_micro_flow()

        validation = "CHƯA XÁC NHẬN"
        
        # PHẦN 3: ĐỘNG LƯỢNG HIỆN TẠI & KIỂM ĐỊNH GIẢ THUYẾT
        print(f" ⚡ 3. ĐỘNG LƯỢNG KẾT HỢP (MICRO-FLOW VALIDATION)")
        if micro_flow:
            print(f"    [QUÁ KHỨ GẦN - CHU KỲ 1 TUẦN]")
            print(f"    - Dòng tiền T-5 -> T-1   : {micro_flow['t5']:+.1f} | {micro_flow['t4']:+.1f} | {micro_flow['t3']:+.1f} | {micro_flow['t2']:+.1f} | {micro_flow['t1']:+.1f} (Tỷ VNĐ)")
            print(f"    - TỔNG KẾT TUẦN (Weekly) : {micro_flow['weekly_net']:+.1f} Tỷ")
            print(f"    - Hành vi Lái (Pattern)  : {micro_flow['pattern']}")
            print(f"    - Giả thuyết T0 (Thesis) : {micro_flow['thesis']}")
        
        print(f"    [HIỆN TẠI T0]")
        if "error" not in omni_now:
            print(f"    - Khớp chủ động T0       : {omni_now['net_active_bn']:+.1f} Tỷ")
            print(f"    - Sổ lệnh (Bid/Ask)      : Mất cân bằng {omni_now['imbalance']:+.2f}")

            # 🚀 ĐỘNG CƠ KIỂM ĐỊNH (VALIDATOR)
            t0_active = omni_now['net_active_bn']
            if micro_flow:
                # 1. KỊCH BẢN BỐI CẢNH TÍCH CỰC (GOM HÀNG)
                if any(k in micro_flow['thesis'] for k in ["BULLISH", "BOTTOMING", "TILT BULL"]):
                    if t0_active > 3.0: 
                        validation = "✅ XÁC NHẬN ĐÚNG (Tiền vào chủ động như dự kiến)"
                    elif t0_active < -3.0: 
                        validation = "❌ XÁC NHẬN SAI (Bị xả ngược, Giả thuyết gãy)"
                    else:
                        # Nằm trong khoảng -3 đến +3 là trạng thái Cạn Cung chờ nổ
                        validation = "✅ XÁC NHẬN ĐÚNG (Tiết Cung T0, Dòng tiền đang nén chặt)"
                
                # 2. KỊCH BẢN BỐI CẢNH TIÊU CỰC (XẢ HÀNG)
                elif any(k in micro_flow['thesis'] for k in ["BEARISH", "TILT BEAR"]):
                    if t0_active < -3.0: 
                        validation = "✅ XÁC NHẬN ĐÚNG (Tiếp tục bị đè bán)"
                    elif t0_active > 5.0: 
                        validation = "🔄 ĐẢO CHIỀU CẦU (Lực mua bất ngờ giải cứu)"
                    else:
                        validation = "⚠️ TẠM DỪNG XẢ (Lực bán yếu đi nhưng chưa có Cầu vào)"
                    
            print(f"    => KIỂM ĐỊNH T0          : {validation}")
            print(f"    => KẾT LUẬN OMNI T0      : {omni_now['verdict']}")
        else:
            print(f"    - [!] {omni_now['error']}")

        print("═"*90)

        # Phần 4: Kiểm toán Thỏa thuận
        print("═"*90)
        print(f" 🤝 4. KIỂM TOÁN GIAO DỊCH THỎA THUẬN (OFF-BOOK AUDIT)")
        if not self.df_pt.empty:
            
            # 1. LẬP BẢN ĐỒ TIMELINE (T-5 -> T0)
            # trading_days = sorted(self.df_price['time'].unique())[-6:]

            # XÁC ĐỊNH CHUẨN PHIÊN GIAO DỊCH T-5 ĐẾN T0
            # Lấy 5 ngày giao dịch cũ từ df_price
            past_sessions = sorted(self.df_price['time'].unique())
            if len(past_sessions) >= 5:
                # Nếu ngày cuối cùng của df_price chính là T0 (sau khi kết thúc phiên)
                if past_sessions[-1].date() == self.omni_matrix.t0_date:
                    trading_days = past_sessions[-6:] # Lấy đủ 6 ngày cuối
                else:
                    # Nếu đang trong phiên (df_price mới đến T-1), lấy 5 ngày cũ + T0 hệ thống
                    t0_timestamp = pd.Timestamp(self.omni_matrix.t0_date)
                    trading_days = past_sessions[-5:] + [t0_timestamp]
            else:
                trading_days = past_sessions # Phòng hờ dữ liệu quá ít

            pt_vals = []
            pt_intents = []
            
            for d in trading_days:
                df_day = self.df_pt[self.df_pt['time'] == d].copy()
                if not df_day.empty:
                    val_bn = df_day['match_value'].sum() / 1_000_000_000
                    # Tính % Premium/Discount gia quyền theo Volume của riêng ngày đó
                    df_day['weighted_change'] = df_day['change_percent'] * df_day['match_value']
                    w_change = (df_day['weighted_change'].sum() / df_day['match_value'].sum()) * 100
                    
                    pt_vals.append(f"{val_bn:+.1f}")
                    if w_change > 1.0: pt_intents.append("🔥")     # Premium (Khát hàng)
                    elif w_change < -1.0: pt_intents.append("🧊")  # Discount (Giải chấp/Nội bộ)
                    else: pt_intents.append("⚖️")                  # Neutral (Quanh tham chiếu)
                else:
                    # Nếu ngày đó không có thỏa thuận
                    pt_vals.append(f"{0.0:+.1f}")
                    pt_intents.append(" -")

            print(f"    [LỊCH TRÌNH 6 PHIÊN (T-5 -> T0)]")
            print(f"    - Sang tay (Tỷ VNĐ) : {' | '.join(pt_vals)}")
            print(f"    - Ý đồ (Intent)     : {'  |  '.join(pt_intents)}  (🔥 Đắt / 🧊 Rẻ / ⚖️ Tham chiếu)")
            print(f"    --------------------------------------------------------------------------------------")

            # 2. TỔNG KẾT TÌM ĐIỂM NEO GIÁ 30 NGÀY (ANCHOR FINDER)
            cutoff_pt = datetime.now() - pd.Timedelta(days=30)
            df_pt_recent = self.df_pt[self.df_pt['time'] >= cutoff_pt].copy()
            
            if not df_pt_recent.empty:
                total_pt_val = df_pt_recent['match_value'].sum() / 1_000_000_000
                vwap_pt = (df_pt_recent['price'] * df_pt_recent['volume']).sum() / df_pt_recent['volume'].sum()
                
                df_pt_recent['weighted_change'] = df_pt_recent['change_percent'] * df_pt_recent['match_value']
                avg_change = (df_pt_recent['weighted_change'].sum() / df_pt_recent['match_value'].sum()) * 100

                print(f"    [TỔNG KẾT VÙNG NEO GIÁ (30D)]")
                print(f"    - Tổng quy mô Trao tay  : {total_pt_val:.1f} Tỷ VNĐ")
                print(f"    - Giá vốn Thỏa thuận    : ~{vwap_pt:,.0f} đ")
                
                if avg_change > 1.0:
                    intent = "🔥 PREMIUM (Khát hàng, Chấp nhận mua đắt -> Siêu Bullish)"
                elif avg_change < -1.0:
                    intent = "🧊 DISCOUNT (Trao tay giá rẻ -> Thường là Sang tay nội bộ/Né thuế)"
                else:
                    intent = "⚖️ NEUTRAL (Trao tay quanh giá tham chiếu)"
                    
                print(f"    - Ý đồ Tổng thể (Intent): {intent}")
                
                # Hợp lưu Mua (Chỉ coi là Phòng tuyến nếu Thỏa thuận > 50 Tỷ)
                if total_pt_val > 50.0:
                    print(f"    => 🛡️ TỌA ĐỘ PHÒNG THỦ   : Nếu giá rớt về sát {vwap_pt:,.0f}đ sẽ kích hoạt Lực đỡ Khổng lồ!")

                # TÍCH HỢP 4 BƯỚC PHÂN TÍCH CHUYÊN SÂU (PUT-THROUGH EDGE)
                print(f"    --------------------------------------------------------------------------------------")
                print(f"    [PHÂN TÍCH CHUYÊN SÂU 4 CHIỀU (PUT-THROUGH EDGE)]")
                
                # 1. CỤM MỨC GIÁ (WHALE NODE)
                whale_node = df_pt_recent.groupby('price')['volume'].sum().idxmax()
                print(f"    - Cụm Giá Tập Trung (Node)     : {whale_node:,.0f} đ (Vùng Nam châm hút giá lớn nhất 30D)")

                # 2. CƯỜNG ĐỘ THANH KHOẢN (LIQUIDITY INTENSITY T0)
                avg_vol_20d = self.df_price['volume'].tail(20).mean() if not self.df_price.empty else 0
                t0_pt = df_pt_recent[df_pt_recent['time'].dt.date == self.omni_matrix.t0_date].copy()
                pt_vol_today = t0_pt['volume'].sum()
                
                if avg_vol_20d > 0 and pt_vol_today > 0:
                    liqd_ratio = pt_vol_today / avg_vol_20d
                    print(f"    - Cường độ Ẩn T0 (Liqd Ratio)  : {liqd_ratio:.2f}x so với Khớp lệnh MA20")
                    if liqd_ratio > 2.0:
                        print("      => 🚨 CẢNH BÁO ĐỘT BIẾN: Thỏa thuận bùng nổ! Rủi ro/Cơ hội thay máu Cổ đông lớn.")
                else:
                    print("    - Cường độ Ẩn T0 (Liqd Ratio)  : Không có thỏa thuận trong phiên hôm nay")

                # 3. CHU KỲ TẦN SUẤT (PULSE ANALYSIS)
                # Chỉ lọc các giao dịch LỚN (Định nghĩa: > 10 Tỷ VNĐ) để đo nhịp tim
                large_pts = df_pt_recent[df_pt_recent['match_value'] > 10_000_000_000]
                if not large_pts.empty:
                    unique_dates = sorted(large_pts['time'].dt.date.unique())
                    if len(unique_dates) >= 2:
                        diffs = [(unique_dates[i] - unique_dates[i-1]).days for i in range(1, len(unique_dates))]
                        avg_pulse = sum(diffs) / len(diffs)
                        print(f"    - Chu kỳ Nhịp đập (Pulse)      : ~{avg_pulse:.1f} ngày / một nhịp Thỏa thuận LỚN (>10 Tỷ)")
                    else:
                        print("    - Chu kỳ Nhịp đập (Pulse)      : Chưa đủ số phiên để đo nhịp điệu Tay to")
                else:
                    print("    - Chu kỳ Nhịp đập (Pulse)      : Không có thỏa thuận Khủng (>10 Tỷ) nào")

                # 4. ĐỘ LỆCH PHA THỜI GIAN (TIMING DIVERGENCE T0)
                if not t0_pt.empty and not self.df_price.empty:
                    t0_price_row = self.df_price.iloc[-1] 
                    # Tính biến động giá trên bảng điện (Đóng cửa vs Mở cửa)
                    price_change_today = (t0_price_row['close'] - t0_price_row['open']) / t0_price_row['open'] * 100 if t0_price_row['open'] > 0 else 0
                    
                    t0_pt['weighted_change'] = t0_pt['change_percent'] * t0_pt['match_value']
                    t0_avg_change = (t0_pt['weighted_change'].sum() / t0_pt['match_value'].sum()) * 100
                    
                    print(f"    - Lệch pha T0 (Timing Diverge) : Giá trên sàn {price_change_today:+.2f}% vs Thỏa thuận T0 {t0_avg_change:+.2f}%")
                    
                    if price_change_today < -1.0 and t0_avg_change > 1.0:
                        print("      => 🌟 SIÊU TÍN HIỆU ĐẢO CHIỀU: Đạp sàn ép nhỏ lẻ, Thỏa thuận giá cao (Gom ngầm)!")
                    elif price_change_today > 1.0 and t0_avg_change < -1.0:
                        print("      => ⚠️ TÍN HIỆU BULL-TRAP: Kéo thốc trên sàn dụ FOMO, Thỏa thuận giá bèo táng nội bộ!")
            else:
                print("    - Không phát hiện GD Thỏa thuận nào đáng kể trong 30 ngày qua.")
        else:
            print("    - Không có dữ liệu Giao dịch Thỏa thuận.")
        print("═"*90)

        # PHẦN 5: KẾT LUẬN ĐẦU TƯ (CẬP NHẬT THEO MICRO-FLOW)
        print(" 🎯 HÀNH ĐỘNG KHUYẾN NGHỊ (SNIPER FINAL VERDICT):")
        
        is_validated_bull = "XÁC NHẬN ĐÚNG" in validation and ("BULLISH" in micro_flow['thesis'] or "BOTTOMING" in micro_flow['thesis'] or "TILT BULL" in micro_flow['thesis'])
        
        if sm_result.get("is_danger", False) and not is_validated_bull:
            print("   🚫 KHÔNG MUA: Cá mập dài hạn đang tháo cống. Bỏ qua mọi tín hiệu kỹ thuật.")
        elif "BEARISH" in micro_flow['thesis'] and "XÁC NHẬN ĐÚNG" in validation:
            print("   🩸 KHÔNG MUA (ĐANG TẮM MÁU): Áp lực xả từ tuần trước vẫn tiếp diễn ở T0. Bắt đáy là cụt tay!")
        elif "BOTTOMING" in micro_flow['thesis'] and "XÁC NHẬN SAI" in validation:
            print("   ⚠️ HỦY KẾ HOẠCH BẮT ĐÁY: T-1 đã tiết cung, nhưng T0 lái lại tiếp tục táng hàng. Quá rủi ro!")
        elif "BULLISH" in micro_flow['thesis'] and "XÁC NHẬN SAI" in validation:
            print("   ⚠️ HỦY LỆNH MUA (FADE THE BREAKOUT): Quá khứ gom đẹp nhưng T0 bị xả ngược. Lái tạo Bull-trap!")
        
        # 🚀 TÁCH RIÊNG 2 TRƯỜNG HỢP MUA
        elif is_validated_bull and signal in ['SOS', 'SPRING', 'TEST_CUNG', 'NEUTRAL']:
            sl_price = price - (wyckoff_row.get('ATR', price * 0.02) * 2.5)
            
            if "Tiết Cung T0" in validation:
                print(f"   🌱 GOM VỊ THẾ NỀN (ACCUMULATE): Quá khứ dọn đường + T0 Cạn cung chờ gió Đông!")
                buy_p = board_info['best_bid'] # Mua kê giá thấp
                print(f"      => LỆNH BẮN TỈA: Kê mua rải đinh quanh {buy_p:,.0f} đ | Dừng lỗ cứng tại {sl_price:,.0f} đ.")
            else:
                print(f"   🌟 ĐIỂM NỔ BỨT PHÁ (SNIPER TRIGGER): Quá khứ dọn đường + T0 Tiền vào xác nhận!")
                buy_p = board_info['best_ask'] if board_info['best_ask'] > 0 else price # Mua chủ động
                print(f"      => LỆNH BẮN TỈA: Mua quét lên (Market) quanh {buy_p:,.0f} đ | Dừng lỗ cứng tại {sl_price:,.0f} đ.")
                
            if price < sm_vwap:
                print(f"      => LỢI THẾ KÉP: Giá ({price:,.0f}) đang rẻ hơn giá vốn Cá mập ({sm_vwap:,.0f}).")
                
        else:
            print(f"   ☕ CHỜ ĐỢI: Dòng tiền đang giằng co. Hành vi Lái chưa rõ ràng. Đưa vào Watchlist theo dõi T0.")

        print("═"*90)

if __name__ == "__main__":
    while True:
        ticker_input = input("\nNhập mã cổ phiếu muốn đưa vào Tầm ngắm Bắn tỉa (VD: FTS) hoặc gõ 'Q' để thoát: ").strip()
        if ticker_input.upper() == 'Q':
            break
        if len(ticker_input) >= 3:
            sniper = TargetSniper(ticker=ticker_input)
            sniper.analyze()