import json
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
    def __init__(self, ticker, target_date=None, data_dir='data/parquet'):
        self.ticker = ticker.upper()
        self.data_dir = Path(data_dir)
        self.target_date = pd.to_datetime(target_date).normalize() if target_date else None

        date_str = self.target_date.strftime('%Y-%m-%d') if self.target_date else "LIVE NOW"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Kích hoạt Bắn tỉa V3.0 (Time-Travel). Mục tiêu: [ {self.ticker} ] | Khung thời gian: {date_str}")
        
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
                print(f"[!] Lỗi đọc file {path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_and_filter_data(self):
        """Chỉ nạp và giữ lại dữ liệu của ĐÚNG MÃ CẦN SOI để chạy với tốc độ ánh sáng"""
        # Đọc Master Data
        df_price_l2_raw = self._load_parquet_safe(self.data_dir / 'price/master_price_l2.parquet')
        df_prop_raw = self._load_parquet_safe(self.data_dir / 'macro/prop_flow.parquet')
        df_intra_raw = self._load_parquet_safe(self.data_dir / 'intraday/master_intraday.parquet')
        df_board_raw = self._load_parquet_safe(self.data_dir / 'board/master_board.parquet')
        df_pt_raw = self._load_parquet_safe(self.data_dir / 'intraday/master_put_through.parquet')
        
        # TIME-TRAVEL: CẮT BỎ TOÀN BỘ DỮ LIỆU TƯƠNG LAI
        if self.target_date:
            target_end = self.target_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
            if not df_price_l2_raw.empty: df_price_l2_raw = df_price_l2_raw[df_price_l2_raw['time'] <= target_end]
            if not df_prop_raw.empty: df_prop_raw = df_prop_raw[df_prop_raw['time'] <= target_end]
            if not df_intra_raw.empty: df_intra_raw = df_intra_raw[df_intra_raw['time'] <= target_end]
            if not df_pt_raw.empty: df_pt_raw = df_pt_raw[df_pt_raw['time'] <= target_end]
            if not df_board_raw.empty and 'time' in df_board_raw.columns:
                df_board_raw = df_board_raw[df_board_raw['time'] <= target_end]

        self.df_comp = self._load_parquet_safe(self.data_dir / 'company/master_company.parquet')
        self.df_idx = self._load_parquet_safe(self.data_dir / 'macro/index_components.parquet')

        # Lọc siêu tốc (Chỉ lấy ticker)
        if not df_price_l2_raw.empty:
            self.df_price_l2 = df_price_l2_raw[df_price_l2_raw['ticker'] == self.ticker].copy() # đã clone
            if not self.df_price_l2.empty and 'matched_volume' in self.df_price_l2.columns:
                self.df_price_l2 = self.df_price_l2.rename(columns={'matched_volume': 'volume'})
        else:
            self.df_price_l2 = pd.DataFrame()

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
        self.price_l2_dict = {self.ticker: self.df_price_l2}
        self.prop_dict = {self.ticker: self.df_prop}

        # Khởi tạo Dict chứa Số lượng Cổ phiếu
        self.out_shares_dict = {}
        
        # 1. BASE LEVEL: Lấy từ master_company (Làm nền dự phòng)
        if not self.df_comp.empty and 'ticker' in self.df_comp.columns and 'issue_share' in self.df_comp.columns:
            self.out_shares_dict = self.df_comp[self.df_comp['ticker'] == self.ticker].set_index('ticker')['issue_share'].to_dict()

        # 2. Ghi đè bằng Dữ liệu Tươi (Live Data) từ master_price_l2
        if not self.df_price_l2.empty and 'total_shares' in self.df_price_l2.columns:
            # Lấy dòng dữ liệu mới nhất
            latest_l2 = self.df_price_l2.sort_values('time').iloc[-1]
            if pd.notna(latest_l2.get('total_shares')) and latest_l2['total_shares'] > 0:
                self.out_shares_dict[self.ticker] = latest_l2['total_shares']

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
            price_l2_dict=self.price_l2_dict, 
            prop_dict=self.prop_dict, 
            out_shares_dict=self.out_shares_dict, 
            universe=self.universe
        )
        self.mf_analyzer = MarketFlowAnalyzer()
        self.shadow_profiler = ShadowProfiler(df_l2=self.df_price_l2, df_prop=self.df_prop, verbose=False)
        
        # Băm OmniMatrix
        data_frames = {
            'price_l2': self.df_price_l2, 'prop': self.df_prop,
            'comp': self.df_comp, 'idx': self.df_idx, 'board': self.df_board, 
            'intra': self.df_intra, 'put_through': self.df_pt
        }
        self.omni_matrix = OmniFlowMatrix(data_frames, lookback_days=30)

    def _prepare_latest_ohlcv(self):
        """Gộp nến Intraday vào EOD để ra nến cập nhật nhất"""
        df_p = self.df_price_l2[['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']].copy()
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

        # Kiểm tra xem lệnh này có phải là Thỏa thuận (Đã bị lọc) hay không
        is_pt_f = t1_row.get('is_pt_f', False)
        is_pt_p = t1_row.get('is_pt_p', False)
        
        # Nếu là Thỏa thuận, ép giá trị về 0 để KHÔNG tính vào Xung đột Bảng điện
        t1_f = 0 if is_pt_f else t1_row.get('f_net_bn', 0)
        t1_p = 0 if is_pt_p else t1_row.get('p_net_bn', 0)
        t1_shadow = t1_row.get('shadow_flow_bn', 0)
        
        pattern = "GIẰNG CO (Dòng tiền tuần không rõ ràng)"
        thesis = "NEUTRAL"
        special_warning = ""

        # 🚨 BẮT BẪY 1: XUNG ĐỘT THỂ CHẾ (Khối ngoại xả, Tự doanh đỡ hoặc ngược lại)
        if (t1_f * t1_p < 0) and abs(t1_f) > 15 and abs(t1_p) > 15:
            special_warning = f"⚠️ XUNG ĐỘT TỶ ĐÔ: Ngoại ({t1_f:+.1f}) đánh nhau với Nội ({t1_p:+.1f}). Rủi ro Volatility cực cao!"

        # 🚨 BẮT BẪY 2: SÓNG LÁI NỘI ĐẦU CƠ (Tay to đứng ngoài, Shadow Flow áp đảo)
        elif avg_dominance < 5.0 and t1_shadow > 50:
            special_warning = f"🎭 SÓNG ĐẦU CƠ: Tay to vắng mặt (Chiếm {avg_dominance:.1f}%). Lái nội quay tay ({t1_shadow:.1f} Tỷ)!"
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

    def _log_audit_trail(self, price, signal, sm_vwap, pt_intent, dp_action, final_action, final_reason):
        """Hộp đen ghi nhận phán quyết của Sniper để Forward-test và Phân tích sau này"""
        log_path = self.data_dir.parent / 'invest/sniper_audit_log.csv'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = log_path.exists()
        
        # Bắt lỗi an toàn nếu pt_intent trống
        short_intent = pt_intent.split(' ')[0] if pt_intent and isinstance(pt_intent, str) else "NONE"
        
        row_data = {
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': self.ticker,
            'Price': round(price, 0),
            'Wyckoff': signal,
            'VWAP_SM': round(sm_vwap, 0),
            'DP_Intent': short_intent,
            'Radar_Action': dp_action,
            'Action': final_action, 
            'Reason': final_reason
        }
        
        df_log = pd.DataFrame([row_data])
        if not file_exists:
            df_log.to_csv(log_path, index=False, encoding='utf-8-sig')
        else:
            df_log.to_csv(log_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    def analyze(self):
        """Khớp nối dữ liệu và Báo cáo X-Quang"""
        df_full = self._prepare_latest_ohlcv()
        if df_full.empty:
            print(f"[!] Không có dữ liệu giá cho {self.ticker}.")
            return

        # 1. KỸ THUẬT (Wyckoff)
        forecaster = WyckoffForecaster(
            data_input=df_full, output_dir=None, 
            run_date=self.target_date if self.target_date else datetime.now(),
            verbose=False
        )
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

        sm_result = self.sm_engine.analyze_ticker(self.ticker)
        mf_result = self.mf_analyzer.analyze_flow(self.ticker, self.df_price_l2, self.df_prop)
        sm_vwap = mf_result.get('sm_vwap', 0)
        dtl = mf_result.get('dtl_days', 0)
        shark_status = mf_result.get('sm_status', 'NEUTRAL') # 🚀 Dữ liệu Vị thế Cá mập

        # 3. LÁI NỘI & QUÁ KHỨ (Shadow Profiler & Omni Past)
        past_ctx = self.omni_matrix.explain_past_movement(self.ticker, lookback_days=10)
        rules = self.shadow_profiler.build_criminal_profile([self.ticker], lookback_days=125)
        
        # TRUYỀN THAM SỐ TIME-TRAVEL CHO SHADOW PROFILER
        target_str = self.target_date.strftime('%Y-%m-%d') if self.target_date else None
        # 3.1 TRÍCH XUẤT CHÍNH XÁC STATUS & NOTE CỦA LÁI NỘI
        alerts = self.shadow_profiler.live_shadow_radar([self.ticker], rules, target_date=target_str)
        shadow_status = ""
        shadow_note = "Không phát hiện dấu vết nén nền đầu cơ."
        
        if isinstance(alerts, pd.DataFrame) and not alerts.empty:
            shadow_status = str(alerts.iloc[0].get('Status', ''))
            shadow_note = str(alerts.iloc[0].get('Note', shadow_note))
        elif isinstance(alerts, list) and len(alerts) > 0:
            if isinstance(alerts[0], dict):
                shadow_status = str(alerts[0].get('Status', ''))
                shadow_note = str(alerts[0].get('Note', shadow_note))
            else:
                shadow_note = str(alerts[0])
                
        shadow_msg = f"{shadow_status} - {shadow_note}".strip(' -')

        # 🚀 Radar Dark Pool V3.0
        dp_alerts = self.shadow_profiler.scan_dark_pool_deals([self.ticker], lookback_days=15)

        # 4. DỰ BÁO T0 (Omni Now)
        omni_now = self.omni_matrix.predict_t0_action(self.ticker, past_context=past_ctx)

        # --- IN BÁO CÁO SNIPER DASHBOARD ---
        print("\n" + "═"*65)
        print(f" 🎯 BÁO CÁO X-QUANG ĐỘC LẬP (SNIPER MODE) | MÃ: [ {self.ticker} ] - RỔ: {self.universe}")
        print("═"*65)
        
        # PHẦN 1
        print(f" 📈 1. KỸ THUẬT & CẤU TRÚC GIÁ (WYCKOFF)")
        print(f"    - Thị giá hiện tại : {price:,.0f} đ")
        print(f"    - Tín hiệu Wyckoff : {signal} (Khối lượng: {vpa})")
        print(f"    - Khung Giá Trị    : VAL {val:,.0f} -> Cân bằng POC {poc:,.0f} -> VAH {vah:,.0f}")
        
        if price < val: print("      => Đánh giá: Đang ở Vùng Giá Rẻ (Oversold). Ưu tiên tìm đáy/SPRING.")
        elif price > vah: print("      => Đánh giá: Đã Bứt phá vùng giá trị (Blue Sky). Ưu tiên đà tăng/SOS.")
        else: print("      => Đánh giá: Đang Tích lũy trong hộp. Ưu tiên test cung cạn kiệt.")

        print("-" * 65)
        
        # PHẦN 2
        print(f" 🐋 2. DÒNG TIỀN THỂ CHẾ (DỮ LIỆU CHỐT PHIÊN T-1)")
        print(f"    - Điểm Đồng thuận  : {sm_result.get('total_sm_score', 0)}/100")
        # Xử lý hiển thị VWAP khi Cá mập xả rỗng kho
        if sm_vwap <= 0:
            print(f"    - Giá vốn Tay to   : 🚨 ĐÃ XẢ SẠCH KHO (Mất hoàn toàn bệ đỡ VWAP)")
        else:
            print(f"    - Giá vốn Tay to   : ~{sm_vwap:,.0f} đ (VWAP)")
        print(f"    - Vị thế Cá Mập    : 🎯 {shark_status}")
        print(f"    - Sức ép Xả (DTL)  : {dtl:.1f} ngày")
        
        if sm_result.get("is_danger", False):
            print(f"    - Báo động Đỏ      : 🚨 {' | '.join(sm_result['warnings'])}")

        # Luôn in các hành vi khác dù đã có Báo động Đỏ (hay ko)
        if sm_result.get("sm_details"):
            print(f"    - Các hành vi khác : {' | '.join(sm_result['sm_details'])}")
            
        print(f"    - Radar Nén Nền    : {shadow_msg}")
        
        print("-" * 65)

        # PHẦN 3: ĐỘNG LƯỢNG KẾT HỢP & BÓC TÁCH LEVEL 2
        print(" ⚡ 3. BÓC TÁCH CẤU TRÚC LỆNH (LEVEL-2 X-RAY)")
        
        micro_flow = self._analyze_micro_flow()
        
        if micro_flow:
            print("    [TỔNG NGOẠI + TỰ DOANH (T-5 -> T-1)]")
            history_str = " | ".join([f"{micro_flow[k]:+.1f}" for k in ['t5', 't4', 't3', 't2', 't1']])
            print(f"    - Lưu lượng EOD hàng ngày: {history_str} (Tỷ VNĐ)")
            print(f"    - TỔNG KẾT TUẦN (Weekly) : {micro_flow['weekly_net']:+.1f} Tỷ")
            print(f"    - Hành vi Lái (Pattern)  : {micro_flow['pattern']}")
            print(f"    - Giả thuyết T0 (Thesis) : {micro_flow['thesis']}")
            
            # IN KẾT QUẢ KIỂM TOÁN T-1 LÊN MÀN HÌNH
            l2_data = omni_now.get('l2_data') if omni_now else None
            if l2_data:
                print(f"    [DÒNG TIỀN THỊ TRƯỜNG & THỂ CHẾ (T-5 -> T-1)]")
                active_5d = l2_data.get('active_flow_5d', [])
                if active_5d:
                    flow_str = " | ".join([f"{x:+.1f}" for x in active_5d])
                    print(f"    - Khớp ròng Chủ động EOD : {flow_str} (Tỷ VNĐ)")
            
                w_ratio = l2_data['whale_ratio']
                s_ratio = l2_data['spoofing_ratio']
                avg_buy, avg_sell = l2_data['avg_buy_vol'], l2_data['avg_sell_vol']
                
                print(f"    [HẬU KIỂM LEVEL-2 (PHIÊN T-1: {l2_data.get('t1_date', 'N/A')})]")
                print(f"    - Quy mô Lệnh (L2)       : Mua TB {avg_buy:,.0f} cp/lệnh vs Bán TB {avg_sell:,.0f} cp/lệnh")
                
                if w_ratio >= 1.5:
                    print(f"    - Dấu chân Cá voi (Whale): 🐋 LỆNH MUA TO GẤP {w_ratio:.1f}x LỆNH BÁN")
                elif w_ratio <= 0.6:
                    print(f"    - Dấu chân Cá voi (Whale): 🦈 LỆNH BÁN TO GẤP {1/w_ratio:.1f}x LỆNH MUA")
                else:
                    print(f"    - Dấu chân Cá voi (Whale): ⚖️ Giằng co nhỏ lẻ (Tỷ lệ {w_ratio:.1f}x)")
                    
                if s_ratio > 2.0:
                    print(f"    - Bẫy Kê Lệnh (Spoofing) : ⚠️ DƯ MUA ẢO (Gấp {s_ratio:.1f}x Dư bán) - Cẩn thận dụ Fomo!")
                elif s_ratio < 0.5:
                    print(f"    - Bẫy Kê Lệnh (Spoofing) : ⚠️ DƯ BÁN ẢO (Gấp {1/s_ratio:.1f}x Dư mua) - Cẩn thận đè gom!")

                buy_bn = l2_data.get('t1_buy_active_bn', 0)
                sell_bn = l2_data.get('t1_sell_active_bn', 0)
                net_bn = l2_data.get('t1_net_active_bn', 0)
                print(f"    - Khớp lệnh Chủ động     : 🟢 Mua {buy_bn:.1f} Tỷ vs 🔴 Bán {sell_bn:.1f} Tỷ (Ròng: {net_bn:+.1f} Tỷ)")
        
        print("    [HIỆN TẠI T0 - INTRADAY MOMENTUM]")
        if omni_now and "error" not in omni_now:
            is_offline = omni_now.get('is_offline') if omni_now else None
            if not is_offline:
                print(f"    - Khớp chủ động T0       : {omni_now.get('net_active_bn', 0):+.1f} Tỷ VNĐ")
                l2 = omni_now.get('l2_data', {})
                if l2:
                    bu_bn = l2.get('t0_total_bu_bn', 0)
                    sd_bn = l2.get('t0_total_sd_bn', 0)
                    sh_bu = l2.get('t0_shark_bu_bn', 0)
                    sh_sd = l2.get('t0_shark_sd_bn', 0)
                    sh_bu_c = l2.get('t0_shark_bu_count', 0)
                    sh_sd_c = l2.get('t0_shark_sd_count', 0)
                    
                    print(f"    - Chi tiết Lệnh          : 🟢 Mua {bu_bn:.1f} Tỷ | 🔴 Bán {sd_bn:.1f} Tỷ")
                    if sh_bu > 0 or sh_sd > 0:
                        print(f"    - Dấu chân Cá mập (>1T)  : 🟢 Mua {sh_bu:.1f} Tỷ ({sh_bu_c} lệnh) | 🔴 Bán {sh_sd:.1f} Tỷ ({sh_sd_c} lệnh)")
                f_matched_t0 = omni_now.get('t0_f_matched_net_bn', omni_now.get('f_net_t0', 0))
                f_impact = omni_now.get('t0_f_impact_pct', 0)
                print(f"    - Ngoại T0 (Khớp Lệnh)   : {f_matched_t0:+.1f} Tỷ VNĐ (Impact: {f_impact:.1f}%) (Đã khấu trừ Deal)")
                print(f"    - Sổ lệnh (Bid/Ask)      : Mất cân bằng {omni_now.get('imbalance', 0):+.2f} (Dương=Kê Mua / Âm=Chặn Bán)")
                print(f"    - Tác nhân T0            : {omni_now.get('driver_msg')}")
                print(f"    => BẢN ÁN OMNI T0        : {omni_now.get('verdict', 'N/A')}")
                print(f"    => LÝ DO CHI TIẾT        : {omni_now.get('details', 'N/A')}")
            else:
                print("    - Thị trường Đóng cửa/Chưa có GD T0.")
        else:
            print("    - Chưa có dữ liệu Khớp lệnh T0 hoặc lỗi Omni-Matrix.")

        print("-" * 65)

        # Phần 4: Kiểm toán Thỏa thuận
        print(f" 🤝 4. KIỂM TOÁN GIAO DỊCH THỎA THUẬN (OFF-BOOK AUDIT)")
        intent_str = "NONE"
        dp_action_tag = "NONE"

        if not self.df_pt.empty:
            # XÁC ĐỊNH CHUẨN PHIÊN GIAO DỊCH T-5 ĐẾN T0
            past_sessions = sorted(self.df_price_l2['time'].unique())
            if len(past_sessions) >= 5:
                if past_sessions[-1].date() == self.omni_matrix.t0_date:
                    trading_days = past_sessions[-6:] 
                else:
                    t0_timestamp = pd.Timestamp(self.omni_matrix.t0_date)
                    trading_days = past_sessions[-5:] + [t0_timestamp]
            else:
                trading_days = past_sessions 

            pt_vals = []
            pt_intents = []
            
            for d in trading_days:
                df_day = self.df_pt[self.df_pt['time'] == d].copy()
                if not df_day.empty:
                    val_bn = df_day['match_value'].sum() / 1_000_000_000
                    df_day['weighted_change'] = df_day['change_percent'] * df_day['match_value']
                    w_change = (df_day['weighted_change'].sum() / df_day['match_value'].sum()) * 100
                    
                    pt_vals.append(f"{val_bn:+.1f}")
                    if w_change > 1.0: pt_intents.append("🔥")     # Premium
                    elif w_change < -1.0: pt_intents.append("🧊")  # Discount
                    else: pt_intents.append("⚖️")                  # Neutral
                else:
                    pt_vals.append(f"{0.0:+.1f}")
                    pt_intents.append(" -")

            print(f"    [LỊCH TRÌNH 6 PHIÊN (T-5 -> T0)]")
            print(f"    - Sang tay (Tỷ VNĐ) : {' | '.join(pt_vals)}")
            print(f"    - Ý đồ (Intent)     : {'  |  '.join(pt_intents)}  (🔥 Đắt / 🧊 Rẻ / ⚖️ Tham chiếu)")
            print(f"    --------------------------------------------------------------------------------------")

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
                    intent_str = "🔥 PREMIUM"
                    print(f"    - Ý đồ Tổng thể (Intent): {intent_str} (Khát hàng, Chấp nhận mua đắt -> Siêu Bullish)")
                elif avg_change < -1.0:
                    intent_str = "🧊 DISCOUNT"
                    print(f"    - Ý đồ Tổng thể (Intent): {intent_str} (Trao tay giá rẻ -> Thường là Sang tay nội bộ/Né thuế)")
                else:
                    intent_str = "⚖️ NEUTRAL"
                    print(f"    - Ý đồ Tổng thể (Intent): {intent_str} (Trao tay quanh giá tham chiếu)")
                    
                if total_pt_val > 50.0:
                    print(f"    => 🛡️ TỌA ĐỘ PHÒNG THỦ   : Nếu giá rớt về sát {vwap_pt:,.0f}đ sẽ kích hoạt Lực đỡ Khổng lồ!")

                print(f"    --------------------------------------------------------------------------------------")
                
                # 🚀 TÍCH HỢP DARK POOL RADAR V3.0
                if dp_alerts:
                    print(f"    [📡 RADAR DARK POOL V3.0 - PHÁT HIỆN DẤU CHÂN]")
                    for a in dp_alerts:
                        print(f"    - {a['Type']:<15}: {a['Note']}")
                        # Móc nối logic chặn mua nếu phát hiện Xả Ngầm
                        if 'XẢ NGẦM' in a['Type']:
                            dp_action_tag = 'DANGER'
                else:
                    print("    [📡 RADAR DARK POOL V3.0] Không phát hiện giao dịch ngầm đáng ngờ 15D.")

        else:
            print("    - Không có dữ liệu Giao dịch Thỏa thuận.")
        print("═"*65)

        # =====================================================================
        # KINETIC IGNITION (LUỒNG KÍCH NỔ ĐỘNG LƯỢNG)
        # =====================================================================
        # 1. Tính toán %1D và %5D_Base
        df_p_valid = df_full.dropna(subset=['close']).copy()
        pct_1d = 0.0
        pct_5d_base = 0.0
        if len(df_p_valid) >= 6:
            c_today = float(df_p_valid['close'].iloc[-1]) # Giá chốt T0 (Ngày nổ)
            c_1d = float(df_p_valid['close'].iloc[-2])    # Giá chốt T-1 (Hôm qua)
            c_5d = float(df_p_valid['close'].iloc[-6])    # Giá chốt T-5
            
            # Gia tốc của riêng cây nến bùng nổ T0
            pct_1d = ((c_today - c_1d) / c_1d) * 100 if c_1d > 0 else 0
            
            # ĐỘT PHÁ TOÁN HỌC: Nền nén phải được đo TRƯỚC ngày nổ (T-5 -> T-1)
            pct_5d_base = ((c_1d - c_5d) / c_5d) * 100 if c_5d > 0 else 0

        # 2. Định vị Cờ Bảo Kê (Override Flags)
        sm_details = sm_result.get('sm_details', [])
        has_active_override = any("ACTIVE OVERRIDE" in msg for msg in sm_details)
        
        # BỘ NHỚ DAI DẲNG (PERSISTENT SHADOW MEMORY 30 Phiên giao dịch trước đó)
        has_shadow_override, shadow_memory_days = False, 30
        
        # Lấy danh sách 30 phiên giao dịch gần nhất từ dữ liệu giá
        past_dates = df_p_valid['time'].dt.strftime('%Y-%m-%d').unique().tolist()[-shadow_memory_days:]
        
        # Quét giật lùi từ hiện tại về quá khứ
        for d_str in reversed(past_dates):
            alerts_past = self.shadow_profiler.live_shadow_radar([self.ticker], rules, target_date=d_str)
            st_val = ""
            if isinstance(alerts_past, pd.DataFrame) and not alerts_past.empty:
                st_val = str(alerts_past.iloc[0].get('Status', ''))
            elif isinstance(alerts_past, list) and len(alerts_past) > 0:
                item = alerts_past[0]
                st_val = str(item.get('Status', '')) if isinstance(item, dict) else str(item)

            if "CHÍN MUỒI" in st_val:
                has_shadow_override = True
                break # Chỉ cần từng gom 1 lần trong 30 ngày là đủ điều kiện bảo kê!
        
        # 2.1 Cờ Bảo Kê từ Bảng điện Real-time (T0)
        is_t0_whale_backed = False
        if omni_now:
            t0_active_net = omni_now.get('net_active_bn', 0)
            t0_verdict = omni_now.get('verdict', '')
            whale_thresh = 100.0 if self.universe == 'VN30' else (15.0 if self.universe == 'VNSmallCap' else 50.0)
            
            # Đang kéo mạnh và Cá mập vào lệnh trực tiếp T0
            if ("BULLISH" in t0_verdict) or ("TILT BULL" in t0_verdict and t0_active_net >= whale_thresh):
                is_t0_whale_backed = True

        # Gộp tất cả các lớp bảo kê (Chỉ cần 1 trong 3 điều kiện xảy ra)
        is_money_backed = has_active_override or has_shadow_override or is_t0_whale_backed

        # 3. Kích hoạt Holy Trinity (Bộ 3 Điều Kiện Vàng)
        is_kinetic_ignition = False
        # Nền nén (trước nổ) dao động từ -10% đến +1% (Cho phép đi ngang), Nổ T0 >= 3.0%, Có bảo kê
        if (-10.0 <= pct_5d_base <= 1.0) and (pct_1d >= 3.0) and is_money_backed:
            is_kinetic_ignition = True

        # PHẦN 5: KẾT LUẬN ĐẦU TƯ
        print(" 🎯 HÀNH ĐỘNG KHUYẾN NGHỊ (SNIPER FINAL VERDICT):")
        final_action, final_reason = "WAIT", ""

        if omni_now and "BEARISH (Bẫy Kéo Xả Ảo)" in omni_now.get('verdict', ''):
            final_action = "REJECT"
            final_reason = "BẪY KẾO XẢ LEVEL-2: Lệnh bán thực tế to gấp nhiều lần lệnh mua dù đang kê dư mua ảo."
            print(f"   🚫 KHÔNG MUA: {final_reason}")
            
        elif dp_action_tag == 'DANGER':
            final_action = "REJECT"
            final_reason = "DARK POOL CẢNH BÁO ĐỎ: Phát hiện Lái Xả Ngầm khối lượng lớn (Trao tay phân phối)."
            print(f"   🚫 KHÔNG MUA: {final_reason}")
            
        # ƯU TIÊN 1: LUỒNG KÍCH NỔ ĐỘNG LƯỢNG (BỎ QUA WYCKOFF)
        elif is_kinetic_ignition:
            final_action = "BUY_MARKET"
            sl_price = df_p_valid['low'].iloc[-1]
            buy_p = board_info['best_ask'] if board_info and board_info['best_ask'] > 0 else price
            final_reason = f"KINETIC IGNITION: Nền nén {pct_5d_base:+.1f}%, Bứt phá {pct_1d:+.1f}% + Cờ Bảo kê."

            backed_by = []
            if is_t0_whale_backed: backed_by.append("Cá Mập Real-time (T0)")
            if has_shadow_override: backed_by.append("Lái Nội (Shadow)")
            if has_active_override: backed_by.append("Cầu Đỡ Bảng Điện (Active T-1)")
            
            print(f"   🚀 [KINETIC IGNITION]: PHÁT HIỆN ĐIỂM KÍCH NỔ ĐỘNG LƯỢNG!")
            print(f"      - Nền nén (%5D) : {pct_5d_base:+.1f}% (Đã rũ bỏ/đi ngang thành công)")
            print(f"      - Gia tốc (%1D) : {pct_1d:+.1f}% (Bứt phá V-Shape)")
            print(f"      - Bảo kê bởi    : {' + '.join(backed_by)}")
            print(f"      => LỆNH BẮN TỈA: MUA KHẨN CẤP (MARKET) quanh {buy_p:,.0f} đ.")
            print(f"      => 🛑 CHỐT CHẶN MỎ NEO: Cắt lỗ tuyệt đối nếu giá thủng {sl_price:,.0f} đ.")

        # ƯU TIÊN 2: LUẬT QUẢN TRỊ BÁO ĐỘNG ĐỎ TRUYỀN THỐNG
        elif sm_result.get("is_danger", False):
            is_validated_bull = micro_flow and ("BULLISH" in micro_flow['thesis'] or "BOTTOMING" in micro_flow['thesis'] or "TILT BULL" in micro_flow['thesis'])
            
            # GIẢI CỨU LINH HOẠT
            # Ghi đè Cờ Đỏ nếu T0 có lực Cầu chủ động Bùng nổ (Đảo pha)
            whale_thresh = 100.0 if self.universe == 'VN30' else (15.0 if self.universe == 'VNSmallCap' else 50.0)
            is_reversal = False
            if omni_now:
                is_bullish = "BULLISH" in omni_now.get('verdict', '')
                is_tilt_bull_strong = "TILT BULL" in omni_now.get('verdict', '') and omni_now.get('net_active_bn', 0) >= whale_thresh
                is_reversal = is_bullish or is_tilt_bull_strong

            if not is_validated_bull and not is_reversal:
                final_action = "REJECT"
                final_reason = "Cá mập dài hạn đang tháo cống (Hoặc áp lực xả > Ngưỡng Impact). Bỏ qua mọi kỹ thuật."
                print(f"   🚫 KHÔNG MUA: {final_reason}")
            elif is_reversal:
                # Nếu T0 Bullish, xí xóa cờ đỏ và cho phép lệnh mua đi tiếp
                print(f"   🌟 GIẢI CỨU BÁO ĐỘNG ĐỎ: T0 Cầu chủ động bùng nổ, bẻ gãy áp lực xả 3D của Cá mập!")

        # ƯU TIÊN 3: LUẬT ĐẢO PHA CẤU TRÚC
        elif micro_flow and "BEARISH" in micro_flow['thesis']:
            # GIẢI CỨU LINH HOẠT
            # Mở khóa Đảo pha Cấu trúc (Structural Reversal).
            # Nếu T0 có lực Cầu chủ động kéo thốc (BULLISH), cho phép xí xóa quá khứ xấu.
            whale_thresh = 100.0 if self.universe == 'VN30' else (15.0 if self.universe == 'VNSmallCap' else 50.0)
            is_reversal = False
            if omni_now:
                is_bullish = "BULLISH" in omni_now.get('verdict', '')
                is_tilt_bull_strong = "TILT BULL" in omni_now.get('verdict', '') and omni_now.get('net_active_bn', 0) >= whale_thresh
                is_reversal = is_bullish or is_tilt_bull_strong

            if not is_reversal:
                final_action = "REJECT"
                final_reason = "ĐANG TẮM MÁU: Áp lực xả từ tuần trước vẫn tiếp diễn. Bắt đáy là cụt tay!"
                print(f"   🩸 KHÔNG MUA: {final_reason}")
            else:
                print(f"   🌟 ĐẢO PHA CẤU TRÚC: Quá khứ bị xả, nhưng T0 Cầu chủ động đã vào giải cứu. Cho phép ngắm bắn!")
             
        elif micro_flow and "BOTTOMING" in micro_flow['thesis'] and omni_now and "BEARISH" in omni_now.get('verdict',''):
            final_action = "CANCEL"
            final_reason = "HỦY KẾ HOẠCH BẮT ĐÁY: T-1 đã tiết cung, nhưng T0 lái lại tiếp tục táng hàng."
            print(f"   ⚠️ {final_reason}")
            
        elif micro_flow and "BULLISH" in micro_flow['thesis'] and omni_now and "BEARISH" in omni_now.get('verdict',''):
            final_action = "CANCEL"
            final_reason = "HỦY LỆNH MUA (BULL-TRAP): Quá khứ gom đẹp nhưng T0 bị xả ngược."
            print(f"   ⚠️ {final_reason}")
            
        # ƯU TIÊN 4: CÁC KỊCH BẢN WYCKOFF KÍCH HOẠT MUA TRUYỀN THỐNG
        elif signal in ['SOS', 'SPRING', 'TEST_CUNG', 'NEUTRAL'] and micro_flow and ("BULLISH" in micro_flow['thesis'] or "BOTTOMING" in micro_flow['thesis'] or "TILT BULL" in micro_flow['thesis']):
            sl_price = price - (wyckoff_row.get('ATR', price * 0.02) * 2.5)
            
            if omni_now and "Tiết cung" in omni_now.get('verdict', ''):
                final_action = "BUY_LIMIT"
                buy_p = board_info['best_bid'] if board_info else price
                final_reason = f"GOM VỊ THẾ NỀN. Kê mua rải đinh quanh {buy_p:,.0f} đ."
                print(f"   🌱 GOM VỊ THẾ NỀN (ACCUMULATE): Quá khứ dọn đường + T0 Cạn cung chờ gió Đông!")
                print(f"      => LỆNH BẮN TỈA: Kê mua rải đinh quanh {buy_p:,.0f} đ | Dừng lỗ cứng tại {sl_price:,.0f} đ.")
            elif omni_now and "BULLISH" in omni_now.get('verdict', ''):
                final_action = "BUY_MARKET"
                buy_p = board_info['best_ask'] if board_info and board_info['best_ask'] > 0 else price
                final_reason = f"ĐIỂM NỔ BỨT PHÁ. Mua quét (Market) quanh {buy_p:,.0f} đ."
                print(f"   🌟 ĐIỂM NỔ BỨT PHÁ (SNIPER TRIGGER): Quá khứ dọn đường + T0 Tiền vào xác nhận!")
                print(f"      => LỆNH BẮN TỈA: Mua quét lên (Market) quanh {buy_p:,.0f} đ | Dừng lỗ cứng tại {sl_price:,.0f} đ.")
            else:
                 final_action = "WAIT"
                 final_reason = "Có điểm cộng kỹ thuật, nhưng dòng tiền chưa đồng thuận mạnh."
                 print(f"   ☕ CHỜ ĐỢI: {final_reason}")
                
            if sm_vwap > 0 and price < sm_vwap:
                print(f"      => LỢI THẾ KÉP: Giá ({price:,.0f}) đang rẻ hơn giá vốn Cá mập ({sm_vwap:,.0f}).")
                final_reason += f" (Rẻ hơn VWAP Lái)"
                
        # KỊCH BẢN GIẰNG CO (CHỜ ĐỢI)
        else:
            final_action = "WAIT"
            final_reason = "Dòng tiền giằng co. Hành vi Lái chưa rõ ràng. Cần theo dõi thêm."
            print(f"   ☕ CHỜ ĐỢI: {final_reason}")

        print("═"*65)

        # GỌI HỘP ĐEN GHI LẠI DỮ LIỆU
        self._log_audit_trail(price, signal, sm_vwap, intent_str, dp_action_tag, final_action, final_reason)

if __name__ == "__main__":
    while True:
        ticker_input = input("\nNhập mã cổ phiếu (VD: FPT) hoặc 'Q' để thoát: ").strip()
        if ticker_input.upper() == 'Q':
            break
            
        date_input = input("Nhập ngày giả lập (YYYY-MM-DD) hoặc ấn Enter để chạy Live: ").strip()
        target_dt = date_input if date_input != "" else None
        
        if len(ticker_input) >= 3:
            try:
                sniper = TargetSniper(ticker=ticker_input, target_date=target_dt)
                sniper.analyze()
            except Exception as e:
                print(f"[!] Lỗi giả lập: {e}")