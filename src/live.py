import pandas as pd
import numpy as np
import os
import shutil
import json
import textwrap
import time
import sys
from datetime import datetime
from pathlib import Path

from src.forecaster import WyckoffForecaster
from src.smart_money import SmartMoneyEngine
from src.market_flow import MarketFlowAnalyzer
from src.reporter_by_group import GroupCashFlowReporter
from src.shadow_profiler import ShadowProfiler
from src.portfolio import QuantPortfolioEngine
from src.market_tracker import MarketTracker
from src.blacklist_guard import BlacklistGuard
from src.omni_matrix import OmniFlowMatrix

# --- CẤU HÌNH RỦI RO CƠ BẢN ---
RISK_PER_TRADE = 1_000_000
MAX_POSITION_SIZE = 5000
BASE_SCORE_THRESHOLD = 70

# --- DANH SÁCH ĐEN (CÁC MÃ LỖI DỮ LIỆU / KHÔNG GIAO DỊCH) ---
IGNORE_TICKERS = []

# ==============================================================================

class LiveAssistant:
    def __init__(self, universe='VN30'):
        self.universe = universe
        self.temp_dir = Path(f'data/temp_live_{universe.lower()}')
        self.live_dir = Path('data/live')
        self.parquet_dir = Path('data/parquet')
        self._ensure_temp_dir()
        
        # Load dữ liệu Bảng giá và BCTC vào RAM (siêu nhanh nhờ Parquet)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp Master Data vào RAM...")
        self.df_board = self._load_parquet_safe(self.parquet_dir / 'board/master_board.parquet')
        self.df_fin = self._load_parquet_safe(self.parquet_dir / 'financial/master_financial.parquet')

        # Nạp Khối ngoại
        self.df_foreign = self._load_parquet_safe(self.parquet_dir / 'macro/foreign_flow.parquet')
        self.foreign_dict = self._load_foreign_flow_dict(self.parquet_dir / 'macro/foreign_flow.parquet')

        # Nạp Tự doanh
        self.df_prop = self._load_parquet_safe(self.parquet_dir / 'macro/prop_flow.parquet')
        self.prop_dict = self._load_prop_flow_dict(self.parquet_dir / 'macro/prop_flow.parquet')

        # Nạp OHLCV
        self.df_price = self._load_parquet_safe(self.parquet_dir / 'price/master_price.parquet')
        self.price_dict = self._load_price_dict(self.parquet_dir / 'price/master_price.parquet')

        # Nạp Intraday
        self.df_intra = self._load_parquet_safe(self.parquet_dir / 'intraday/master_intraday.parquet')

        # Nạp Put-through
        self.df_pt = self._load_parquet_safe(self.parquet_dir / 'intraday/master_put_through.parquet')

        # Nạp danh sách mã theo ngành
        self.df_ind = self._load_parquet_safe(self.parquet_dir / 'macro/groups_by_industries.parquet')

        # Nạp Tổng khối lượng lưu hành để đo lường Cung/Cầu
        self.out_shares_dict = {}
        df_comp = self._load_parquet_safe(self.parquet_dir / 'company/master_company.parquet')
        if not df_comp.empty and 'ticker' in df_comp.columns and 'issue_share' in df_comp.columns:
            # Tạo dictionary O(1) tra cứu siêu tốc: { 'FPT': 1200000000, ... }
            self.out_shares_dict = df_comp.set_index('ticker')['issue_share'].to_dict()

        # Nạp quỹ trái phiếu
        self.df_funds = self._load_parquet_safe(self.parquet_dir / 'macro/bond_fund.parquet')

        # Lọc rổ Cổ phiếu
        self._filter_universe()

        # Đánh giá Ma trận Thanh khoản Vĩ mô
        self.macro_buy_threshold_adj = 0
        self.macro_risk_factor = 1.0
        self.macro_status = "NEUTRAL"
        self._evaluate_macro_environment()

        # Đánh giá Sóng Ngành Tự Động
        self.dynamic_thematic_tickers = []
        self._evaluate_sector_themes()

        # CHIẾN LƯỢC THEO QUÝ
        self._set_seasonality_rules()

        # CÔNG TẮC VĨ MÔ TỰ ĐỘNG
        self._analyze_macro_conditions()

        # Khởi tạo Động cơ Smart Money
        try:
            self.sm_engine = SmartMoneyEngine(
                foreign_dict=self.foreign_dict, 
                prop_dict=self.prop_dict, 
                out_shares_dict=self.out_shares_dict, 
                price_dict=self.price_dict, 
                universe=self.universe
            )
        except Exception as e:
            print(f"[!] Lỗi khởi động Smart Money: {e}")

        # KHỞI ĐỘNG BỘ NÃO SĂN LÁI NỘI
        try:
            self.shadow_profiler = ShadowProfiler(price_df=self.df_price, verbose=False)
            all_tickers = self.shadow_profiler.df_price['ticker'].unique().tolist()
            market_tickers = [t for t in all_tickers if len(str(t)) == 3]
            self.shadow_candidates = self.shadow_profiler._filter_shadow_candidates(market_tickers)
            self.shadow_rules = self.shadow_profiler.build_criminal_profile(self.shadow_candidates, lookback_days=250)
        except Exception as e:
            print(f"[!] Lỗi khởi động Shadow Profiler: {e}")

        # KHỞI ĐỘNG ĐÀI QUAN SÁT VĨ MÔ & ORDER FLOW (MARKET TRACKER)
        self.market_status = 'NEUTRAL'
        self.market_net_active = 0
        self.intraday_dict = {}
        try:
            self.market_tracker = MarketTracker(data_dir=self.parquet_dir, verbose=False)
            intraday_result = self.market_tracker.analyze_full_intraday_macro(intraday_df=self.df_intra)
            if intraday_result:
                self.market_status = intraday_result.get('market_status', 'NEUTRAL')
                self.market_net_active = intraday_result.get('market_net_active', 0)
                self.intraday_dict = intraday_result.get('intraday_dict', {})
        except Exception as e:
            print(f"[!] Lỗi khởi động Market Tracker: {e}")

        # Khởi tạo Vệ binh
        try:
            self.blacklist_guard = BlacklistGuard(universe=self.universe, verbose=False)
        except Exception as e:
            print(f"[!] Lỗi khởi động Blacklist Guard: {e}")

        # KHỞI ĐỘNG HỆ THỐNG X-QUANG ĐA CHIỀU (OMNI-MATRIX)
        try:
            data_frames = {
                'price': self.df_price,
                'foreign': self.df_foreign,
                'prop': self.df_prop,
                'comp': df_comp,
                'idx': self._load_parquet_safe(self.parquet_dir / 'macro/index_components.parquet'),
                'board': self.df_board,
                'intra': self.df_intra,
                'put_through': self.df_pt
            }
            self.omni_matrix = OmniFlowMatrix(data_frames, lookback_days=30)
        except Exception as e:
            print(f"[!] Lỗi khởi động OmniFlowMatrix: {e}")
            self.omni_matrix = None

    def _filter_universe(self):
        """Lọc Master Price theo Rổ cổ phiếu Chuẩn MECE (HOSE, VN30, VNMID, VNSMALL)"""
        if self.df_price.empty: return

        valid_tickers = []
        index_path = self.parquet_dir / 'macro/index_components.parquet' 
        
        if index_path.exists():
            df_idx = pd.read_parquet(index_path)
            vn30_tickers = df_idx[df_idx['index_code'] == 'VN30']['ticker'].tolist()
            mid_tickers = df_idx[df_idx['index_code'] == 'VNMidCap']['ticker'].tolist()
            small_tickers = df_idx[df_idx['index_code'] == 'VNSmallCap']['ticker'].tolist()
            hose_tickers = df_idx[df_idx['index_code'] == 'HOSE']['ticker'].tolist()

            if self.universe == "VN30":
                valid_tickers = vn30_tickers
            elif self.universe == "VNMidCap":
                valid_tickers = mid_tickers
            elif self.universe == "VNSmallCap":
                valid_tickers = small_tickers
            elif self.universe == "HOSE" or self.universe == "ALL":
                valid_tickers = hose_tickers
                
        if valid_tickers:
            # 1. LỌC ĐỒNG LOẠT CÁC DATAFRAME (GIẢI PHÓNG RAM)
            if not self.df_price.empty and 'ticker' in self.df_price.columns:
                self.df_price = self.df_price[self.df_price['ticker'].isin(valid_tickers)]
                
            if not self.df_intra.empty and 'ticker' in self.df_intra.columns:
                self.df_intra = self.df_intra[self.df_intra['ticker'].isin(valid_tickers)]
                
            if not self.df_fin.empty and 'ticker' in self.df_fin.columns:
                self.df_fin = self.df_fin[self.df_fin['ticker'].isin(valid_tickers)]
                
            if not self.df_foreign.empty and 'ticker' in self.df_foreign.columns:
                self.df_foreign = self.df_foreign[self.df_foreign['ticker'].isin(valid_tickers)]
                
            if not self.df_prop.empty and 'ticker' in self.df_prop.columns:
                self.df_prop = self.df_prop[self.df_prop['ticker'].isin(valid_tickers)]
                
            if not self.df_board.empty:
                col_name = 'symbol' if 'symbol' in self.df_board.columns else 'ticker'
                if col_name in self.df_board.columns:
                    self.df_board = self.df_board[self.df_board[col_name].isin(valid_tickers)]

            if not self.df_pt.empty:
                col_name = 'symbol' if 'symbol' in self.df_pt.columns else 'ticker'
                if col_name in self.df_pt.columns:
                    self.df_pt = self.df_pt[self.df_pt[col_name].isin(valid_tickers)]

            # 2. LỌC ĐỒNG LOẠT CÁC DICTIONARY BẰNG SET (TRA CỨU O(1))
            valid_set = set(valid_tickers) # Chuyển list thành set để tăng tốc độ lặp lên 100 lần
            
            if hasattr(self, 'price_dict'):
                self.price_dict = {k: v for k, v in self.price_dict.items() if k in valid_set}
                
            if hasattr(self, 'foreign_dict'):
                self.foreign_dict = {k: v for k, v in self.foreign_dict.items() if k in valid_set}
                
            if hasattr(self, 'prop_dict'):
                self.prop_dict = {k: v for k, v in self.prop_dict.items() if k in valid_set}
                
            if hasattr(self, 'out_shares_dict'):
                self.out_shares_dict = {k: v for k, v in self.out_shares_dict.items() if k in valid_set}
                
        print(f"[*] Đã thanh lọc Toàn bộ Master Data: Giữ lại tối đa {len(valid_tickers)} mã thuộc rổ {self.universe}.")

    def _check_historical_shadow_profile(self, ticker, sos_date, lookback_window=15):
        """
        Bộ nhớ Rình mồi: Lùi về quá khứ (tối đa 15 PHIÊN GIAO DỊCH trước ngày nổ SOS) 
        để xem mã này đã từng được Radar Lái nội đánh giá là 'CHÍN MUỒI' hay chưa.
        """
        if not hasattr(self, 'shadow_profiler') or not hasattr(self, 'shadow_rules'):
            return None

        # 1. Lấy trục thời gian giao dịch thực tế từ Bảng giá
        df_price = self.price_dict.get(ticker)
        if df_price is None or df_price.empty:
            return None

        # Cắt đứt tương lai (chỉ lấy đến ngày quét SOS)
        df_price_valid = df_price[df_price['time'] <= pd.to_datetime(sos_date).normalize()]
        if df_price_valid.empty:
            return None

        # Rút trích danh sách CÁC NGÀY CÓ GIAO DỊCH THỰC TẾ
        trading_dates = df_price_valid['time'].sort_values().unique()

        # Nếu cổ phiếu mới lên sàn chưa đủ 2 phiên thì bỏ qua
        if len(trading_dates) < 2:
            return None

        # 2. Quét lùi theo đúng Từng Phiên (Trading Sessions)
        # Bỏ qua đúng phiên SOS hiện tại (trading_dates[-1]), bắt đầu lùi từ T-1 (trading_dates[-2])
        max_lookback = min(lookback_window, len(trading_dates) - 1)

        for i in range(1, max_lookback + 1):
            check_date = trading_dates[-(i + 1)] # -(i+1) đảm bảo T-1, T-2, T-3... chuẩn xác
            date_str = pd.to_datetime(check_date).strftime('%Y-%m-%d')
            
            # Mượn Cỗ máy thời gian của Profiler để soi lại quá khứ
            raw_alerts = self.shadow_profiler.live_shadow_radar(
                [ticker], 
                self.shadow_rules, 
                target_date=date_str
            )
            
            if raw_alerts:
                alert = raw_alerts[0]
                if "CHÍN MUỒI" in alert['Status'] or "CHỜ ĐỢI" in alert['Status']:
                    # Ghi nhận lại độ trễ THEO PHIÊN GIAO DỊCH
                    alert['Days_Delayed'] = i 
                    alert['Memory_Date'] = date_str
                    return alert
                    
        return None # Quét 15 phiên không thấy dấu vết Lái nội

    def _ensure_temp_dir(self):
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _load_parquet_safe(self, path):
        """Hàm đọc Parquet an toàn, tránh lỗi nếu file chưa tồn tại"""
        if path.exists():
            try: 
                return pd.read_parquet(path)
            except: 
                print(f"Could NOT read {path}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _load_foreign_flow_dict(self, path):
        """Đọc file Parquet khối ngoại và băm thành Dictionary O(1) chỉ giữ 20 phiên"""
        df = self._load_parquet_safe(path)
        if df.empty or 'ticker' not in df.columns:
            return {}
            
        foreign_dict = {}
        # Gom nhóm và cắt ngọn (Tail)
        for ticker, group in df.groupby('ticker'):
            # CHỈ GIỮ 130 DÒNG CUỐI (Tương đương 6 tháng giao dịch)
            group = group.sort_values('time').tail(130)
            foreign_dict[ticker] = group
            
        return foreign_dict

    def _load_prop_flow_dict(self, path):
        """Đọc file Parquet Tự doanh và băm thành Dictionary O(1) chỉ giữ 20 phiên"""
        df = self._load_parquet_safe(path)
        if df.empty or 'ticker' not in df.columns:
            return {}
            
        prop_dict = {}
        for ticker, group in df.groupby('ticker'):
            # CHỈ GIỮ 130 DÒNG CUỐI (Tương đương 6 tháng giao dịch)
            group = group.sort_values('time').tail(130)
            prop_dict[ticker] = group
            
        return prop_dict

    def _load_price_dict(self, path):
        df = self._load_parquet_safe(path)
        if df.empty or 'ticker' not in df.columns:
            return {}
            
        price_dict = {}
        for ticker, group in df.groupby('ticker'):
            # CHỈ GIỮ 130 DÒNG CUỐI (Tương đương 6 tháng giao dịch)
            group = group.sort_values('time').tail(130)
            price_dict[ticker] = group
            
        return price_dict

    def _evaluate_macro_environment(self):
        """Phân tích Ma trận Thanh khoản từ dữ liệu Vĩ mô (CPI & Tín dụng)"""
        print("\n" + "="*65)
        print(" 🌍 ĐÁNH GIÁ MA TRẬN THANH KHOẢN TỪ DỮ LIỆU VĨ MÔ (CPI & Tín dụng)")
        print("="*65)
        
        try:
            # 1. Đọc Lạm phát (CPI YoY)
            cpi_path = self.parquet_dir / 'macro/vn_cpi.parquet'
            df_cpi = self._load_parquet_safe(cpi_path)
            current_cpi = 0.0
            if not df_cpi.empty and 'name' in df_cpi.columns:
                df_cpi_yoy = df_cpi[df_cpi['name'] == 'So sánh với cùng kỳ năm trước'].dropna(subset=['value'])
                if not df_cpi_yoy.empty:
                    current_cpi = float(df_cpi_yoy.sort_values('time').iloc[-1]['value'])
            
            # 2. Đọc Tăng trưởng Tín dụng (MS)
            ms_path = self.parquet_dir / 'macro/vn_ms.parquet'
            df_ms = self._load_parquet_safe(ms_path)
            current_credit = 0.0
            if not df_ms.empty and 'name' in df_ms.columns:
                df_credit = df_ms[df_ms['name'] == 'Tăng trưởng tín dụng'].dropna(subset=['value'])
                if not df_credit.empty:
                    current_credit = float(df_credit.sort_values('time').iloc[-1]['value'])
                    
            print(f"[*] Lạm phát (CPI YoY) hiện tại : {current_cpi}%")
            print(f"[*] Tăng trưởng Tín dụng hiện tại: {current_credit}%")
            
            # 3. CHẤM ĐIỂM KỊCH BẢN VĨ MÔ
            if current_cpi >= 4.0:
                self.macro_status = "TIGHTENING (RỦI RO HÚT TIỀN)"
                self.macro_buy_threshold_adj = 15  # Tăng độ khó để mua (Ép không cho mua bậy)
                self.macro_risk_factor = 0.5       # Cắt giảm một nửa tỷ trọng vốn
                print("   🚨 CẢNH BÁO: Lạm phát cao > 4.0%. NHNN có nguy cơ thắt chặt.")
                print("   => HÀNH ĐỘNG: Kích hoạt chế độ PHÒNG THỦ (Khó mua hơn, mua ít tiền hơn)!")
                
            elif current_cpi < 3.5 and current_credit >= 8.0:
                self.macro_status = "EASING (BƠM TIỀN MẠNH)"
                self.macro_buy_threshold_adj = -10 # Nới lỏng điểm mua (Dễ mua hơn)
                self.macro_risk_factor = 1.5       # Bơm thêm 50% vốn giải ngân
                print("   🌊 XÁC NHẬN: Tiền rẻ ngập thị trường. Lạm phát an toàn.")
                print("   => HÀNH ĐỘNG: Kích hoạt chế độ TẤN CÔNG (Dễ mua hơn, giải ngân mạnh tay)!")
                
            else:
                self.macro_status = "NEUTRAL (TRUNG LẬP)"
                print("   ⚖️ XÁC NHẬN: Vĩ mô ổn định. Không có đột biến.")
                print("   => HÀNH ĐỘNG: Duy trì chế độ BÌNH THƯỜNG.")
                
        except Exception as e:
            print(f"[!] Lỗi khi đọc dữ liệu Vĩ mô: {e}")

    def _evaluate_sector_themes(self):
        """Tự động phát hiện Sóng ngành dựa trên tăng trưởng Vĩ mô YoY"""
        print("\n" + "="*65)
        print(" 🎯 ĐÁNH GIÁ SÓNG NGÀNH TỰ ĐỘNG (MACRO SECTOR ROTATION)")
        print("="*65)

        themes_activated = []

        # 1. Rổ cổ phiếu mặc định (Fallback nếu file parquet lỗi)
        TICKERS_KCN = ['SZC', 'IDC', 'KBC', 'VGC', 'PHR', 'GVR', 'SIP']
        TICKERS_RETAIL = ['MWG', 'PNJ', 'DGW', 'FRT', 'PET', 'MSN']
        TICKERS_EXPORT = ['VHC', 'ANV', 'FMC', 'TNG', 'MSH', 'HAH', 'GMD']

        # 2. ĐỘNG CƠ QUÉT NGÀNH TỰ ĐỘNG (Dynamic ICB Mapping)
        try:
            df_ind = self._load_parquet_safe(self.parquet_dir / 'macro/groups_by_industries.parquet')
            if not df_ind.empty and 'symbol' in df_ind.columns:
                # Gộp các cấp ngành (name2, name3, name4) lại thành một chuỗi để dễ quét từ khóa
                df_ind['icb_search'] = df_ind['icb_name2'].fillna('') + " " + df_ind['icb_name3'].fillna('') + " " + df_ind['icb_name4'].fillna('')
                df_ind['icb_search'] = df_ind['icb_search'].str.lower()

                # BỘ LỌC TỪ KHÓA ĐA CHIỀU
                # a. Khu Công Nghiệp
                mask_kcn = df_ind['icb_search'].str.contains('khu công nghiệp', na=False)
                if mask_kcn.any():
                    TICKERS_KCN = df_ind[mask_kcn]['symbol'].unique().tolist()

                # b. Bán lẻ
                mask_retail = df_ind['icb_search'].str.contains('bán lẻ', na=False)
                if mask_retail.any():
                    TICKERS_RETAIL = df_ind[mask_retail]['symbol'].unique().tolist()

                # c. Xuất khẩu (Bao gồm Thủy sản, Dệt may, Cảng biển, Vận tải)
                mask_export = df_ind['icb_search'].str.contains('thủy sản|dệt may|dệt|vận tải|cảng', na=False)
                if mask_export.any():
                    TICKERS_EXPORT = df_ind[mask_export]['symbol'].unique().tolist()
                    
                print(f"[*] Phân loại theo ngành (ICB Database): KCN ({len(TICKERS_KCN)} mã), Bán lẻ ({len(TICKERS_RETAIL)} mã), Xuất khẩu & Logistics ({len(TICKERS_EXPORT)} mã).")
            else:
                print("[-] Không tìm thấy file groups_by_industries.parquet. Sử dụng rổ cổ phiếu đại diện mặc định.")
        except Exception as e:
            print(f"[!] Lỗi khi quét danh mục ngành ICB: {e}. Sử dụng rổ mặc định.")

        # 3. QUÉT VĨ MÔ ĐỂ KÍCH HOẠT THEMATIC
        try:
            # A. SÓNG KHU CÔNG NGHIỆP (Dựa trên FDI Giải ngân)
            fdi_path = self.parquet_dir / 'macro/vn_fdi.parquet'
            df_fdi = self._load_parquet_safe(fdi_path)
            if not df_fdi.empty:
                # Lấy số liệu giải ngân và tính YoY (So với cùng kỳ năm ngoái -> lùi 12 tháng)
                df_disburse = df_fdi[(df_fdi['group_name'] == 'Tổng FDI') & (df_fdi['name'] == 'Giải ngân')].dropna(subset=['value']).sort_values('time')
                if len(df_disburse) >= 13:
                    curr_val = float(df_disburse.iloc[-1]['value'])
                    last_year_val = float(df_disburse.iloc[-13]['value'])
                    yoy = (curr_val - last_year_val) / last_year_val * 100
                    print(f"[*] FDI Giải ngân YoY: {yoy:+.2f}%")
                    
                    if yoy > 5.0: # FDI tăng trưởng > 5% là một cú hích lớn
                        self.dynamic_thematic_tickers.extend(TICKERS_KCN)
                        themes_activated.append(f"Khu Công Nghiệp (FDI {yoy:+.1f}%)")

            # B. SÓNG BÁN LẺ TIÊU DÙNG (Dựa trên Doanh thu Bán lẻ)
            retail_path = self.parquet_dir / 'macro/vn_retail.parquet'
            df_retail = self._load_parquet_safe(retail_path)
            if not df_retail.empty:
                df_total = df_retail[df_retail['name'] == 'TỔNG SỐ:'].dropna(subset=['value']).sort_values('time')
                if len(df_total) >= 13:
                    curr_val = float(df_total.iloc[-1]['value'])
                    last_year_val = float(df_total.iloc[-13]['value'])
                    yoy = (curr_val - last_year_val) / last_year_val * 100
                    print(f"[*] Doanh thu Bán lẻ YoY: {yoy:+.2f}%")
                    
                    if yoy > 8.0: # Bán lẻ tăng trên 8% chứng tỏ dân đang tiêu tiền mạnh
                        self.dynamic_thematic_tickers.extend(TICKERS_RETAIL)
                        themes_activated.append(f"Bán Lẻ (Doanh thu {yoy:+.1f}%)")

            # C. SÓNG XUẤT KHẨU & CẢNG BIỂN (Dựa trên Tổng trị giá Xuất khẩu)
            ie_path = self.parquet_dir / 'macro/vn_ie.parquet'
            df_ie = self._load_parquet_safe(ie_path)
            if not df_ie.empty:
                df_export = df_ie[df_ie['name'] == 'Tổng trị giá Xuất khẩu'].dropna(subset=['value']).sort_values('time')
                if len(df_export) >= 13:
                    curr_val = float(df_export.iloc[-1]['value'])
                    last_year_val = float(df_export.iloc[-13]['value'])
                    yoy = (curr_val - last_year_val) / last_year_val * 100
                    print(f"[*] Tổng Xuất khẩu YoY: {yoy:+.2f}%")
                    
                    if yoy > 5.0:
                        self.dynamic_thematic_tickers.extend(TICKERS_EXPORT)
                        themes_activated.append(f"Xuất Khẩu & Logistics (Trị giá {yoy:+.1f}%)")

        except Exception as e:
            print(f"[!] Lỗi khi đánh giá sóng ngành: {e}")

        # 🌟 TÍCH HỢP SMART MONEY SECTOR ROTATION TỪ REPORTER.PY
        try:
            print("\n" +"="*65)
            print(" 💸 ĐÁNH GIÁ DÒNG TIỀN VÀ SÓNG NGÀNH - TRONG 5 PHIÊN GẦN NHẤT")
            print("="*65)
            
            reporter = GroupCashFlowReporter(self.df_foreign, self.df_prop, self.df_ind, self.df_price, verbose=False)
            sector_flow, flow_report_df = reporter.generate_report(timeframe='week')
            
            if sector_flow is not None and not sector_flow.empty:
                # 1. TÌM TOP 3 NGÀNH HÚT TIỀN NHẤT ĐỂ ĐÁNH LÊN (LONG)
                top_3_sectors = sector_flow.head(3)['industry'].tolist()
                
                # Quét tìm tất cả các mã cổ phiếu thuộc 3 ngành này
                hot_tickers = flow_report_df[flow_report_df['industry'].isin(top_3_sectors)]['ticker'].tolist()
                
                # Bơm các mã này vào rổ THEMATIC ĐỘNG để được ưu tiên cộng 20đ
                self.dynamic_thematic_tickers.extend(hot_tickers)
                themes_activated.append(f"Smart Money Gom Tuần ({', '.join(top_3_sectors)})")
                
                # 2. TÌM TOP 2 NGÀNH BỊ XẢ RÁT NHẤT ĐỂ ĐƯA VÀO DANH SÁCH ĐEN NGÀNH (SECTOR BLACKLIST)
                bottom_2_sectors = sector_flow.tail(2)['industry'].tolist()
                self.sector_blacklist_tickers = flow_report_df[flow_report_df['industry'].isin(bottom_2_sectors)]['ticker'].tolist()
                print(f"   [!] Đưa các mã thuộc ngành ({', '.join(bottom_2_sectors)}) vào Danh sách Phạt (Bị xả ròng).")

        except Exception as e:
            print(f"   [!] Lỗi khi nhúng Smart Money Sector Flow: {e}")
            self.sector_blacklist_tickers = [] # Tránh lỗi nếu không chạy được

        # Lọc trùng lặp mã cổ phiếu (Trường hợp 1 mã nằm ở 2 list)
        self.dynamic_thematic_tickers = list(set(self.dynamic_thematic_tickers))

        if themes_activated:
            print(f"   => 🚀 KÍCH HOẠT THEMATIC (+20đ): {', '.join(themes_activated)}")
        else:
            print("   => ⚖️ Không có sóng ngành vĩ mô nào đủ mạnh được kích hoạt tháng này.")
        print("="*65)

    def _set_seasonality_rules(self):
        current_month = datetime.now().month
        
        if current_month in [1, 2, 3]:
            # Q1: Chậm rãi, Thận trọng trước rủi ro chính sách
            self.season = "Q1_DEFENSIVE"
            self.buy_threshold = BASE_SCORE_THRESHOLD + 10 # 80đ mới mua (Cực kỳ khắt khe)
            self.risk_factor = 0.5 # Chỉ giải ngân 50% quy mô thông thường
        elif current_month in [4, 5, 6, 7, 8, 9]:
            # Q2 & Q3: Ổn định, Tấn công thu gom tài sản
            self.season = "Q2_Q3_AGGRESSIVE"
            self.buy_threshold = BASE_SCORE_THRESHOLD # 70đ là mua
            self.risk_factor = 1.0 # Đánh đủ 100% tỷ trọng
        else:
            # Q4: Chốt lãi, Thu quân
            self.season = "Q4_HARVEST"
            self.buy_threshold = BASE_SCORE_THRESHOLD + 5 # 75đ mới mua
            self.risk_factor = 0.7 # Hạ tỷ trọng mua mới

        # ĐIỀU CHỈNH THEO RỔ CỔ PHIẾU (UNIVERSE TWEAKS)
        if getattr(self, 'universe', 'ALL') == "VN30":
            self.buy_threshold -= 5
            self.risk_factor *= 1.2 # Cho phép đánh vốn lớn hơn 20%
        elif getattr(self, 'universe', 'ALL') == "VNMidCap":
            self.buy_threshold += 5
            self.risk_factor *= 0.7 # Cắt bớt 30% vốn mỗi lệnh
        elif getattr(self, 'universe', 'ALL') == "VNSmallCap":
            self.buy_threshold += 15
            self.risk_factor *= 0.3 # Cắt bớt 70% vốn mỗi lệnh
            
        print(f"\n[***] Chế độ Vận hành: {self.season} | Rổ: {getattr(self, 'universe', 'ALL')} | Điểm mua: {self.buy_threshold} | Tỷ trọng: {self.risk_factor*100:.0f}%")

    def get_top_bond_funds(self, top_n=2):
        """
        Quét API Fmarket để lấy danh sách Quỹ Trái Phiếu tốt nhất hiện tại
        dựa trên Lợi nhuận 12 tháng gần nhất (nav_change_12m).
        """
        df_bond = self.df_funds
        if df_bond.empty:
            print("   [!] Không tìm thấy Quỹ Trái phiếu nào trong danh sách.")
            return []

        # 2. XÁC ĐỊNH CỘT LỢI NHUẬN VÀ SẮP XẾP
        return_col = 'nav_change_12m' if 'nav_change_12m' in df_bond.columns else None
        
        if not return_col:
            # Fallback tìm cột có chứa chữ 12m
            return_col = next((c for c in df_bond.columns if '12m' in str(c).lower() and 'annual' not in str(c).lower()), None)

        if return_col:
            # Ép kiểu float, MẤU CHỐT LÀ ĐIỀN 0 CHO CÁC QUỸ NULL ĐỂ KHÔNG BỊ LỖI
            df_bond[return_col] = pd.to_numeric(df_bond[return_col], errors='coerce').fillna(0)
            # Sắp xếp giảm dần (Quỹ lãi cao nhất lên đầu)
            df_bond = df_bond.sort_values(return_col, ascending=False)
            
        # 3. TRÍCH XUẤT THÔNG TIN TOP N
        top_funds = []
        for _, row in df_bond.head(top_n).iterrows():
            # Lấy tên viết tắt (short_name) của quỹ
            fund_name = row.get('short_name', row.get('fund_code', 'Unknown'))
            yield_pct = row.get(return_col, 0) if return_col else 0
            
            top_funds.append({
                'fund_name': fund_name,
                'expected_yield': float(yield_pct)
            })
            
        return top_funds

    def _analyze_macro_conditions(self):
        self.macro_status = "BÌNH THƯỜNG"
        self.macro_warnings = []
        
        # Lấy thông số cơ sở từ Seasonality (Quý)
        base_threshold = self.buy_threshold
        base_risk = self.risk_factor

        try:
            # -----------------------------------------------------
            # 1. ĐÁNH GIÁ TỶ GIÁ USD/VND (Sát thủ của thị trường chứng khoán)
            # -----------------------------------------------------
            fx_path = self.parquet_dir / 'macro/usd_vnd.parquet'
            if fx_path.exists():
                df_fx = pd.read_parquet(fx_path)
                
                # 1. Chuyển đổi thời gian từ Epoch ms
                if pd.api.types.is_numeric_dtype(df_fx['time']):
                    df_fx['time'] = pd.to_datetime(df_fx['time'], unit='ms').dt.normalize()
                else:
                    df_fx['time'] = pd.to_datetime(df_fx['time']).dt.normalize()
                
                # 2. Lọc bỏ các dòng null và chỉ lấy "Tỷ giá trung tâm"
                df_fx = df_fx.dropna(subset=['value'])
                if 'name' in df_fx.columns:
                    df_fx = df_fx[df_fx['name'].str.contains("trung tâm", na=False, case=False)]
                
                # 3. Sắp xếp và lấy dữ liệu
                df_fx = df_fx.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                
                if len(df_fx) >= 6:
                    current_fx = float(df_fx.iloc[-1]['value'])
                    fx_1w_ago = float(df_fx.iloc[-6]['value']) # So với 1 tuần trước (5-6 phiên)
                    
                    fx_roc_1w = (current_fx - fx_1w_ago) / fx_1w_ago * 100
                    
                    if fx_roc_1w > 0.5:
                        self.macro_warnings.append(f"Tỷ giá USD/VND căng thẳng ({current_fx:,.0f}đ, +{fx_roc_1w:.2f}%/tuần)")
                        base_threshold += 10  # Ép điểm mua khắt khe hơn
                        base_risk *= 0.5      # Cắt giảm một nửa quy mô giải ngân
                        
            # -----------------------------------------------------
            # 2. ĐÁNH GIÁ GIÁ DẦU THÔ (Chỉ báo Lạm phát & Lãi suất)
            # -----------------------------------------------------
            oil_path = self.parquet_dir / 'macro/crude_oil.parquet'
            if oil_path.exists():
                df_oil = pd.read_parquet(oil_path)
                
                # 1. Chuyển đổi thời gian từ Epoch ms
                if pd.api.types.is_numeric_dtype(df_oil['time']):
                    df_oil['time'] = pd.to_datetime(df_oil['time'], unit='ms').dt.normalize()
                else:
                    df_oil['time'] = pd.to_datetime(df_oil['time']).dt.normalize()
                
                df_oil = df_oil.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                
                if len(df_oil) >= 20:
                    current_oil = float(df_oil.iloc[-1]['close']) # Giá trị nằm ở cột 'close'
                    oil_1m_ago = float(df_oil.iloc[-21]['close']) # So với ~1 tháng trước (20-21 phiên)
                    
                    oil_roc_1m = (current_oil - oil_1m_ago) / oil_1m_ago * 100
                    
                    if current_oil > 85 or oil_roc_1m > 10:
                        self.macro_warnings.append(f"Giá dầu neo cao/tăng nóng (${current_oil:.1f}, +{oil_roc_1m:.1f}%/tháng)")
                        base_threshold += 5
                        base_risk *= 0.7 # Hạ tỷ trọng 30%

        except Exception as e:
            print(f"[!] Lỗi phân tích Vĩ mô: {e}")

        # -----------------------------------------------------
        # CẬP NHẬT LẠI THÔNG SỐ VÀ IN RA BÁO CÁO
        # -----------------------------------------------------
        print("\n" + "="*65)
        print(" 🌍 ĐÁNH GIÁ VĨ MÔ DỰA THEO TỈ GIÁ USD VÀ GIÁ DẦU")
        print("="*65)
        
        if self.macro_warnings:
            self.macro_status = "PHÒNG THỦ CAO (RỦI RO VĨ MÔ)"
            self.buy_threshold = min(base_threshold, 95)
            self.risk_factor = max(base_risk, 0.0)
            
            print(f"   🔴 TRẠNG THÁI: {self.macro_status}")
            for w in self.macro_warnings:
                print(f"      ⚠️ LƯU Ý: {w}")
        else:
            self.macro_status = "ỦNG HỘ (TÍCH CỰC)"
            print(f"   🟢 TRẠNG THÁI: {self.macro_status}")
            print("      ✅ Tỷ giá ổn định, Giá dầu trong tầm kiểm soát.")
            
        # print("-" * 65)
        print(f"   🎯 Điểm Mua (sau đánh giá): {self.buy_threshold}đ")
        print(f"   💰 Tỷ trọng (sau đánh giá): {self.risk_factor*100:.0f}%")
        print("="*65)

    def _check_market_regime(self):
        """Kiểm tra Xu hướng thị trường chung (VN-Index) từ Parquet"""
        print("\n" + "="*65)
        print(" 🏛️ KIỂM TRA XU HƯỚNG THỊ TRƯỜNG CHUNG (VN-INDEX)")
        print("="*65)
        
        vnindex_path = self.parquet_dir / 'macro/vnindex.parquet'
        
        if not vnindex_path.exists():
            print("[!] Không tìm thấy file vnindex.parquet. Bỏ qua bộ lọc thị trường chung.")
            return True
            
        try:
            df_vnindex = pd.read_parquet(vnindex_path)
            
            # 1. Chuẩn hóa thời gian và làm sạch
            if pd.api.types.is_numeric_dtype(df_vnindex['time']):
                df_vnindex['time'] = pd.to_datetime(df_vnindex['time'], unit='ms').dt.normalize()
            else:
                df_vnindex['time'] = pd.to_datetime(df_vnindex['time']).dt.normalize()
                
            df_vnindex = df_vnindex.sort_values('time').drop_duplicates(subset=['time'], keep='last')
            df_vnindex = df_vnindex.dropna(subset=['close'])
            
            # 2. Tính toán Logic
            if len(df_vnindex) >= 89:
                # Tính đường trung bình động dài hạn EMA89
                df_vnindex['ema89'] = df_vnindex['close'].ewm(span=89, adjust=False).mean()
                
                current_close = float(df_vnindex.iloc[-1]['close'])
                current_ema89 = float(df_vnindex.iloc[-1]['ema89'])
                
                if current_close < current_ema89:
                    print(f"🚨 CẢNH BÁO: VN-Index ({current_close:,.1f}) NẰM DƯỚI EMA89 ({current_ema89:,.1f}).")
                    print("   => BẬT CÔNG TẮC ĐÓNG BĂNG: Dừng giải ngân mua mới!")
                    print("="*65)
                    return False # Downtrend
                else:
                    print(f"✅ UPTREND: VN-Index ({current_close:,.1f}) NẰM TRÊN EMA89 ({current_ema89:,.1f}).")
                    print("   => Xu hướng ủng hộ: Cho phép quét tín hiệu Wyckoff.")
                    print("="*65)
                    return True # Uptrend
            else:
                print(f"[!] Dữ liệu VN-Index quá ngắn ({len(df_vnindex)} phiên), chưa đủ để vẽ EMA89.")
                
        except Exception as e:
            print(f"[!] Lỗi khi phân tích VN-Index: {e}")
        
        # Mặc định cho phép chạy nếu lỗi để không làm "đứng" cả hệ thống
        return True

    def _get_fundamental_data(self, ticker):
        """Trích xuất dữ liệu CANSLIM từ DataFrame BCTC (Đã cập nhật theo cấu trúc vnstock_data)"""
        # Kiểm tra xem dữ liệu BCTC có tồn tại trong RAM không
        if self.df_fin.empty or 'ticker' not in self.df_fin.columns: return None
        
        # 1. Lọc dữ liệu theo mã cổ phiếu hiện tại
        df_ticker = self.df_fin[self.df_fin['ticker'] == ticker]
        if df_ticker.empty: return None
        
        # 2. Tìm cột Quý gần nhất (Ví dụ: '2025-Q4', '2024-Q3')
        # Lọc các cột chứa ký tự '-' và 'Q' để tự động nhận diện cột thời gian
        quarter_cols = [col for col in df_ticker.columns if '-' in str(col) and 'Q' in str(col).upper()]
        if not quarter_cols: return None
        
        # Sắp xếp giảm dần để lấy quý mới nhất (vd: '2025-Q4' sẽ nằm ở index 0)
        latest_quarter = sorted(quarter_cols, reverse=True)[0]
        
        # 3. Chuyển đổi DataFrame thành Dictionary: { 'item_id': giá_trị_tại_quý_mới_nhất }
        # Việc này giúp truy xuất các chỉ số bằng key cực kỳ dễ dàng (giống JSON gốc)
        df_clean = df_ticker.dropna(subset=['item_id'])
        metrics = df_clean.set_index('item_id')[latest_quarter].to_dict()
        
        try:
            # 4. Trích xuất các chỉ số CANSLIM
            # Tăng trưởng lợi nhuận sau thuế (Đại diện cho C - Current Earnings)
            eps_growth = metrics.get('profit_after_tax_for_shareholders_of_the_parent_company', 0)
            if eps_growth is None: eps_growth = 0
            
            # Hiệu quả sử dụng vốn (ROE)
            roe = metrics.get('roe', metrics.get('roe_trailling', 0))
            if roe is None: roe = 0
            
            # Định giá (P/E)
            pe = metrics.get('p_e', 0)
            if pe is None: pe = 0
            
            # LƯU Ý QUAN TRỌNG: 
            # Dữ liệu từ vnstock_data trả về % dưới dạng số nguyên/thập phân (VD: ROE 21.53 nghĩa là 21.53%)
            # Chúng ta cần chia 100 để khớp với logic chấm điểm >= 0.15 của hàm calculate_confluence_score
            return {
                "quarter": latest_quarter, 
                "eps_change": float(eps_growth) / 100.0, 
                "roe": float(roe) / 100.0,               
                "pe": float(pe)                
            }
        except Exception as e:
            print(f"[!] Lỗi trích xuất CANSLIM {ticker}: {e}")
            return None

    def _get_market_sentiment(self, ticker):
        """Trích xuất Order Book & Foreign Flow từ DataFrame Bảng giá"""
        if self.df_board.empty or 'symbol' not in self.df_board.columns: return None

        # df_ticker = self.df_board[self.df_board['ticker'] == ticker]
        df_ticker = self.df_board[self.df_board['symbol'] == ticker]
        if df_ticker.empty: return None

        row = df_ticker.iloc[0]
        
        # HÀM CHUYỂN ĐỔI AN TOÀN (CHỐNG CRASH TỪ CHUỖI "ATO", "ATC", DỮ LIỆU RỖNG)
        def safe_float(val, default=0.0):
            if pd.isna(val): return float(default)
            val_str = str(val).strip().upper()
            if val_str in ["", "ATO", "ATC", "NULL", "NONE"]:
                return float(default)
            try:
                return float(val)
            except ValueError:
                return float(default)

        try:
            bid_vol = safe_float(row.get('bid_vol_1', 0)) + safe_float(row.get('bid_vol_2', 0)) + safe_float(row.get('bid_vol_3', 0))
            ask_vol = safe_float(row.get('ask_vol_1', 0)) + safe_float(row.get('ask_vol_2', 0)) + safe_float(row.get('ask_vol_3', 0))
            foreign_net_vol = safe_float(row.get('foreign_buy_volume', 0)) - safe_float(row.get('foreign_sell_volume', 0))
            foreign_net_value = safe_float(row.get('average_price', 0)) * foreign_net_vol
            net_foreign = 0
            if foreign_net_value > 0 or foreign_net_value < 0:
                net_foreign = foreign_net_value / 1_000_000_000
            
            # Lấy thông tin giá trần/sàn và bid/ask tốt nhất
            return {
                "sentiment_ratio": bid_vol / ask_vol if ask_vol > 0 else 10.0,
                "net_foreign_vol": foreign_net_vol,
                "net_foreign_value": foreign_net_vol,
                "net_foreign": net_foreign, # khối ngoại mua ròng theo tỷ
                "best_bid": safe_float(row.get('bid_price_1', row.get('close_price', 0))),  # Giá đang chờ mua cao nhất
                "best_ask": safe_float(row.get('ask_price_1', row.get('close_price', 0))),  # Giá đang chào bán rẻ nhất
                "ceil": safe_float(row.get('ceiling_price', row.get('high_price', 0))),
                "floor": safe_float(row.get('floor_price', row.get('low_price', 0)))
            }
        except Exception as e: 
            print(f"_get_market_sentiment: Error -> {e}")
            return None

    def _calculate_volume_profile(self, df_ticker_price, lookback_days=130):
        """
        Tính POC, VAL (Value Area Low) và VAH (Value Area High).
        """
        if df_ticker_price is None or df_ticker_price.empty:
            return 0.0, 0.0, 0.0
        
        df_recent = df_ticker_price.sort_values('time').tail(lookback_days).copy()
        if df_recent.empty:
            return 0.0, 0.0, 0.0
            
        # Gom nhóm theo mức giá (bin_size = 0.5% giá hiện tại để linh hoạt cho cả Bluechip & Penny)
        current_p = df_recent['close'].iloc[-1]
        bin_size = max(0.1, current_p * 0.005) 
        df_recent['price_bin'] = (df_recent['close'] / bin_size).round() * bin_size
        
        vol_profile = df_recent.groupby('price_bin')['volume'].sum().sort_index()
        if vol_profile.empty:
            return 0.0, 0.0, 0.0
            
        poc_price = float(vol_profile.idxmax())
        
        # Tìm Value Area (70% Khối lượng)
        total_vol = vol_profile.sum()
        target_vol = total_vol * 0.70
        
        # Sắp xếp các mức giá theo khối lượng từ cao xuống thấp
        sorted_profile = vol_profile.sort_values(ascending=False)
        cum_vol = sorted_profile.cumsum()
        
        # Lọc ra các mức giá nằm trong 70% khối lượng lớn nhất
        value_area_bins = cum_vol[cum_vol <= target_vol].index
        
        if len(value_area_bins) > 0:
            val = float(min(value_area_bins))
            vah = float(max(value_area_bins))
        else:
            val = vah = poc_price
            
        return poc_price, val, vah

    def update_and_prepare_data(self, df_price, df_intra):
        """Gộp dữ liệu Lịch sử Giá và Khớp lệnh Intraday thành 1 file duy nhất cho Forecaster"""
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang tổng hợp Nến Provisional...")
        
        if df_price.empty:
            print("[!] Không tìm thấy dữ liệu giá lịch sử.")
            return None

        # 1. CHUẨN HÓA THỜI GIAN LỊCH SỬ GIÁ
        # Dữ liệu time có thể lưu ở dạng epoch (ms) như trong file JSON hoặc datetime
        if pd.api.types.is_numeric_dtype(df_price['time']):
            df_price['time'] = pd.to_datetime(df_price['time'], unit='ms')
        else:
            df_price['time'] = pd.to_datetime(df_price['time'])
            
        # Đưa thời gian về 00:00:00 để so sánh chuẩn theo ngày
        df_price['time'] = df_price['time'].dt.normalize() 

        # 2. XỬ LÝ INTRADAY THÀNH NẾN NGÀY (OHLCV)
        if not df_intra.empty:
            if pd.api.types.is_numeric_dtype(df_intra['time']):
                df_intra['time'] = pd.to_datetime(df_intra['time'], unit='ms')
            else:
                df_intra['time'] = pd.to_datetime(df_intra['time'])
                
            # Chỉ lấy dữ liệu khớp lệnh của ngày gần nhất (hôm nay)
            latest_date = df_intra['time'].dt.date.max()
            df_today = df_intra[df_intra['time'].dt.date == latest_date].copy()
            
            if not df_today.empty:
                # Sắp xếp đúng theo thời gian khớp lệnh để lấy chuẩn giá Mở/Đóng cửa
                df_today = df_today.sort_values(by=['ticker', 'time'])
                
                # TỔNG HỢP NẾN BẰNG PANDAS AGGREGATION
                today_candles = df_today.groupby('ticker').agg(
                    open=('price', 'first'),   # Giá khớp đầu tiên
                    high=('price', 'max'),     # Giá cao nhất
                    low=('price', 'min'),      # Giá thấp nhất
                    close=('price', 'last'),   # Giá khớp cuối cùng
                    volume=('volume', 'sum')   # Tổng khối lượng
                ).reset_index()
                
                # Gán time cho nến intraday là ngày hôm nay (00:00:00)
                today_candles['time'] = pd.to_datetime(latest_date)
                
                # 3. MERGE VÀO LỊCH SỬ GIÁ
                df_combined = pd.concat([df_price, today_candles], ignore_index=True)
                
                # Nếu master_price đã có nến ngày hôm nay rồi, Pandas sẽ ghi đè bằng nến Intraday mới nhất (nhờ keep='last')
                df_combined = df_combined.sort_values(['ticker', 'time']).drop_duplicates(subset=['ticker', 'time'], keep='last')
                df_price = df_combined
        
        # ==========================================================
        # 🛡️ BỘ LỌC NHIỄU & THANH KHOẢN (NOISE & LIQUIDITY FILTER)
        # ==========================================================
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Lọc Nhiễu (Loại bỏ hàng bo cung, thanh khoản kém)...")
        
        # 1. Bộ lọc Cổ phiếu Trưởng thành (Tối thiểu 89 phiên để vẽ EMA)
        ticker_counts = df_price['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= 89].index
        df_clean = df_price[df_price['ticker'].isin(valid_tickers)]
        
        # 2. Bộ lọc Thanh khoản & Nến dị dạng (Dựa trên 20 phiên gần nhất)
        # Tách 20 phiên cuối của mỗi mã để đánh giá phong độ hiện tại
        df_last_20 = df_clean.groupby('ticker').tail(20)
        
        # Tính toán bằng cơ chế Vectorized siêu tốc của Pandas
        metrics = df_last_20.groupby('ticker').agg(
            adv=('volume', 'mean'),         # Khối lượng trung bình 20 phiên
            avg_price=('close', 'mean'),    # Giá đóng cửa trung bình
            # Đếm số nến gạch ngang (High == Low) => Dấu hiệu tiết cung/nhốt sàn
            flat_days=('high', lambda x: (x == df_last_20.loc[x.index, 'low']).sum())
        )
        
        # Công thức tính Giá trị = ADV * Avg_Price
        # Đặt ngưỡng tối thiểu là 3 Tỷ VNĐ / phiên -> Tương đương hệ số 3,000,000,000
        metrics['adv_value'] = metrics['adv'] * metrics['avg_price']
        
        # 3. ĐIỀU KIỆN TỬ THẦN (KẾT LIỄU NHỮNG KẺ THAO TÚNG)
        # - adv_value >= 3_000_000_000 (Thanh khoản > 3 Tỷ VNĐ)
        # - flat_days <= 2 (Không quá 2 phiên nến gạch ngang trong 1 tháng)
        if self.universe == "VN30":
            tradable_tickers = metrics[
                (metrics['adv_value'] >= 100_000_000_000) 
                & (metrics['flat_days'] <= 2)
            ].index
        elif self.universe == "VNMidCap":
            tradable_tickers = metrics[
                (metrics['adv_value'] >= 50_000_000_000) 
                & (metrics['flat_days'] <= 2)
            ].index
        elif self.universe == "VNSmallCap":
            tradable_tickers = metrics[
                (metrics['adv_value'] >= 3_000_000_000) 
                & (metrics['flat_days'] <= 2)
            ].index
        elif self.universe == "HOSE" or self.universe == "ALL":
            tradable_tickers = metrics.index
        
        # Ghi đè lại df_price chỉ chứa các mã tinh khiết nhất
        df_price = df_clean[df_clean['ticker'].isin(tradable_tickers)]
        print(f"[*] Đã thanh lọc: Giữ lại {len(tradable_tickers)}/{len(ticker_counts)} mã đạt chuẩn Tổ chức lớn.")
        # ==========================================================

        # ko lưu mà trả về DataFrame trực tiếp để xử lý ngay trên RAM
        return df_price

    def calculate_confluence_score(self, row, board_info, fund_info, sm_result, mf_result, vol_profile, omni_result=None):
        """Hệ thống chấm điểm """
        score = 0
        details = []
        ticker = row['Ticker']
        signal = row['Signal']
        price = row['Price']
        poc_price, val_p, vah_p = vol_profile

        # 1. CORE SIGNAL (Tối đa 40đ)
        if signal in ['SOS', 'SPRING', 'TEST_CUNG']:
            score += 30; details.append("Tín hiệu Chuẩn (+30)")
            if signal == 'SOS' and row.get('VPA_Status') in ['High', 'Ultra High']:
                score += 10; details.append("Vol Uy tín (+10)")
            elif signal == 'TEST_CUNG':
                if row.get('VPA_Status') == 'Low':
                    score += 10
                    details.append("Thanh khoản cạn kiệt (< 60% MA20) (+10 điểm)")
                elif row.get('VPA_Status') == 'Normal':
                    score += 5
                    details.append("Cầu đang chờ xác nhận (+5 điểm)")
                elif row.get('VPA_Status') in ['High', 'Ultra High']:
                    # Nếu Test Cung mà Volume lại cao, chứng tỏ thuật toán nhận diện bị nhiễu 
                    # Hoặc đây là lực Bán ngầm (Hidden Supply) chứ không phải Test Cung.
                    score -= 10 
                    details.append("Tín hiệu nhiễu! Nến giảm biên độ hẹp nhưng Volume cao -> Phân phối ngầm! (-10 điểm)")
            elif signal == 'SPRING' and row.get('VPA_Status') == 'Low':
                score += 10; details.append("Vol Cạn (+10)")

        # EFFORT VS RESULT (Định luật 3 Wyckoff)
        spread = float(row.get('Spread', 0))
        #   Biến spread_pct đo lường tỷ lệ biên độ nến so với thị giá 
        #   (Giúp so sánh công bằng giữa Bluechip giá 100k và Penny giá 10k)
        spread_pct = (spread / price) if price > 0 else 0
        vol_z = float(row.get('Vol_Z_Score', 0))
        if signal == 'SOS':
            # Nỗ lực lớn (Volume đột biến > 1.5 StdDev) nhưng biên độ giá hẹp (< 1.5%)
            if vol_z > 1.5 and spread_pct < 0.015:
                score -= 15
                details.append(f"⚠️ Bất thường: Khối lượng nổ lớn (Z: {vol_z:.1f}) nhưng nến hẹp (Spread: {spread_pct*100:.1f}%) -> Cung ngầm chặn trên (-15)")
            # Nỗ lực lớn và Kết quả xứng đáng (Biên độ nến mở rộng > 3%)
            elif vol_z > 1.5 and spread_pct >= 0.03:
                score += 10
                details.append(f"Đẩy giá dứt khoát (Z: {vol_z:.1f}, Spread: {spread_pct*100:.1f}%) (+10)")

        # 2. VĨ MÔ, CÂU CHUYỆN & MA TRẬN DÒNG TIỀN (FLOW DIVERGENCE MATRIX)
        # Chuyển string details của Smart Money thành 1 text để dễ dò từ khóa
        sm_context = " | ".join(sm_result["sm_details"]) if sm_result["sm_details"] else ""
        
        # Nhận diện Động lượng Ngắn hạn (1-3 phiên gần nhất)
        is_buying_recently = "gom" in sm_context  # Tây hoặc Tự doanh đang gom ngắn hạn
        is_selling_recently = "xả" in sm_context  # Tây hoặc Tự doanh đang xả ngắn hạn

        # KỊCH BẢN A: CỔ PHIẾU NẰM TRONG TOP NGÀNH HÚT TIỀN (SÓNG TĂNG)
        if ticker in getattr(self, 'dynamic_thematic_tickers', []):
            if is_selling_recently:
                # Hiện tượng: Ngành thì đẹp, Tổng tuần Mua, nhưng 1-2 phiên nay Tay to lén xả! (Bull Trap)
                score -= 10
                details.append("⚠️ Bẫy Kéo Xả: Ngành hút tiền nhưng Mã bị xả lén (-10)")
            else:
                # Hiện tượng: Ngành đẹp, Mã cũng đang được gom đều (Thuận buồm xuôi gió)
                score += 20
                details.append("Hưởng lợi Vĩ mô/Sóng Ngành Thuận (+20)")

        # KỊCH BẢN B: CỔ PHIẾU NẰM TRONG DANH SÁCH ĐEN NGÀNH (BỊ XẢ RÒNG TUẦN)
        elif ticker in getattr(self, 'sector_blacklist_tickers', []):
            if is_buying_recently:
                # Hiện tượng: Cả Tuần/Ngành bị xả, nhưng Mã này đang được mua ngược lại (Rũ bỏ thành công)
                # Đây là Cổ phiếu Dẫn dắt (Leader) đang nảy lên từ đáy!
                score += 15
                details.append("🌟 Rũ bỏ thành công: Ngành bị xả nhưng Mã được gom ngược (+15)")
            else:
                # Hiện tượng: Ngành chết, Mã cũng đang bị xả rát (Tắm máu)
                score -= 15
                details.append("Thuộc Ngành bị xả & Đang bị bán tháo (-15)")

        # 3. TREND FILTER & CHỮ 'N' CANSLIM (Tối đa 20đ)
        ema89 = row.get('EMA89', 0)
        dist_52w = row.get('Dist_to_52W_High_%', -100)
        
        if price > ema89: score += 10; details.append("Uptrend (+10)")
        if dist_52w >= 0:
            score += 10; details.append("Vượt đỉnh 52 Tuần (+10)")
        elif dist_52w >= -5.0:
            score += 5; details.append("Tiệm cận đỉnh 52W (+5)")

        # 4. CANSLIM FUNDAMENTALS (Tối đa 20đ)
        if fund_info:
            if fund_info['eps_change'] >= 0.15: score += 10; details.append("EPS Tăng Mạnh (+10)")
            if fund_info['roe'] >= 0.15: score += 10; details.append("ROE Xuất Sắc (+10)")

        # 5. SMART MONEY FOOTPRINT & INTRADAY T0 (Tối đa +30đ, Thấp nhất -30đ)
        # Áp dụng hệ số từ kết quả Optimizer
        base_sm_score = sm_result.get("total_sm_score", 0)
        if self.universe == "VN30":
            base_sm_score = base_sm_score * 1.0
        elif self.universe == "VNMidCap":
            base_sm_score = base_sm_score * 1.5
        elif self.universe == "VNSmallCap":
            base_sm_score = base_sm_score * 1.5
        elif self.universe == "HOSE" or self.universe == "ALL":
            base_sm_score = base_sm_score * 1.5

        # Giới hạn điểm Smart Money tối đa không vượt quá 30 để tránh lấn át Wyckoff
        sm_score_optimized = max(-30, min(30, base_sm_score)) 

        # Đánh giá đồng thuận T0
        today_foreign_net = board_info.get('net_foreign', 0) if board_info else 0 # Đơn vị: Tỷ VNĐ
        if signal in ['SOS', 'SPRING', 'TEST_CUNG']:
            if sm_result.get("is_danger", False):
                score -= 30
            else:
                score += sm_score_optimized
            
            if sm_result["sm_details"]:
                details.append(" | ".join(sm_result["sm_details"]))

        # 6. VŨ KHÍ BẮN TỈA: VOLUME PROFILE (POC) & GIÁ VỐN CÁ MẬP (VWAP)
        if poc_price > 0:
            dist_to_poc = (price - poc_price) / poc_price * 100
            
            # A. Bệ phóng POC (Giá đang nằm ngay trên hoặc bứt phá khỏi vùng Volume lớn nhất)
            if 0 <= dist_to_poc <= 5.0:
                score += 10
                details.append(f"Tựa nền POC ({poc_price:,.1f}) (+10)")
            elif -3.0 <= dist_to_poc < 0:
                score += 5
                details.append(f"Kéo giật lại POC ({poc_price:,.1f}) (+5)")

            # B. 🌟 ĐÒN HỢP LƯU TỐI THƯỢNG (CONFLUENCE STRIKE) 🌟
            # Nếu mốc POC trùng khớp với Giá vốn của Nhà cái (Sai số 3%)
            sm_vwap = mf_result.get('sm_vwap', 0) if mf_result else 0
            if sm_vwap > 0:
                poc_vwap_diff = abs(poc_price - sm_vwap) / sm_vwap * 100
                if poc_vwap_diff <= 3.0:
                    # POC và VWAP dính chặt vào nhau => "Bê tông cốt thép"
                    # Nếu giá hiện tại cũng nằm gần vùng này thì đây là Siêu Cơ Hội!
                    if abs(price - poc_price) / poc_price * 100 <= 5.0:
                        score += 20
                        details.append(f"🌟 HỢP LƯU TỐI THƯỢNG (POC + Giá vốn nhà cái) (+20)")

        # Lọc bằng VAL (Value Area Low) và VAH (Value Area High)
        if val_p > 0 and price > 0 and vah_p > 0:
            # Vùng giá rẻ, ưu tiên tìm SPRING
            if price < val_p:
                if signal == 'SPRING':
                    details.append(f"Giá dưới Vùng giá rẻ (VAL): ưu tiên SPRING, nhớ kiểm tra lực xả")
                else:
                    score -= 10
                    details.append(f"Giá dưới Vùng giá rẻ (VAL): Áp lực xả còn lớn (-10)")
            
            # Giữa VAL và VAH: Vùng tích lũy, ưu tiên tìm TEST_CUNG
            if price >= val_p and price < vah_p:
                if signal == 'TEST_CUNG':
                    details.append(f"Vùng tích lũy VAL < Price < VAH: ưu tiên TEST_CUNG")

            # Trên VAH: Vùng bứt phá, ưu tiên tìm SOS dứt khoát
            # A: Xác nhận bứt phá "Blue Sky"
            if vah_p > 0 and signal == 'SOS':
                if price > vah_p:
                    score += 15
                    details.append(f"🚀 BLUE SKY: Bứt phá vùng giá trị (VAH). Phía trên là bầu trời! (+15)")
                elif price < vah_p and (vah_p - price) / price < 0.02:
                    score -= 10
                    details.append(f"⚠️ ÁP LỰC TRẦN: Tín hiệu SOS nhưng chưa thoát được VAH (-10)")

            # B: Bộ lọc Quá mua (Mean Reversion)
            if vah_p > 0 and price > (vah_p * 1.05):
                score -= 15
                details.append(f"🚨 QUÁ XA VÙNG GIÁ TRỊ: Giá vượt VAH > 5%, rủi ro điều chỉnh về POC cao (-15)")

            # C: Hợp lưu Đẩy giá (VAH làm bệ đỡ)
            if vah_p > 0 and signal in ['TEST_CUNG', 'SPRING']:
                if abs(price - vah_p) / vah_p <= 0.015:
                    score += 10
                    details.append(f"🛡️ RE-TEST VAH: Giá kiểm định lại trần cũ thành công (+10)")

        # OMNI-MATRIX T0 PREDICTION (VŨ KHÍ TỐI THƯỢNG)
        if omni_result and "error" not in omni_result:
            verdict = omni_result.get('verdict', '')
            details_str = omni_result.get('details', '')
            short_detail = details_str.split(' | ')[0] # Chỉ lấy ý chính cho ngắn gọn
            
            if "BULLISH" in verdict and "TILT" not in verdict:
                score += 15
                details.append(f"X-Quang T0: BULLISH (+15) ({short_detail})")
            elif "BEARISH" in verdict and "TILT" not in verdict:
                score -= 20
                details.append(f"X-Quang T0: BEARISH (-20) (Bị xả trong phiên, rủi ro Bull-trap!)")
            elif "RŨ BỎ" in verdict:
                score += 5
                details.append(f"X-Quang T0: Đè gom rũ bỏ (+5)")
            elif "TILT BULL" in verdict:
                score += 5
                details.append(f"X-Quang T0: Cầu chủ động nghiêng Mua (+5)")

        return score, details

    def load_portfolio(self, p_type="paper"):
        """Đọc danh mục: p_type có thể là 'paper' (Mô phỏng) hoặc 'real' (Thực chiến)"""
        pf_path = Path(f"data/invest/{p_type}_{self.universe.lower()}.json")
        if not pf_path.exists(): return {} 
        try:
            with open(pf_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def save_portfolio(self, portfolio, p_type="paper"):
        pf_path = Path(f"data/invest/{p_type}_{self.universe.lower()}.json")
        pf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(pf_path, 'w', encoding='utf-8') as f: 
                json.dump(portfolio, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu danh mục {p_type} {self.universe}: {e}")

    def _load_watchlist(self):
        """Đọc danh sách Tầm ngắm từ File (Đã phân rổ)"""
        wl_path = Path(f"data/live/watchlist_{self.universe.lower()}.json")
        if not wl_path.exists(): return {}
        try:
            with open(wl_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def _save_watchlist(self, wl_dict):
        """Lưu danh sách Tầm ngắm xuống File (Đã phân rổ)"""
        wl_path = Path(f"data/live/watchlist_{self.universe.lower()}.json")
        wl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(wl_path, 'w', encoding='utf-8') as f: 
                json.dump(wl_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu Watchlist {self.universe}: {e}")

    def _log_trade(self, ticker, action, price, details=""):
        """Ghi nhận lịch sử giao dịch (Audit Trail) ra file CSV (Đã phân rổ)"""
        log_path = Path(f"data/invest/trade_history_{self.universe.lower()}.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("Time,Ticker,Action,Price,Details\n")
                
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{time_str},{ticker},{action},{price},\"{details}\"\n")
        except Exception as e:
            print(f"[!] Lỗi ghi log {self.universe}: {e}")

    def _get_inventory_metrics(self, ticker, lookback_days=130):
        """
        Đo lường Tồn kho Lũy kế (6 tháng) và Tỷ lệ chi phối (Phiên hiện tại)
        """
        df_price = self.price_dict.get(ticker)
        df_f = self.foreign_dict.get(ticker)
        df_p = self.prop_dict.get(ticker)

        if df_price is None or df_price.empty:
            return 0, 0, 0 # cum_inventory, dominance_pct, shadow_flow

        # Lấy khung thời gian 6 tháng gần nhất
        df_price_recent = df_price.tail(lookback_days).copy()
        
        DIVISOR = 1_000_000_000
        df_price_recent['market_val_bn'] = (df_price_recent['close'] * df_price_recent['volume']) / DIVISOR

        # Merge dữ liệu Dòng tiền
        df_merged = pd.merge(df_price_recent[['time', 'market_val_bn']], 
                             df_f[['time', 'foreign_net_value', 'foreign_buy_value', 'foreign_sell_value']] if df_f is not None else pd.DataFrame(columns=['time', 'foreign_net_value', 'foreign_buy_value', 'foreign_sell_value']), 
                             on='time', how='left')
        df_merged = pd.merge(df_merged, 
                             df_p[['time', 'prop_net_value', 'prop_buy_value', 'prop_sell_value']] if df_p is not None else pd.DataFrame(columns=['time', 'prop_net_value', 'prop_buy_value', 'prop_sell_value']), 
                             on='time', how='left').fillna(0)

        # 1. Tính Tồn kho lũy kế (Cumulative Inventory)
        df_merged['total_net_bn'] = (df_merged['foreign_net_value'] + df_merged['prop_net_value']) / DIVISOR
        cum_inventory = df_merged['total_net_bn'].sum() # Tổng gom/xả trong 6 tháng qua

        # 2. Tính Tỷ lệ chi phối phiên cuối cùng (Signal Day Dominance)
        last_row = df_merged.iloc[-1]
        mkt_val = last_row['market_val_bn']
        
        f_gross = (last_row['foreign_buy_value'] + last_row['foreign_sell_value']) / DIVISOR
        p_gross = (last_row['prop_buy_value'] + last_row['prop_sell_value']) / DIVISOR
        sm_participation = (f_gross + p_gross) / 2
        
        dominance_pct = (sm_participation / mkt_val) * 100 if mkt_val > 0 else 0
        shadow_flow = max(0, mkt_val - sm_participation)

        return cum_inventory, dominance_pct, shadow_flow

    def _run_portfolio_management_logic(self, portfolio, p_type_label, report, sm_info_dict):
        """Hàm Lõi lõi xử lý Trailing Stop và Báo bán cho 1 danh mục cụ thể"""
        if not portfolio: return {}
        
        print("\n" + "-"*65)
        print(f" 🛡️ QUẢN TRỊ DANH MỤC: {p_type_label}")
        print("-"*65)
        
        active_portfolio = {}

        for ticker, pos in portfolio.items():
            row_data = report[report['Ticker'] == ticker]

            if row_data.empty: 
                active_portfolio[ticker] = pos 
                continue
            
            row = row_data.iloc[0]
            current_price = float(row['Price'])
            signal = row['Signal']
            atr = float(row.get('ATR', current_price * 0.02))
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            
            # --- RADAR CÁ MẬP ---
            sm_result = sm_info_dict.get(ticker, {})
            sm_warnings = sm_result.get("warnings", [])
            if sm_warnings:
                print(f"   🚨 BÁO ĐỘNG ĐỎ [{ticker}] (Lãi/Lỗ: {pnl_pct:+.2f}%):")
                print(f"      Lý do: {' | '.join(sm_warnings)}!")
                print(f"      => HÀNH ĐỘNG: Smart Money đang thoát hàng. Cân nhắc hạ 50% tỷ trọng hoặc chốt lời NGAY LẬP TỨC để bảo toàn vốn!")

            # --- TRAILING STOP ---
            highest_price = float(pos.get('highest_price', pos.get('entry_price', current_price)))
            if current_price > highest_price:
                pos['highest_price'] = current_price
                new_sl = current_price - (atr * 2.5)
                old_sl = pos.get('sl_price', 0)
                if new_sl > old_sl:
                    pos['sl_price'] = new_sl
                    msg = f"Nâng SL (Trailing) từ {old_sl:,.0f} -> {new_sl:,.0f}đ"
                    print(f"   {ticker}: 🔼 {msg}")
                    # Chỉ ghi log nếu là danh mục THỰC CHIẾN
                    if "REAL" in p_type_label:
                        self._log_trade(ticker, "UPDATE_SL_REAL", current_price, msg)

            sell_reasons = []
            action = "HOLD"
            
            if current_price <= pos.get('sl_price', 0):
                sell_reasons.append(f"Chạm SL/Trailing ({pos['sl_price']:,.0f})")
                action = "SELL (STOP LOSS)"

            if self.season == "Q4_HARVEST" and pnl_pct > 5 and signal in ['UT', 'SOW']:
                sell_reasons.append(f"Q4 Thu quân: {signal}")
                action = "SELL (Q4 HARVEST)"
            elif signal in ['UT', 'SOW', 'MA_BREAK']:
                sell_reasons.append(f"Kỹ thuật Xấu: {signal}")
                action = "SELL (WYCKOFF SIGNAL)"

            if action != "HOLD":
                print(f"   {ticker}: 🔴 SELL | PnL: {pnl_pct:+.2f}% | Lý do: {', '.join(sell_reasons)}")
                if "REAL" in p_type_label:
                    self._log_trade(ticker, "SELL_REAL", current_price, f"PnL: {pnl_pct:+.2f}%. Lý do: {', '.join(sell_reasons)}")
            else:
                print(f"   {ticker}: 🟢 HOLD | PnL: {pnl_pct:+.2f}% | SL: {pos['sl_price']:,.0f}")
                active_portfolio[ticker] = pos 
            print("  " + "-" * 50)

        return active_portfolio

    def manage_investment(self, report, sm_info_dict):
        """Quản trị Song song 2 Danh mục"""
        print("\n" + "="*65)
        print(f" 💰 TỔNG QUẢN TRỊ DANH MỤC (PAPER & REAL)")
        print("="*65)
        
        # 1. Chăm sóc Danh mục Mô phỏng (Của Bot)
        paper_pf = self.load_portfolio("paper")
        updated_paper = self._run_portfolio_management_logic(paper_pf, "💻 MÔ PHỎNG (PAPER TRADING)", report, sm_info_dict)
        self.save_portfolio(updated_paper, "paper")

        # 2. Chăm sóc Danh mục Thực chiến (Của User)
        real_pf = self.load_portfolio("real")
        updated_real = self._run_portfolio_management_logic(real_pf, "🔥 THỰC CHIẾN (REAL MONEY)", report, sm_info_dict)
        self.save_portfolio(updated_real, "real")

    def scan_opportunities(self):
        # 0. Gọi bộ lọc thị trường chung TRƯỚC TIÊN
        is_uptrend = self._check_market_regime()

        temp_price_df = self.update_and_prepare_data(self.df_price, self.df_intra)
        if not temp_price_df.empty:
            df_full_price = temp_price_df
        else:
            print(f"Lỗi Live.scan_opportunities: price DataFrame không có giá trị")
            return None

        # Khởi tạo X-Ray Engine
        mf_analyzer = MarketFlowAnalyzer()

        # 2. Chạy Forecaster (Truyền đường dẫn file Parquet tạm vào)
        forecaster = WyckoffForecaster(data_input=df_full_price, output_dir=self.temp_dir, run_date=datetime.now(), verbose=False)
        report = forecaster.run_forecast()

        if report.empty: return

        # Lọc bỏ các mã nằm trong Danh sách đen
        report = report[~report['Ticker'].isin(IGNORE_TICKERS)].copy()
        print(f"[*] Đã lọc bỏ các mã lỗi, còn lại {len(report)} mã hợp lệ.")

        # =====================================================================
        # 🧭 BƯỚC 2: MÁY QUÉT ĐỘ RỘNG THỊ TRƯỜNG (MARKET BREADTH)
        # =====================================================================
        print("\n" + "="*65)
        print(" 🧭 ĐO LƯỜNG ĐỘ RỘNG THỊ TRƯỜNG (MARKET BREADTH)")
        print("="*65)
        
        # Đếm số mã có Giá nằm trên đường EMA89 (Đang trong Uptrend trung hạn)
        total_valid_stocks = len(report)
        uptrend_stocks = len(report[report['Price'] > report['EMA89']])
        
        market_breadth_pct = (uptrend_stocks / total_valid_stocks) * 100 if total_valid_stocks > 0 else 0
        
        print(f"[*] Tổng số mã sạch đạt chuẩn: {total_valid_stocks} mã")
        print(f"[*] Số mã đang giữ được Uptrend (Giá > EMA89): {uptrend_stocks} mã")
        print(f"[*] Chỉ số Market Breadth: {market_breadth_pct:.2f}%")
        
        breadth_warning = False
        
        # Kịch bản 1: Chỉ số lừa dối (Xanh vỏ đỏ lòng)
        if is_uptrend and market_breadth_pct < 30.0:
            print("   🚨 CẢNH BÁO 'XANH VỎ ĐỎ LÒNG': VN-Index trên EMA89 nhưng < 30% cổ phiếu Uptrend!")
            print("   => Bọn chúng đang kéo trụ để xả hàng. Tuyệt đối cẩn thận!")
            breadth_warning = True
            
        # Kịch bản 2: Phân kỳ tạo đáy (Cơ hội kim cương)
        elif not is_uptrend and market_breadth_pct > 50.0:
            print("   🌟 TÍN HIỆU TẠO ĐÁY: VN-Index gãy nhưng > 50% cổ phiếu đã bẻ chart đi lên trước!")
            print("   => Dòng tiền thông minh đã nhập cuộc âm thầm. Ưu tiên quét các mã này.")
            
        # Kịch bản 3: Sập hầm diện rộng (Tắm máu)
        elif market_breadth_pct < 15.0:
            print("   🩸 THỊ TRƯỜNG ĐỔ MÁU: Hơn 85% cổ phiếu đang gãy trend. Rủi ro hệ thống CỰC CAO!")
            breadth_warning = True
            
        else:
            print("   ✅ Trạng thái thị trường nội tại: ỔN ĐỊNH ĐỒNG PHA.")
            
        # NẾU CÓ CẢNH BÁO NỘI TẠI MỤC NÁT -> TỰ ĐỘNG SIẾT CHẶT VAN BƠM TIỀN
        if breadth_warning:
            self.macro_buy_threshold_adj += 10  # Tăng thêm 10 điểm độ khó cho bất kỳ lệnh mua nào
            self.macro_risk_factor *= 0.5       # Ép quy mô giải ngân giảm đi một nửa
            print(f"   => 🛡️ HỆ THỐNG PHÒNG THỦ: Đã tự động TĂNG ĐỘ KHÓ ĐIỂM MUA và CẮT 50% QUY MÔ VỐN!")
        print("="*65)
        # =====================================================================

        # Lấy thông tin Bảng giá và BCTC sẵn cho vòng lặp
        board_info_dict, fund_info_dict, sm_info_dict, mf_info_dict = {}, {}, {}, {}
        for ticker in report['Ticker']:
            board_info_dict[ticker] = self._get_market_sentiment(ticker)
            fund_info_dict[ticker] = self._get_fundamental_data(ticker)
            sm_info_dict[ticker] = self.sm_engine.analyze_ticker(ticker, board_info_dict[ticker])
            # Quét X-Quang Tồn kho Cá mập
            df_p_ticker = df_full_price[df_full_price['ticker'] == ticker]
            df_f_ticker = self.foreign_dict.get(ticker)
            df_pr_ticker = self.prop_dict.get(ticker)
            mf_info_dict[ticker] = mf_analyzer.analyze_flow(ticker, df_p_ticker, df_f_ticker, df_pr_ticker)

        # =====================================================================
        # 📡 EARLY RADAR (ĐƯA VÀO TẦM NGẮM CÁC KÈO TÂY CHỚM GOM)
        # =====================================================================
        watchlist = self._load_watchlist()
        next_watchlist = {}

        print("\n" + "="*65)
        print(" 📡 RADAR CẢNH BÁO SỚM & KIỂM ĐỊNH BẪY TAY TO")
        print("="*65)
        
        for _, row in report.iterrows():
            ticker = row['Ticker']
            signal = row['Signal']
            price = float(row['Price'])
            ema89 = float(row.get('EMA89', 0))
            sm_result = sm_info_dict.get(ticker, {}) 
            
            # 1. KIỂM CHỨNG THÀNH CÔNG: Đã nổ tín hiệu Mua
            if signal in ['SOS', 'SPRING', 'TEST_CUNG']:
                if ticker in watchlist:
                    print(f"🔥 BÙNG NỔ: {ticker} đã nổ {signal} đúng như Radar dự báo từ {watchlist[ticker]['date_added']}!")
                continue 
                
            # 2. XỬ LÝ CÁC MÃ ĐANG NẰM TRONG WATCHLIST CŨ (Phát hiện Bẫy)
            if ticker in watchlist:
                days_pending = (datetime.now() - datetime.strptime(watchlist[ticker]['date_added'], '%Y-%m-%d')).days
                
                if price < ema89:
                    print(f"🚨 PHÁT HIỆN BẪY: {ticker} gãy EMA89. Lực gom của {watchlist[ticker]['reason']} đã thất bại. Xóa khỏi tầm ngắm!")
                    continue 
                    
                if days_pending > 10:
                    print(f"⏳ HẾT HẠN GOM: {ticker} quá 10 ngày không nổ SOS. Bỏ theo dõi để tránh chôn vốn.")
                    continue

            # 3. QUÉT TÌM CƠ HỘI MỚI HOẶC DUY TRÌ THEO DÕI
            if price > ema89:
                is_radar = False
                radar_reasons = []

                # TẬN DỤNG KẾT QUẢ TỪ SMART MONEY ENGINE (O(1))
                # Không tự ý cắt DataFrame nữa, chỉ đọc 'Bản án' từ Engine
                sm_score = sm_result.get("total_sm_score", 0)
                sm_details = sm_result.get("sm_details", [])
                is_danger = sm_result.get("is_danger", False)

                # Nếu Tây/Tự doanh có dấu hiệu gom (Điểm > 0) và không có rủi ro xả tháo cống
                if sm_score > 0 and not is_danger:
                    is_radar = True
                    # Trích xuất các hành động gom hàng (Các câu có chứa dấu cộng điểm, VD: "(+5)")
                    positive_actions = [msg for msg in sm_details if "(+" in msg]
                    
                    if positive_actions:
                        radar_reasons.extend(positive_actions)
                    else:
                        radar_reasons.append(f"Smart Money tích cực (Điểm: {sm_score})")

                if is_radar:
                    reason_str = " | ".join(radar_reasons)
                    if ticker not in watchlist:
                        print(f"   🎯 TẦM NGẮM MỚI: {ticker} ({reason_str}) | Giá: {price:,.0f}đ")
                        next_watchlist[ticker] = {
                            "date_added": datetime.now().strftime('%Y-%m-%d'),
                            "price_added": price,
                            "reason": reason_str
                        }
                    else:
                        next_watchlist[ticker] = watchlist[ticker]
        
        self._save_watchlist(next_watchlist)
        
        if not next_watchlist: print("[*] Hiện tại chưa có mã nào lọt vào Tầm ngắm sớm.")
        else: print(f"[*] 🎯 Phát hiện {len(next_watchlist)} mã vào Tầm ngắm sớm.")
        print("="*65)

        # =====================================================================
        # ☠️ RADAR DANH SÁCH ĐEN (CẤM BẮT ĐÁY & QUẢN LÝ ÁN TREO)
        # =====================================================================
        print("\n" + "="*65)
        print(" ☠️  RADAR DANH SÁCH ĐEN (SMART MONEY XẢ HÀNG)")
        print("="*65)

        next_blacklist = self.blacklist_guard.load()
        
        # Gọi Vệ binh quét Ân Xá
        current_date_str = getattr(self, 'run_date', None)
        tickers_to_pardon = self.blacklist_guard.evaluate_pardons(
            current_blacklist=next_blacklist,
            board_info_dict=board_info_dict,
            sm_info_dict=sm_info_dict,
            foreign_dict=self.foreign_dict,
            current_date_str=current_date_str
        )
        
        # Thực thi gỡ mã
        for t in tickers_to_pardon:
            del next_blacklist[t]
        
        for _, row in report.iterrows():
            ticker = row['Ticker']
            sm_result = sm_info_dict.get(ticker, {}) 

            # Chỉ cần check 1 cờ duy nhất từ Engine
            if sm_result["is_danger"]:
                date_str = sm_result["last_trade_date"].strftime('%Y-%m-%d')
                reasons_str = " + ".join(sm_result["warnings"])
                
                if ticker not in next_blacklist:
                    days_penalty = (datetime.now() - datetime.strptime(date_str, '%Y-%m-%d')).days
                    if days_penalty <= 5:
                        print(f"   🚨 PHÁT HIỆN: {ticker} ({reasons_str})")
                        next_blacklist[ticker] = {"date_added": date_str, "reason": reasons_str}
                else:
                    if next_blacklist[ticker]['date_added'] != date_str:
                        print(f"   🚨 BỒI THÊM ÁN: {ticker} (Tiếp tục bị xả, reset mốc án treo)")
                        next_blacklist[ticker] = {"date_added": date_str, "reason": reasons_str}

        # Lưu lại trí nhớ
        self.blacklist_guard.save(next_blacklist)
        
        if not next_blacklist: print("[*] 🛡️ Thị trường an toàn, Blacklist trống.")
        else: print(f"[*] 🛡️ Hệ thống đang duy trì Blacklist gồm {len(next_blacklist)} mã (Đang chịu án treo).")
        print("="*65)

        # =====================================================================
        # 3. Quản trị Danh mục (Check Lệnh đang giữ)
        self.manage_investment(report, sm_info_dict)

        dynamic_threshold = self.buy_threshold + self.macro_buy_threshold_adj

        # 4. KHÓA TÍN HIỆU MUA NẾU THỊ TRƯỜNG XẤU
        print("\n" + "="*65)
        print(f"[*] NGƯỠNG MUA ĐỘNG: {dynamic_threshold}đ | CHẾ ĐỘ: {self.macro_status})")
        print("="*65)
        if not is_uptrend:
            if market_breadth_pct > 50.0:
                print("\n[!] VN-INDEX DOWNTREND NHƯNG NỘI TẠI KHỎE (PHÂN KỲ ĐÁY).")
                print("   => Cho phép mở vị thế Bắt đáy (SPRING) với quy mô vốn thăm dò.")
                dynamic_threshold += 5 # Khó tính hơn một chút
                self.macro_risk_factor *= 0.5     # Đánh 50% vốn
                # signals = report[report['Signal'].isin(['SOS', 'SPRING', 'TEST_CUNG'])].copy()
            else:
                print("\n[!] THỊ TRƯỜNG CHUNG DOWNTREND: Hủy bỏ toàn bộ tín hiệu SOS/SPRING/TEST_CUNG.")
                # signals = pd.DataFrame() 

                # --- KÍCH HOẠT LÁ CHẮN FMARKET ---
                # Chỉ gửi cảnh báo nếu hôm nay có cổ phiếu bị bán (Vừa thu tiền mặt về) 
                # hoặc danh mục đang trống trơn (Full tiền mặt)
                active_portfolio = self.load_portfolio("paper")
                if len(active_portfolio) < 2: # Đang cầm quá ít mã (Nhiều tiền mặt)
                    safe_funds = self.get_top_bond_funds(top_n=2)
                    if safe_funds:
                        print("\n🛡️ KÍCH HOẠT CHẾ ĐỘ PHÒNG THỦ:")
                        print("Thị trường chung rủi ro. Khuyến nghị luân chuyển Tiền mặt nhàn rỗi sang Quỹ Trái Phiếu để hưởng lãi suất kép an toàn:")
                        for f in safe_funds:
                            print(f"🔹 Quỹ {f['fund_name']} - Lợi nhuận kỳ vọng: {f['expected_yield']:.1f}%/năm")
        # else:
        #     signals = report[report['Signal'].isin(['SOS', 'SPRING', 'TEST_CUNG'])].copy()

        # Mặc định kiềm tra signal
        signals = report[report['Signal'].isin(['SOS', 'SPRING', 'TEST_CUNG'])].copy()

        if signals.empty:
            print("\nKhông có tín hiệu MUA hợp lệ hôm nay (Hoặc đã bị đóng băng do VNI).")
            self._cleanup()
            return
        else:
            print("\n=== QUÉT CƠ HỘI MỚI ===")

        # KIỂM DIỆN VĨ MÔ TOÀN THỊ TRƯỜNG (MARKET-WIDE STATUS)
        if hasattr(self, 'market_status'):
            if self.market_status == 'RED':
                print(f"🚨 BÁO ĐỘNG VĨ MÔ ĐỎ: Cá mập đang xả ròng toàn TT. Tuyệt đối không mua mới!")
            elif self.market_status == 'YELLOW':
                print(f"⚠️ VĨ MÔ RỦI RO (ĐÈN VÀNG): Cảnh báo Bull-Trap ảo, hạn chế FOMO")
            elif self.market_status == 'GREEN':
                print(f"🌐 VĨ MÔ THUẬN LỢI (ĐÈN XANH): Tiền vào dứt khoát toàn TT. Cờ tới tay!")

        if hasattr(self, 'market_net_active'):
            if self.market_net_active > 0:
                print(f"🔥 DÒNG TIỀN ĐANG VÀO {self.universe}: (+{self.market_net_active:.1f} Tỷ)")
            elif self.market_net_active < 0:
                print(f"🚨 DÒNG TIỀN ĐANG RÚT KHỎI {self.universe}: ({self.market_net_active:.1f} Tỷ)")

        # X-QUANG LỆNH TỪNG MÃ (MICROSTRUCTURE)
        if hasattr(self, 'intraday_dict') and ticker in self.intraday_dict:
            intra_data = self.intraday_dict[ticker]
            net_active = intra_data.get('net_active_bn', 0)
            vwap = intra_data.get('vwap', 0)
            last_price = intra_data.get('last_price', 0)
            
            if net_active > 0:
                if last_price >= vwap:
                    print(f"🌊 TIỀN VÀO CHỦ ĐỘNG: Cầu nuốt trọn cung (+{net_active:.1f} Tỷ), Giá neo trên VWAP vững chắc.")
                else:
                    print(f"⚖️ LỰC CẦU GIẰNG CO: Mua chủ động (+{net_active:.1f} Tỷ) nhưng Lái đang đè giá dưới VWAP.")
            elif net_active <= 0:
                print(f"🩸 BẪY LỆNH ẢO (SPOOFING): Breakout nhưng Bán chủ động áp đảo ({net_active:.1f} Tỷ). Lái đang kê mua xả bán!")

        # Tạo list lưu các mã đã vượt qua vòng chấm điểm và các mã đã được chấm điểm
        selected_candidates, score_candidates = [], []
        # Nạp lại blacklist để làm bộ lọc
        active_blacklist = self.blacklist_guard.load()
        # Nạp lại watchlist để làm bộ lọc
        active_watchlist = self._load_watchlist()

        for _, row in signals.iterrows():
            ticker = row['Ticker']
            price = row['Price']
            signal = row['Signal']
            atr = row.get('ATR', price * 0.02)
          
            # Lấy bản án từ X-Ray Engine
            mf_result = mf_info_dict.get(ticker, {})

            # Tính toán POC cho mã cổ phiếu
            df_p_ticker = df_full_price[df_full_price['ticker'] == ticker]
            vol_profile = self._calculate_volume_profile(df_p_ticker) 

            board_info = board_info_dict.get(ticker)
            fund_info = fund_info_dict.get(ticker)
            sm_result = sm_info_dict.get(ticker)

            # OMNI-MATRIX KIỂM DUYỆT T0
            omni_now = {}
            if hasattr(self, 'omni_matrix') and self.omni_matrix:
                past_ctx = self.omni_matrix.explain_past_movement(ticker, lookback_days=10)
                omni_now = self.omni_matrix.predict_t0_action(ticker, past_context=past_ctx)

            # KHIÊN CHỐNG ĐỔ VỎ (ĐỌC BẢN ÁN TRỰC TIẾP TỪ ENGINE O(1))
            dump_warnings = sm_result.get("warnings", [])
            
            total_score, score_details = self.calculate_confluence_score(row, board_info, fund_info, sm_result, mf_result, vol_profile, omni_now)

            score_candidates.append({
                'Ticker': ticker,
                'Price': price,
                'ATR': atr,
                'Total_Score': total_score,
                'Score_Detail': score_details,
                'Stoploss': price - (atr * (3.0 if price > 100000 else 2.5)),
                'Takeprofit_20%': price + price * 0.2
            })

            # 🚀 CỘNG THƯỞNG CHO MÃ "CÁ MẬP NHỐT HÀNG" (DTL CAO)
            dtl = mf_result.get('dtl_days', 0)
            if dtl > 5.0:
                total_score += 5
                score_details.append(f"Cá mập nhốt hàng (DTL: {dtl:.1f} ngày) (+5)")
            elif dtl > 2.0:
                total_score += 3
                score_details.append(f"An toàn thanh khoản (DTL: {dtl:.1f} ngày) (+3)")
            
            # Lọc theo ngưỡng linh hoạt của từng Quý
            # if total_score < dynamic_threshold: continue

            # Lưu lại row này vào danh sách được chọn
            row_dict = row.to_dict()
            row_dict['Total_Score'] = total_score

            # chỉ đưa vào mô phỏng giao dịch khi score >= 60 điểm
            if total_score >= 60:
                selected_candidates.append(row_dict)

            # Dynamic Risk Sizing (Quản trị vốn theo Quý)
            atr_multiplier = 3.0 if price > 100000 else 2.5
            stop_loss = price - (atr * atr_multiplier)
            # take_profit = price + price * 0.2

            # Lấy thông tin sổ lệnh để gợi ý giá
            best_bid = board_info['best_bid'] if board_info else price
            best_ask = board_info['best_ask'] if board_info else price
            ceil_price = board_info['ceil'] if board_info else 0
            floor_price = board_info['floor'] if board_info else 0

            # Xử lý Logic Hiển thị Giá
            if row['Signal'] == 'SOS':
                # Đánh Breakout (SOS) -> Cần mua nhanh, ưu tiên khớp ngay ở Ask 1
                suggested_buy = best_ask
                buy_strategy = "Mua chủ động (MP) quét lên"
            else:
                # Đánh Bắt đáy (SPRING) -> Không vội, ưu tiên kê lệnh ở Bid 1 hoặc dưới giá hiện tại
                suggested_buy = best_bid
                buy_strategy = "Kê lệnh chờ mua (Limit) giá thấp"

            # Tính vốn thực tế theo Vĩ mô
            adj_risk_per_trade = RISK_PER_TRADE * self.macro_risk_factor

            # Tính lại số lượng cổ phiếu theo mức giá gợi ý thực tế (thay vì giá đóng cửa)
            sl_distance = suggested_buy - stop_loss
            shares = 0
            if sl_distance > 0:
                shares = int(adj_risk_per_trade / sl_distance)
                shares = (shares // 100) * 100
                # Ép khối lượng theo Risk Factor của Quý hiện tại
                adj_max_position = int(MAX_POSITION_SIZE * self.risk_factor)
                shares = min(shares, adj_max_position)

            # Gọi Radar Đo lường Tồn kho & Dòng tiền Ẩn
            cum_inventory, dominance_pct, shadow_flow = self._get_inventory_metrics(ticker)


            # KIỂM TOÁN LÁI NỘI (VỚI BỘ NHỚ T+15)
            current_date = pd.to_datetime(self.run_date) if hasattr(self, 'run_date') else pd.to_datetime('today')
            shadow_memory = self._check_historical_shadow_profile(ticker, current_date, lookback_window=15)

            # Lấy giá vốn Lái từ Market Flow
            sm_vwap = mf_result.get('sm_vwap', 0)

            poc_price, val, vah = vol_profile

            print("-" * 65)
            print(f"✅ MUA | {ticker} | Điểm: {total_score}/100 | Signal: {row['Signal']}")
            print(f"   Lý do: {', '.join(score_details)}")
            print(f"   📊 X-Ray: Giá vốn Cá mập ~{sm_vwap:,.0f}đ | Sức ép xả (DTL): {mf_result.get('dtl_days',0):.1f} ngày")
            if omni_now and "error" not in omni_now:
                print(f"   ⚡ X-Ray T0: {omni_now['verdict']} | Khớp chủ động: {omni_now['net_active_bn']:+.1f} Tỷ")
            print(f"   📊 Sổ lệnh hiện tại: Bán rẻ nhất {best_ask:,.0f} | Mua cao nhất {best_bid:,.0f}")
            print(f"   📊 Price: {price:,.0f} | VAL: {val:,.0f} | VAH {vah:,.0f}")
            print(f"   🎯 HÀNH ĐỘNG: {buy_strategy} {shares:,} cp quanh {suggested_buy:,.0f}đ")
            print(f"   🛑 Cắt lỗ: {stop_loss:,.0f}đ | Biên độ ngày: {floor_price:,.0f} - {ceil_price:,.0f}")
            if (total_score < dynamic_threshold or ticker in active_blacklist or mf_result.get('divergence') == "BEARISH_TRAP" or dump_warnings):
                print("   " + "-" * 62)
                print(f"  >>> LƯU Ý:")
                if total_score < dynamic_threshold:
                    print(f"   Đánh giá: scrore / threshold => {total_score} / {dynamic_threshold}")
                if ticker in active_blacklist:
                    print(f"   🚫 TỪ CHỐI MUA {ticker}: vì nằm trong BLACKLIST.")
                # BỘ LỌC CHỐNG BẪY KÉO XẢ (Xử lý các case thao túng như HRC)
                if mf_result.get('divergence') == "BEARISH_TRAP":
                    # Chỉ check bẫy Kéo Xả nếu thực sự đo lường được Giá vốn (> 0đ)
                    if sm_vwap > 0:
                        # Nếu giá hiện tại vọt lên quá 1.5% so với giá vốn, nhưng tồn kho đang xả
                        if price > (sm_vwap * 1.015) and mf_result.get('divergence') == 'BEARISH_TRAP':
                            print(f"   🚫 TỪ CHỐI MUA {ticker}: Kéo xả ảo (Giá vốn tạo lập: {sm_vwap:,.0f}đ, Tồn kho đang bị xả!)")
                # Lệnh cấm tuyệt đối: Nến có đẹp đến mấy cũng loại ngay từ vòng gửi xe!
                if dump_warnings:
                    print(f"   🚫 TỪ CHỐI MUA {ticker}: Khối ngoại/Tự doanh vừa có nhịp phân phối rát!")
                    print(f"      Dấu vết: {' | '.join(dump_warnings)}")
            
            # Lịch sử gom hàng của lái
            if shadow_memory:
                print(f"  >>> KIỂM TOÁN LÁI NỘI:")
                status = shadow_memory['Status']
                memory_date = shadow_memory['Memory_Date']
                delay = shadow_memory['Days_Delayed']
                if "CHÍN MUỒI" in status:
                    print(f"   🌟 BỆ PHÓNG HOÀN HẢO (Ủ mưu từ {delay} ngày trước - {memory_date}): {shadow_memory['Note']}")
                elif "CHỜ ĐỢI" in status:
                    print(f"   ⏳ CẢNH BÁO NỔ SỚM (Ghi nhận lúc {memory_date}): {shadow_memory['Note']}")
                elif "NGUY HIỂM" in status:
                    print(f"   🩸 BẪY PHÂN PHỐI LÁI NỘI (Ghi nhận lúc {memory_date}): {shadow_memory['Note']}")
            
            print("-" * 65)

        # Các mã đã được chấm điểm => lưu lại để kiểm tra
        score_df = pd.DataFrame(score_candidates)
        if not score_df.empty:
            today_str = datetime.now().strftime('%d_%m_%Y')
            output_path = self.live_dir / f"forecast_{self.universe.lower()}_{today_str}.csv"
            score_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Đã phân tích xong {len(score_df)} mã. Kết quả lưu tại: {output_path}")

        # Kiểm tra xem có mã nào được chọn không
        if not selected_candidates:
            print("\nKhông có mã nào đủ tiêu chuẩn Mô Phỏng Giải Ngân hôm nay.")
            self._cleanup()
            return

        print("\n>>> Tối ưu hóa Danh mục (Quant Max-Sharpe) cho các mã đã lọc...")
        # Chuyển list thành DataFrame
        df_selected = pd.DataFrame(selected_candidates)
        
        # Gọi danh mục đang cầm hiện tại ra
        active_portfolio = self.load_portfolio("paper")
        
        try:
            engine_quant = QuantPortfolioEngine(price_dir=self.parquet_dir / 'price', output_dir=self.temp_dir)
            weights = engine_quant.optimize_portfolio(df_selected)
            if weights:
                for t, w in sorted(weights.items(), key=lambda item: item[1], reverse=True):
                    if w >= 0.01:
                        # CHỐNG MUA TRÙNG: Nếu đã có trong danh mục thì bỏ qua (Không nhồi lệnh)
                        if t in active_portfolio:
                            print(f"[*] Mã {t} đã có trong PAPER. Bỏ qua.")
                            continue

                        # Trích xuất thông tin mã được mua từ selected_candidates
                        info = next((item for item in selected_candidates if item["Ticker"] == t), None)
                        if info:
                            price = info['Price']
                            atr = info.get('ATR', price * 0.02)
                            sl_price = price - (atr * (3.0 if price > 100000 else 2.5))
                                                        
                            # Ghi nhớ vào danh mục để ngày mai Robot theo dõi
                            active_portfolio[t] = {
                                'ticker': t,
                                'entry_price': price,
                                'sl_price': sl_price,
                                'highest_price': price,
                                'date_bought': datetime.now().strftime('%Y-%m-%d')
                            }

                            # Ghi Log Mô phỏng
                            self._log_trade(t, "BUY_PAPER", price, f"Mô phỏng Phân bổ {w*100:.1f}%")
                
                # Cập nhật sổ cái
                self.save_portfolio(active_portfolio, "paper")
                
        except ImportError as e:
            print(f"[!] Bỏ qua Quant Max-Sharpe (không load được thư viện scipy): {e}")
        except Exception as e:
            print(f"[!] Lỗi Scipy: {e}")

        self._cleanup()

    def _cleanup(self):
        try: 
            shutil.rmtree(self.temp_dir)
        except: pass

class DualLogger(object):
    """Lớp hỗ trợ ghi log song song ra Terminal và File"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Đẩy dữ liệu vào file ngay lập tức (Real-time)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    # 1. Tạo thư mục chứa file log nếu chưa có
    log_dir = Path("data/run_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Định dạng tên file: run_YYYY_MM_DD_HH_MM_SS.log
    log_filename = log_dir / f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    
    # 3. Gắn bộ chia luồng (Tee) vào hệ thống
    sys.stdout = DualLogger(log_filename)
    
    print("="*65)
    print(f"[*] BẮT ĐẦU PHIÊN LÀM VIỆC. Log được lưu tại: {log_filename}")
    print("="*65)

    try:
        assistant = LiveAssistant()
        assistant.scan_opportunities()
    except Exception as e:
        print(f"\n[!!!] LỖI NGHIÊM TRỌNG TRONG QUÁ TRÌNH CHẠY BOT: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] KẾT THÚC PHIÊN LÀM VIỆC.")