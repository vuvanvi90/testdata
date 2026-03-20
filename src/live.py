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

"""
LIVE TRADING SYSTEM - QUY TRÌNH LUỒNG DỮ LIỆU (DATA FLOW)
=========================================================
1. collector.py:   Kéo dữ liệu thô từ API (vnstock_data, vci, mbk) -> Lưu thành Parquet.
2. update_and_prepare_data(): Ghép nến Intraday hôm nay vào master_price (OHLCV chuẩn).
3. Forecaster:     Tính toán EMA, ATR, VPA, Kháng cự/Hỗ trợ -> Sinh tín hiệu SOS/SPRING.
4. Confluence:     Chấm điểm hội tụ (Kỹ thuật + Vĩ mô + CANSLIM + Khối ngoại).
5. Portfolio:      Quản trị rủi ro bằng Trailing Stop (Json) & Tối ưu tỷ trọng (Scipy).
6. Notifier:       Bắn tín hiệu thực thi qua Telegram Bot.
"""

# --- CẤU HÌNH RỦI RO CƠ BẢN ---
RISK_PER_TRADE = 500000
MAX_POSITION_SIZE = 5000
BASE_SCORE_THRESHOLD = 70

# ==============================================================================
# 1. CẤU HÌNH THEMATIC 2026 (VĨ MÔ: HẠ TẦNG AI, FDI & CHÍNH SÁCH TIỀN TỆ)
# ==============================================================================

# Nhóm KCN & Bất động sản KCN (Hưởng lợi FDI & Nhu cầu Hạ tầng Data Center):
# Bổ sung các "ông trùm" quỹ đất cao su chuyển đổi và KCN phía Nam/Bắc
THEME_KCN = [
    'VGC', 'IDC', 'KBC', 'SZC', 'GVR', 'SIP', 
    'BCM', 'PHR', 'DPR', 'NTC', 'TIP', 'D2D', 'SNZ', 'LHG'
]

# Nhóm Năng lượng & Hạ tầng Điện (Cung cấp điện/Khí cho Data Center & Sản xuất):
# Bổ sung các doanh nghiệp Xây lắp điện, Thủy điện, Điện khí và Năng lượng tái tạo
THEME_ENERGY = [
    'POW', 'PC1', 'TV2', 'REE', 'GAS', 
    'NT2', 'HDG', 'GEG', 'VSH', 'QTP', 'PVD', 'PVS', 'BCG', 'GEX'
]

# Nhóm Công nghệ & Tài chính số (Trực tiếp làm AI, Viễn thông & Ngân hàng số):
# Bổ sung các mã Viễn thông, Phần mềm, Phần cứng và các Ngân hàng dẫn đầu về CASA/Tech
THEME_TECH_FIN = [
    'FPT', 'CMG', 'TCB', 'TCX',
    'ELC', 'ITD', 'VGI', 'FOX', 'CTR', 'LCG', 'MBB', 'VPB', 'VIB'
]

# Nhóm Đầu tư công & Vật liệu xây dựng (Đẩy mạnh hạ tầng 2026):
# Bổ sung nhóm Thép, Đá, Nhựa đường, Thi công (Hưởng lợi gián tiếp từ chu kỳ bơm tiền)
THEME_INFRASTRUCTURE = [
    'HPG', 'HSG', 'NKG', 'KSB', 'VLC', 'PLC', 'HHV', 'VCG', 'C4G'
]

# Tổng hợp tất cả các mã được cộng điểm Vĩ mô
THEMATIC_2026 = THEME_KCN + THEME_ENERGY + THEME_TECH_FIN + THEME_INFRASTRUCTURE

# ==============================================================================

# --- DANH SÁCH ĐEN (CÁC MÃ LỖI DỮ LIỆU / KHÔNG GIAO DỊCH) ---
IGNORE_TICKERS = ['ABR', 'ACG', 'ADP', 'AFX', 'ANT', 'ACL', 'CTR', 'DSC', 'TCI', 'TDP']

# ==============================================================================

class LiveAssistant:
    def __init__(self):
        self.temp_dir = Path('data/temp_live')
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

        # Nạp danh sách mã theo ngành
        self.df_ind = self._load_parquet_safe(self.parquet_dir / 'macro/groups_by_industries.parquet')

        # Nạp Tổng khối lượng lưu hành để đo lường Cung/Cầu
        self.out_shares_dict = {}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp Dữ liệu Doanh nghiệp (Company)...")
        df_comp = self._load_parquet_safe(self.parquet_dir / 'company/master_company.parquet')
        if not df_comp.empty and 'ticker' in df_comp.columns and 'issue_share' in df_comp.columns:
            # Tạo dictionary O(1) tra cứu siêu tốc: { 'FPT': 1200000000, ... }
            self.out_shares_dict = df_comp.set_index('ticker')['issue_share'].to_dict()

        # Nạp quỹ trái phiếu
        self.df_funds = self._load_parquet_safe(self.parquet_dir / 'macro/bond_fund.parquet')

        # Đánh giá Ma trận Thanh khoản Vĩ mô
        self.macro_buy_threshold_adj = 0
        self.macro_risk_factor = 1.0
        self.macro_status = "NEUTRAL"
        self._evaluate_macro_environment()

        # Đánh giá Sóng Ngành Tự Động
        self.dynamic_thematic_tickers = []
        self._evaluate_sector_themes()

        # 1. KHỞI TẠO SEASONALITY ENGINE (CHIẾN LƯỢC THEO QUÝ)
        self._set_seasonality_rules()

        # 2. KHỞI TẠO MACRO ENGINE (CÔNG TẮC VĨ MÔ TỰ ĐỘNG) - Bổ sung mới!
        self._analyze_macro_conditions()

        # Khởi tạo Động cơ Smart Money
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang khởi động Smart Money Engine...")
        self.sm_engine = SmartMoneyEngine(self.foreign_dict, self.prop_dict, self.out_shares_dict)

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
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang băm dữ liệu Khối Ngoại vào RAM (O(1))...")
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
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang băm dữ liệu Tự Doanh vào RAM (O(1)...")
        prop_dict = {}
        for ticker, group in df.groupby('ticker'):
            # CHỈ GIỮ 130 DÒNG CUỐI (Tương đương 6 tháng giao dịch)
            group = group.sort_values('time').tail(130)
            prop_dict[ticker] = group
            
        return prop_dict

    def _load_price_dict(self, path):
        """Đọc file Parquet Tự doanh và băm thành Dictionary O(1) chỉ giữ 20 phiên"""
        df = self._load_parquet_safe(path)
        if df.empty or 'ticker' not in df.columns:
            return {}
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang băm dữ liệu OHLCV vào RAM (O(1)...")
        price_dict = {}
        for ticker, group in df.groupby('ticker'):
            # CHỈ GIỮ 130 DÒNG CUỐI (Tương đương 6 tháng giao dịch)
            group = group.sort_values('time').tail(130)
            price_dict[ticker] = group
            
        return price_dict

    def _evaluate_macro_environment(self):
        """Phân tích Ma trận Thanh khoản từ dữ liệu Vĩ mô (CPI & Tín dụng)"""
        print("\n" + "="*65)
        print(" 🌍 ĐÁNH GIÁ MA TRẬN THANH KHOẢN (LIQUIDITY MATRIX)")
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
                    
                print(f"[*] Đã nạp thành công ICB Database: KCN ({len(TICKERS_KCN)} mã), Bán lẻ ({len(TICKERS_RETAIL)} mã), Xuất khẩu & Logistics ({len(TICKERS_EXPORT)} mã).")
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
            print("\n   [+] Đang kích hoạt Radar Dòng tiền Tuần (Smart Money Flow)...")
            
            reporter = GroupCashFlowReporter(self.df_foreign, self.df_prop, self.df_ind)
            sector_flow, flow_report_df = reporter.generate_report(timeframe='week')
            
            if sector_flow is not None and not sector_flow.empty:
                # 1. TÌM TOP 3 NGÀNH HÚT TIỀN NHẤT ĐỂ ĐÁNH LÊN (LONG)
                top_3_sectors = sector_flow.head(3)['industry'].tolist()
                
                # Quét tìm tất cả các mã cổ phiếu thuộc 3 ngành này
                hot_tickers = flow_report_df[flow_report_df['industry'].isin(top_3_sectors)]['ticker'].tolist()
                
                # Bơm các mã này vào rổ THEMATIC ĐỘNG để được ưu tiên cộng 20đ
                self.dynamic_thematic_tickers.extend(hot_tickers)
                themes_activated.append(f"Smart Money Gom Tuần: {', '.join(top_3_sectors)}")
                
                # 2. TÌM TOP 2 NGÀNH BỊ XẢ RÁT NHẤT ĐỂ ĐƯA VÀO DANH SÁCH ĐEN NGÀNH (SECTOR BLACKLIST)
                bottom_2_sectors = sector_flow.tail(2)['industry'].tolist()
                self.sector_blacklist_tickers = flow_report_df[flow_report_df['industry'].isin(bottom_2_sectors)]['ticker'].tolist()
                print(f"   [!] Đưa các mã thuộc ngành {', '.join(bottom_2_sectors)} vào Danh sách Phạt (Bị xả ròng).")

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
        """Điều chỉnh 'Tính cách' của Model theo từng giai đoạn trong năm 2026"""
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
            
        print(f"[*] Chế độ Vận hành: {self.season} | Điểm mua: {self.buy_threshold} | Tỷ trọng: {self.risk_factor*100}%")

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
        """Đọc file Parquet Vĩ mô và tự động siết/nới lỏng chiến lược"""
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
        print("\n" + "="*45)
        print(" 🌍 ĐÁNH GIÁ VĨ MÔ TOP-DOWN")
        print("="*45)
        
        if self.macro_warnings:
            self.macro_status = "PHÒNG THỦ CAO (RỦI RO VĨ MÔ)"
            self.buy_threshold = min(base_threshold, 95)
            self.risk_factor = max(base_risk, 0.0)
            
            print(f"🔴 TRẠNG THÁI: {self.macro_status}")
            for w in self.macro_warnings:
                print(f"   ⚠️ LƯU Ý: {w}")
        else:
            self.macro_status = "ỦNG HỘ (TÍCH CỰC)"
            print(f"🟢 TRẠNG THÁI: {self.macro_status}")
            print("   ✅ Tỷ giá ổn định, Giá dầu trong tầm kiểm soát.")
            
        print("-" * 45)
        print(f"🎯 Điểm Mua Kỹ thuật Tối thiểu: {self.buy_threshold}đ")
        print(f"💰 Tỷ trọng Giải ngân Tối đa:   {self.risk_factor*100:.0f}%")
        print("="*45)

    def _check_market_regime(self):
        """Kiểm tra Xu hướng thị trường chung (VN-Index) từ Parquet"""
        print("\n" + "="*45)
        print(" 🏛️ KIỂM TRA XU HƯỚNG THỊ TRƯỜNG CHUNG (VN-INDEX)")
        print("="*45)
        
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
                    print("   -> BẬT CÔNG TẮC ĐÓNG BĂNG: Dừng giải ngân mua mới!")
                    print("="*45)
                    return False # Downtrend
                else:
                    print(f"✅ UPTREND: VN-Index ({current_close:,.1f}) NẰM TRÊN EMA89 ({current_ema89:,.1f}).")
                    print("   -> Xu hướng ủng hộ: Cho phép quét tín hiệu Wyckoff.")
                    print("="*45)
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
            # print(f"[!] Lỗi trích xuất CANSLIM {ticker}: {e}")
            return None

    def _get_market_sentiment(self, ticker):
        """Trích xuất Order Book & Foreign Flow từ DataFrame Bảng giá"""
        # if self.df_board.empty or 'ticker' not in self.df_board.columns: return None
        if self.df_board.empty or 'symbol' not in self.df_board.columns: return None

        # df_ticker = self.df_board[self.df_board['ticker'] == ticker]
        df_ticker = self.df_board[self.df_board['symbol'] == ticker]
        if df_ticker.empty: return None
        
        row = df_ticker.iloc[0]
        try:
            bid_vol = float(row.get('bid_vol_1', 0)) + float(row.get('bid_vol_2', 0)) + float(row.get('bid_vol_3', 0))
            ask_vol = float(row.get('ask_vol_1', 0)) + float(row.get('ask_vol_2', 0)) + float(row.get('ask_vol_3', 0))
            net_foreign = float(row.get('foreign_buy_volume', 0)) - float(row.get('foreign_sell_volume', 0))
            
            # Lấy thông tin giá trần/sàn và bid/ask tốt nhất
            return {
                "sentiment_ratio": bid_vol / ask_vol if ask_vol > 0 else 10.0,
                "net_foreign": net_foreign,
                "best_bid": float(row.get('bid_price_1', row.get('close_price', 0))),  # Giá đang chờ mua cao nhất
                "best_ask": float(row.get('ask_price_1', row.get('close_price', 0))),  # Giá đang chào bán rẻ nhất
                "ceil": float(row.get('ceiling_price', row.get('high_price', 0))),
                "floor": float(row.get('floor_price', row.get('low_price', 0)))
            }
        except Exception as e: 
            return None

    def _calculate_poc(self, df_ticker_price, lookback_days=130):
        """
        Tính toán Point of Control (POC) - Mức giá tập trung Khối lượng lớn nhất trong 6 tháng.
        """
        if df_ticker_price is None or df_ticker_price.empty:
            return 0.0
        
        # Cắt lấy dữ liệu của 6 tháng gần nhất (khoảng 130 phiên)
        df_recent = df_ticker_price.sort_values('time').tail(lookback_days).copy()
        if df_recent.empty:
            return 0.0
            
        # Gom nhóm theo mức giá (Làm tròn 1 chữ số thập phân để gộp các bước giá sát nhau)
        df_recent['price_bin'] = df_recent['close'].round(1)
        
        # Tính tổng khối lượng cho từng mức giá
        vol_profile = df_recent.groupby('price_bin')['volume'].sum()
        
        if vol_profile.empty:
            return 0.0
            
        # Tìm mức giá có tổng khối lượng lớn nhất
        poc_price = float(vol_profile.idxmax())
        return poc_price

    def update_and_prepare_data(self, df_price, df_intra):
        """Gộp dữ liệu Lịch sử Giá và Khớp lệnh Intraday thành 1 file duy nhất cho Forecaster"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang tổng hợp Nến Provisional...")
        
        # # Load dữ liệu từ Parquet
        # df_price = self.df_price
        # df_intra = self.df_intra
        
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang chạy Bộ lọc Nhiễu (Loại bỏ hàng bo cung, thanh khoản kém)...")
        
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
        # tradable_tickers = metrics[
        #     (metrics['adv_value'] >= 3000000000) & (metrics['flat_days'] <= 2)
        # ].index
        tradable_tickers = metrics.index
        
        # Ghi đè lại df_price chỉ chứa các mã tinh khiết nhất
        df_price = df_clean[df_clean['ticker'].isin(tradable_tickers)]
        print(f"[*] Đã thanh lọc: Giữ lại {len(tradable_tickers)}/{len(ticker_counts)} mã đạt chuẩn Tổ chức lớn.")
        # ==========================================================

        # 4. LƯU RA 1 FILE DUY NHẤT ĐỂ FORECASTER ĐỌC
        temp_price_path = self.temp_dir / "master_price_live.parquet"
        df_price.to_parquet(temp_price_path, engine='pyarrow')
        return temp_price_path

    def calculate_confluence_score(self, row, board_info, fund_info, sm_result, mf_result, poc_price):
        """Hệ thống chấm điểm """
        score = 0
        details = []
        ticker = row['Ticker']
        signal = row['Signal']
        price = row['Price']

        # 1. CORE SIGNAL (Tối đa 40đ)
        if signal in ['SOS', 'SPRING']:
            score += 30; details.append("Tín hiệu Chuẩn (+30)")
            if signal == 'SOS' and row.get('VPA_Status') in ['High', 'Ultra High']:
                score += 10; details.append("Vol Uy tín (+10)")
            elif signal == 'SPRING' and row.get('VPA_Status') == 'Low':
                score += 10; details.append("Vol Cạn (+10)")

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

        # 5. SMART MONEY FOOTPRINT (Tối đa +30đ, Thấp nhất -30đ)
        # Áp dụng hệ số x1.5 từ kết quả Optimizer
        sm_score_optimized = sm_result["total_sm_score"] * 1.5
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

        # 7. KIỂM ĐỊNH SỰ ĐỒNG THUẬN NGÀY BREAKOUT (CHỐNG BULL TRAP)
        # Lấy dữ liệu Dòng tiền Tự doanh và Khối ngoại của đúng phiên giao dịch hôm nay
        df_f_today = self.foreign_dict.get(ticker)
        df_p_today = self.prop_dict.get(ticker)
        today_foreign_net = df_f_today.iloc[-1]['foreign_net_value'] if (df_f_today is not None and not df_f_today.empty) else 0
        today_prop_net = df_p_today.iloc[-1]['prop_net_value'] if (df_p_today is not None and not df_p_today.empty) else 0
        today_total_net = today_foreign_net + today_prop_net
        
        if signal in ['SOS', 'SPRING']:
            # Kịch bản Bẫy: Nổ tín hiệu đẹp nhưng tổng Tây + Tự doanh lại mang hàng ra Táng (Giá trị < 0)
            if today_total_net < 0:
                score -= 30 # Trừ điểm cực nặng để giết chết lệnh mua này
                details.append(f"☠️ BẪY BREAKOUT: Nến {signal} nhưng Tay to xả ròng hôm nay (-30)")
            # Kịch bản Vàng: Nổ tín hiệu và Tay to cũng đua lệnh Mua ròng phụ họa
            elif today_total_net > 3_000_000_000: # Lớn hơn 3 Tỷ VNĐ (Dòng tiền thực chất)
                score += 15
                details.append(f"🔥 BREAKOUT UY TÍN: Tay to đồng thuận đẩy giá mạnh (+15)")
            elif today_total_net > 0:
                score += 5
                details.append(f"✔️ Dòng tiền ngoại/tự doanh ủng hộ nhẹ (+5)")

        # # 8. Gọi Radar Đo lường Tồn kho & Dòng tiền Ẩn
        # cum_inventory, dominance_pct, shadow_flow = self._get_inventory_metrics(ticker)

        # # LUẬT 1: PHẠT NẶNG LÁI NỘI ĐẨY GIÁ (SHADOW FLOW TRAP)
        # # Nổ thanh khoản to nhưng Tay to đứng ngoài (Chi phối < 5%)
        # if signal in ['SOS', 'SPRING'] and dominance_pct < 5.0 and shadow_flow > 15.0:
        #     score -= 25
        #     details.append(f"⚠️ Rủi ro Lái Nội: Breakout nhưng Tay to không tham gia (Chi phối: {dominance_pct:.1f}% | Dòng tiền ẩn: {shadow_flow:.1f} Tỷ) [-25]")

        # # LUẬT 2: THƯỞNG ĐIỂM TỒN KHO TAY TO (INVENTORY BASE)
        # if cum_inventory > 500: # Tay to ôm ròng trên 500 tỷ trong 6 tháng qua
        #     score += 15
        #     details.append(f"📦 Bệ phóng Vàng: Tồn kho Tay to Lũy kế 6T rất lớn (+{cum_inventory:,.0f} Tỷ) [+15]")
        # elif cum_inventory < -500: # Tay to xả ròng hơn 500 tỷ trong 6 tháng
        #     score -= 15
        #     details.append(f"🩸 Cản trên dày: Tay to đã xả ròng Lũy kế 6T ({cum_inventory:,.0f} Tỷ) [-15]")
        
        # # LUẬT 3: THƯỞNG BREAKOUT CÓ SỰ BẢO KÊ CỦA TAY TO
        # if dominance_pct > 20.0:
        #     score += 10
        #     details.append(f"🎯 Lực đẩy Uy tín: Tay to tham gia chi phối mạnh phiên nổ Vol ({dominance_pct:.1f}%) [+10]")

        return score, details

    def load_investment(self):
        """Đọc danh mục dưới dạng Dictionary { 'TICKER': {thông tin} }"""
        pf_path = Path("data/invest/current.json")
        if not pf_path.exists(): return {} # Sửa từ [] thành {}
        try:
            with open(pf_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def save_investment(self, portfolio):
        pf_path = Path("data/invest/current.json")
        pf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(pf_path, 'w', encoding='utf-8') as f: 
                json.dump(portfolio, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu danh mục: {e}")

    def _load_watchlist(self):
        """Đọc danh sách Tầm ngắm từ File"""
        wl_path = Path("data/live/watchlist.json")
        if not wl_path.exists(): return {}
        try:
            with open(wl_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def _save_watchlist(self, wl_dict):
        """Lưu danh sách Tầm ngắm xuống File"""
        wl_path = Path("data/live/watchlist.json")
        wl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(wl_path, 'w', encoding='utf-8') as f: 
                json.dump(wl_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu Watchlist: {e}")

    def _load_blacklist(self):
        """Đọc danh sách Đen từ File"""
        bl_path = Path("data/live/blacklist.json")
        if not bl_path.exists(): return {}
        try:
            with open(bl_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def _save_blacklist(self, bl_dict):
        """Lưu danh sách Đen xuống File"""
        bl_path = Path("data/live/blacklist.json")
        bl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(bl_path, 'w', encoding='utf-8') as f: 
                json.dump(bl_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu Blacklist: {e}")

    def _log_trade(self, ticker, action, price, details=""):
        """Ghi nhận lịch sử giao dịch (Audit Trail) ra file CSV"""
        log_path = Path("data/invest/trade_history.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Nếu file chưa tồn tại, tạo file và ghi Header
        if not log_path.exists():
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("Time,Ticker,Action,Price,Details\n")
                
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                # Dùng dấu nháy kép cho details để tránh lỗi dấu phẩy trong nội dung
                f.write(f"{time_str},{ticker},{action},{price},\"{details}\"\n")
        except Exception as e:
            print(f"[!] Lỗi ghi log: {e}")

    def _check_smart_money_distribution(self, ticker):
        """Quét xem Khối ngoại hoặc Tự doanh có đang âm thầm xả hàng không"""
        warnings = []
        current_date = pd.Timestamp.now().normalize()
        MAX_DELAY_DAYS = 15
        
        # Lấy dữ liệu 10 phiên gần nhất để tính Trung bình Lực Gom
        def check_flow(df, net_col, actor_name):
            if df is not None and not df.empty:
                last_date = pd.to_datetime(df['time'].max())
                if (current_date - last_date).days <= MAX_DELAY_DAYS:
                    df_recent = df.tail(10)
                    df_3d = df.tail(3)
                    
                    # Quy tắc 1 (Cũ): Xả ròng 3 phiên liên tiếp (Xả rỉ rả)
                    if len(df_3d[df_3d[net_col] < 0]) == 3:
                        warnings.append(f"{actor_name} xả ròng 3 phiên liên tiếp")
                        
                    # 🚀 QUY TẮC 2 (MỚI): XẢ ĐỘT BIẾN (SUDDEN DUMP) TRONG PHIÊN GẦN NHẤT
                    today_val = df.iloc[-1][net_col]
                    if today_val < 0: # Nếu hôm nay là phiên Xả
                        # Tính trung bình các phiên Gom trước đó để so sánh
                        avg_buy = df_recent[df_recent[net_col] > 0][net_col].mean()
                        
                        # Nếu giá trị xả hôm nay LỚN HƠN 1.5 lần mức độ gom trung bình trước đó
                        # Tức là: Gom thì nhỏ giọt, mà xả thì xả 1 cục to đùng!
                        if pd.notna(avg_buy) and avg_buy > 0 and abs(today_val) > (avg_buy * 1.5):
                            warnings.append(f"CÚ XẢ ĐỘT BIẾN từ {actor_name} (Xả 1 phiên lấp luôn lực gom)")

        # Chạy kiểm tra cho cả Tây và Tự doanh
        check_flow(self.foreign_dict.get(ticker), 'foreign_net_value', 'Khối ngoại')
        check_flow(self.prop_dict.get(ticker), 'prop_net_value', 'Tự doanh')
                
        return warnings

    def _get_inventory_metrics(self, ticker, lookback_days=120):
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

    def manage_investment(self, report, board_info_dict):
        """Quản trị danh mục: Đã tích hợp nhạy cảm theo Quý"""
        portfolio = self.load_investment()
        if not portfolio: return {} # Trả về Dict rỗng

        print("\n=== QUẢN TRỊ DANH MỤC (BÁO BÁN & TRAILING STOP) ===")
        active_portfolio = {} # Danh sách mã tiếp tục nắm giữ

        for ticker, pos in portfolio.items():
            row_data = report[report['Ticker'] == ticker]

            if row_data.empty: 
                active_portfolio[ticker] = pos # Giữ lại nếu không có data hôm nay
                continue
            
            row = row_data.iloc[0]
            current_price = float(row['Price'])
            signal = row['Signal']
            atr = float(row.get('ATR', current_price * 0.02))
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            
            # -----------------------------------------------------------------
            # RADAR PHÁT HIỆN CÁ MẬP THÁO CHẠY
            # -----------------------------------------------------------------
            sm_warnings = self._check_smart_money_distribution(ticker)
            if sm_warnings:
                print(f"   🚨 BÁO ĐỘNG ĐỎ [{ticker}] (Lãi/Lỗ: {pnl_pct:+.2f}%):")
                print(f"      Lý do: {' | '.join(sm_warnings)}!")
                print(f"      => HÀNH ĐỘNG: Smart Money đang thoát hàng. Cân nhắc hạ 50% tỷ trọng hoặc chốt lời NGAY LẬP TỨC để bảo toàn vốn!")

            # --- CẬP NHẬT TRAILING STOP ---
            highest_price = float(pos.get('highest_price', pos.get('entry_price', current_price)))
            if current_price > highest_price:
                pos['highest_price'] = current_price
                new_sl = current_price - (atr * 2.5)
                # Kéo SL lên nếu giá tạo đỉnh mới và ghi Lịch sử
                old_sl = pos.get('sl_price', 0)
                if new_sl > old_sl:
                    pos['sl_price'] = new_sl
                    msg = f"Nâng chặn lãi (Trailing Stop) từ {old_sl:,.0f} lên {new_sl:,.0f}đ"
                    print(f"🔼 {ticker}: {msg}")
                    # Ghi Log sự kiện Nâng SL
                    self._log_trade(ticker, "UPDATE_SL", current_price, msg)

            sell_reasons = []
            action = "HOLD"
            
            # Kiểm tra Cắt lỗ / Trailing Stop
            if current_price <= pos.get('sl_price', 0):
                sell_reasons.append(f"Chạm Cắt lỗ/Trailing Stop ({pos['sl_price']:,.0f})")
                action = "SELL (STOP LOSS)"

            # THEMATIC EXIT RULES: Giao dịch theo Quý
            if self.season == "Q4_HARVEST":
                # Q4: Chốt lãi sớm nếu tiệm cận target hoặc có dấu hiệu suy yếu nhẹ
                if pnl_pct > 5 and signal in ['UT', 'SOW']:
                    sell_reasons.append(f"Q4 Thu quân sớm: Tín hiệu {signal}")
                    action = "SELL (Q4 HARVEST)"
            elif signal in ['UT', 'SOW', 'MA_BREAK']:
                sell_reasons.append(f"Kỹ thuật Xấu: {signal}")
                action = "SELL (WYCKOFF SIGNAL)"

            if action != "HOLD":
                msg = f"🔴 <b>SELL: {ticker}</b> | Lãi/Lỗ: {pnl_pct:+.2f}% | Lý do: {', '.join(sell_reasons)}"
                print(msg.replace("<b>", "").replace("</b>", ""))
                print("-" * 50)
                # Ghi Log sự kiện BÁN
                self._log_trade(ticker, "SELL", current_price, f"PnL: {pnl_pct:+.2f}%. Lý do: {', '.join(sell_reasons)}")
                # Lệnh bán thì KHÔNG đưa vào active_portfolio nữa
            else:
                print(f"🟢 HOLD: {ticker} | PnL: {pnl_pct:+.2f}% | SL Hiện tại: {pos['sl_price']:,.0f}")
                print("-" * 50)
                active_portfolio[ticker] = pos # Tiếp tục giữ

        # Lưu lại trạng thái Trailing Stop mới
        self.save_investment(active_portfolio)
        return active_portfolio

    def scan_opportunities(self):
        # 0. Gọi bộ lọc thị trường chung TRƯỚC TIÊN
        is_uptrend = self._check_market_regime()

        # 1. Chuẩn bị file dữ liệu tạm thời (temp_master_price.parquet)
        temp_price_path = self.update_and_prepare_data(self.df_price, self.df_intra)
        if not temp_price_path: return

        # Nạp toàn bộ dữ liệu giá vào RAM cho X-Ray Engine
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp Lịch sử giá vào RAM cho X-Ray Engine...")
        df_full_price = pd.read_parquet(temp_price_path)
        
        # Khởi tạo X-Ray Engine
        mf_analyzer = MarketFlowAnalyzer()

        # 2. Chạy Forecaster (Truyền đường dẫn file Parquet tạm vào)
        print("\n>>> Đang chạy phân tích Wyckoff & Vĩ mô...")
        forecaster = WyckoffForecaster(price_dir=temp_price_path, output_dir=self.temp_dir, run_date=datetime.now())
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

        current_date = pd.Timestamp.now().normalize()
        MAX_DELAY_DAYS = 15
        
        for _, row in report.iterrows():
            ticker = row['Ticker']
            signal = row['Signal']
            price = float(row['Price'])
            ema89 = float(row.get('EMA89', 0))
            sm_result = sm_info_dict.get(ticker, {}) # Lấy kết quả đã phân tích
            
            # 1. KIỂM CHỨNG THÀNH CÔNG: Đã nổ tín hiệu Mua
            if signal in ['SOS', 'SPRING']:
                if ticker in watchlist:
                    print(f"🔥 BÙNG NỔ: {ticker} đã nổ {signal} đúng như Radar dự báo từ {watchlist[ticker]['date_added']}!")
                continue 
                
            # 2. XỬ LÝ CÁC MÃ ĐANG NẰM TRONG WATCHLIST CŨ (Phát hiện Bẫy)
            if ticker in watchlist:
                days_pending = (datetime.now() - datetime.strptime(watchlist[ticker]['date_added'], '%Y-%m-%d')).days
                
                # PHÁT HIỆN BẪY (BULL TRAP): Cổ phiếu gãy Trend
                if price < ema89:
                    print(f"🚨 PHÁT HIỆN BẪY: {ticker} gãy EMA89. Lực gom của {watchlist[ticker]['reason']} đã thất bại. Xóa khỏi tầm ngắm!")
                    continue # Không đưa vào next_watchlist
                    
                # PHÁT HIỆN KẸP HÀNG: Ngâm quá 10 phiên không kéo
                if days_pending > 10:
                    print(f"⏳ HẾT HẠN GOM: {ticker} quá 10 ngày không nổ SOS. Bỏ theo dõi để tránh chôn vốn.")
                    continue

            # 3. QUÉT TÌM CƠ HỘI MỚI HOẶC DUY TRÌ THEO DÕI
            if price > ema89:
                is_radar = False
                radar_reasons = []

                df_f = self.foreign_dict.get(ticker)
                df_p = self.prop_dict.get(ticker)

                # 3A. Tây lông (Dùng cờ valid_f từ Engine để bỏ qua mã lãng quên)
                if sm_result["valid_f"] and df_f is not None:
                    df_3d = df_f.tail(3)
                    net_buy_3d = len(df_3d[df_3d['foreign_net_volume'] > 0])
                    if df_f.tail(20)['foreign_net_volume'].sum() > 0 and net_buy_3d >= 1:
                        is_radar = True
                        radar_reasons.append(f"Tây gom {net_buy_3d} phiên" if net_buy_3d > 1 else "Tây chớm gom")

                # 3B. Tự doanh (Dùng cờ valid_p từ Engine)
                if sm_result["valid_p"] and df_p is not None:
                    df_3d = df_p.tail(3)
                    net_buy_3d = len(df_3d[df_3d['prop_net_volume'] > 0])
                    if net_buy_3d >= 1:
                        is_radar = True
                        radar_reasons.append(f"Tự doanh gom {net_buy_3d} phiên" if net_buy_3d > 1 else "Tự doanh chớm gom")

                if is_radar:
                    # Ghi chú rõ lý do vào Tên mã để in ra màn hình
                    if ticker not in watchlist:
                        print(f"   🎯 TẦM NGẮM MỚI: {ticker} ({' + '.join(radar_reasons)}) | Giá: {price:,.0f}đ")
                        next_watchlist[ticker] = {
                            "date_added": datetime.now().strftime('%Y-%m-%d'),
                            "price_added": price,
                            "reason": " + ".join(radar_reasons)
                        }
                    else:
                        # Vẫn thỏa mãn tiêu chí, tiếp tục giữ trong Watchlist
                        next_watchlist[ticker] = watchlist[ticker]
        
        self._save_watchlist(next_watchlist)
        
        if not next_watchlist: print("[*] Hiện tại chưa có mã nào lọt vào Tầm ngắm sớm.")
        else: print(f"[*] Phát hiện {len(next_watchlist)} mã vào Tầm ngắm sớm.")
        print("="*65)

        # =====================================================================
        # ☠️ RADAR DANH SÁCH ĐEN (CẤM BẮT ĐÁY & QUẢN LÝ ÁN TREO)
        # =====================================================================
        print("\n" + "="*65)
        print(" ☠️  RADAR DANH SÁCH ĐEN (SMART MONEY XẢ HÀNG)")
        print("="*65)
        
        current_blacklist = self._load_blacklist()
        next_blacklist = current_blacklist.copy()
        
        for _, row in report.iterrows():
            ticker = row['Ticker']
            sm_result = sm_info_dict[ticker]

            # Chỉ cần check 1 cờ duy nhất từ Engine
            if sm_result["is_danger"]:
                date_str = sm_result["last_trade_date"].strftime('%Y-%m-%d')
                reasons_str = " + ".join(sm_result["warnings"])
                
                if ticker not in current_blacklist:
                    days_penalty = (datetime.now() - datetime.strptime(date_str, '%Y-%m-%d')).days
                    if days_penalty <= 5:
                        print(f"   🚨 PHÁT HIỆN: {ticker} ({reasons_str})")
                        next_blacklist[ticker] = {"date_added": date_str, "reason": reasons_str}
                else:
                    if current_blacklist[ticker]['date_added'] != date_str:
                        print(f"   🚨 BỒI THÊM ÁN: {ticker} (Tiếp tục bị xả, reset mốc án treo)")
                        next_blacklist[ticker] = {"date_added": date_str, "reason": reasons_str}

        # Rà soát & Ân xá
        tickers_to_remove = []
        for t, info in next_blacklist.items():
            days_penalty = (datetime.now() - datetime.strptime(info['date_added'], '%Y-%m-%d')).days
            if days_penalty > 5:
                tickers_to_remove.append(t)

        for t in tickers_to_remove:
            del next_blacklist[t]
            print(f"   🔓 Ân xá: {t} đã qua 5 ngày không bị xả, gỡ khỏi Blacklist.")

        # Lưu lại trí nhớ
        self._save_blacklist(next_blacklist)
        
        if not next_blacklist: print("[*] 🛡️ Thị trường an toàn, Blacklist trống.")
        else: print(f"[*] 🛡️ Hệ thống đang duy trì Blacklist gồm {len(next_blacklist)} mã (Đang chịu án treo).")
        print("="*65)

        # =====================================================================
        # 3. Quản trị Danh mục (Check Lệnh đang giữ)
        self.manage_investment(report, board_info_dict)

        dynamic_threshold = self.buy_threshold + self.macro_buy_threshold_adj

        # 4. KHÓA TÍN HIỆU MUA NẾU THỊ TRƯỜNG XẤU
        print("\n" + "="*65)
        print(f"[*] Ngưỡng mua động: {dynamic_threshold}đ | Chế độ: {self.macro_status}) ===")
        print("="*65)
        if not is_uptrend:
            if market_breadth_pct > 50.0:
                print("\n[!] VN-INDEX DOWNTREND NHƯNG NỘI TẠI KHỎE (PHÂN KỲ ĐÁY).")
                print("   => Cho phép mở vị thế Bắt đáy (SPRING) với quy mô vốn thăm dò.")
                dynamic_threshold += 5 # Khó tính hơn một chút
                self.macro_risk_factor *= 0.5     # Đánh 50% vốn
                signals = report[report['Signal'].isin(['SOS', 'SPRING'])].copy()
            else:
                print("\n[!] THỊ TRƯỜNG CHUNG DOWNTREND: Hủy bỏ toàn bộ tín hiệu SOS/SPRING.")
                signals = pd.DataFrame() 

                # --- KÍCH HOẠT LÁ CHẮN FMARKET ---
                # Chỉ gửi cảnh báo nếu hôm nay có cổ phiếu bị bán (Vừa thu tiền mặt về) 
                # hoặc danh mục đang trống trơn (Full tiền mặt)
                active_portfolio = self.load_investment()
                if len(active_portfolio) < 2: # Đang cầm quá ít mã (Nhiều tiền mặt)
                    safe_funds = self.get_top_bond_funds(top_n=2)
                    if safe_funds:
                        print("\n🛡️ KÍCH HOẠT CHẾ ĐỘ PHÒNG THỦ:")
                        print("Thị trường chung rủi ro. Khuyến nghị luân chuyển Tiền mặt nhàn rỗi sang Quỹ Trái Phiếu để hưởng lãi suất kép an toàn:")
                        for f in safe_funds:
                            print(f"🔹 Quỹ {f['fund_name']} - Lợi nhuận kỳ vọng: {f['expected_yield']:.1f}%/năm")
        else:
            signals = report[report['Signal'].isin(['SOS', 'SPRING'])].copy()

        # Test
        signals = report[report['Signal'].isin(['SOS', 'SPRING'])].copy()

        if signals.empty:
            print("\nKhông có tín hiệu MUA hợp lệ hôm nay (Hoặc đã bị đóng băng do VNI).")
            self._cleanup()
            return
        else:
            print("\n=== QUÉT CƠ HỘI MỚI ===")

        # Tạo list lưu các mã đã vượt qua vòng chấm điểm và các mã đã được chấm điểm
        selected_candidates, score_candidates = [], []
        # Nạp lại blacklist để làm bộ lọc
        active_blacklist = self._load_blacklist()
        # Nạp lại watchlist để làm bộ lọc
        active_watchlist = self._load_watchlist()

        for _, row in signals.iterrows():
            ticker = row['Ticker']
            price = row['Price']
            signal = row['Signal']
            atr = row.get('ATR', price * 0.02)

            # if ticker in active_blacklist:
            #     print(f"   🚫 Bỏ qua mã {ticker} vì nằm trong blacklist.")
            #     continue # Nhảy cóc sang mã khác luôn, không thèm tính điểm nữa
            
            # Lấy bản án từ X-Ray Engine
            mf_result = mf_info_dict.get(ticker, {})

            # Tính toán POC cho mã cổ phiếu
            df_p_ticker = df_full_price[df_full_price['ticker'] == ticker]
            poc_price = self._calculate_poc(df_p_ticker)

            # # BỘ LỌC CHỐNG BẪY KÉO XẢ (Xử lý các case thao túng như HRC)
            # if mf_result.get('divergence') == "BEARISH_TRAP":
            #     print(f"   🚫 TỪ CHỐI MUA {ticker}: Kéo xả ảo (Giá vốn tạo lập: {mf_result.get('sm_vwap', 0):,.0f}đ, Tồn kho đang bị xả!)")
            #     continue # Nhảy qua mã khác, cấm giải ngân!

            # KHIÊN CHỐNG ĐỔ VỎ (BẪY T+1 / XẢ ĐỘT BIẾN)
            dump_warnings = self._check_smart_money_distribution(ticker)
            # if dump_warnings:
            #     print(f"   🚫 TỪ CHỐI MUA {ticker}: Khối ngoại/Tự doanh vừa có nhịp phân phối rát!")
            #     print(f"      Dấu vết: {' | '.join(dump_warnings)}")
            #     continue # Lệnh cấm tuyệt đối: Nến có đẹp đến mấy cũng loại ngay từ vòng gửi xe!

            board_info = board_info_dict.get(ticker)
            fund_info = fund_info_dict.get(ticker)
            sm_result = sm_info_dict.get(ticker)
            
            total_score, score_details = self.calculate_confluence_score(row, board_info, fund_info, sm_result, mf_result, poc_price)

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
                # shares = int(RISK_PER_TRADE / sl_distance)
                shares = int(adj_risk_per_trade / sl_distance)
                shares = (shares // 100) * 100
                # Ép khối lượng theo Risk Factor của Quý hiện tại
                adj_max_position = int(MAX_POSITION_SIZE * self.risk_factor)
                shares = min(shares, adj_max_position)

            # Gọi Radar Đo lường Tồn kho & Dòng tiền Ẩn
            cum_inventory, dominance_pct, shadow_flow = self._get_inventory_metrics(ticker)

            print("-" * 65)
            print(f"✅ MUA | {ticker} | Điểm: {total_score}/100")
            print(f"   Lý do: {', '.join(score_details)}")
            print(f"   📊 X-Ray: Giá vốn Cá mập ~{mf_result.get('sm_vwap',0):,.0f}đ | Sức ép xả (DTL): {mf_result.get('dtl_days',0):.1f} ngày")
            print(f"   📊 Sổ lệnh hiện tại: Bán rẻ nhất {best_ask:,.0f} | Mua cao nhất {best_bid:,.0f}")
            print(f"   🎯 HÀNH ĐỘNG: {buy_strategy} {shares:,} cp quanh {suggested_buy:,.0f}đ")
            print(f"   🛑 Cắt lỗ: {stop_loss:,.0f}đ | Biên độ ngày: {floor_price:,.0f} - {ceil_price:,.0f}")
            if (total_score < dynamic_threshold or ticker in active_blacklist or mf_result.get('divergence') == "BEARISH_TRAP" or dump_warnings):
                print("   " + "-" * 62)
                print(f"  >>> LƯU Ý:")
                if total_score < dynamic_threshold:
                    print(f"   Đánh giá: scrore / threshold => {total_score} / {dynamic_threshold}")
                if ticker in active_blacklist:
                    print(f"   🚫 BỎ QUA MÃ {ticker}: vì nằm trong BLACKLIST.")
                # BỘ LỌC CHỐNG BẪY KÉO XẢ (Xử lý các case thao túng như HRC)
                if mf_result.get('divergence') == "BEARISH_TRAP":
                    print(f"   🚫 TỪ CHỐI MUA {ticker}: Kéo xả ảo (Giá vốn tạo lập: {mf_result.get('sm_vwap', 0):,.0f}đ, Tồn kho đang bị xả!)")
                # Lệnh cấm tuyệt đối: Nến có đẹp đến mấy cũng loại ngay từ vòng gửi xe!
                if dump_warnings:
                    print(f"   🚫 TỪ CHỐI MUA {ticker}: Khối ngoại/Tự doanh vừa có nhịp phân phối rát!")
                    print(f"      Dấu vết: {' | '.join(dump_warnings)}")
            print("-" * 65)

        # Các mã đã được chấm điểm => lưu lại để kiểm tra
        score_df = pd.DataFrame(score_candidates)
        if not score_df.empty:
            today_str = datetime.now().strftime('%d_%m_%Y')
            output_path = self.live_dir / f"forecast_{today_str}.csv"
            score_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Đã phân tích xong {len(score_df)} mã. Kết quả lưu tại: {output_path}")

        # Kiểm tra xem có mã nào được chọn không
        if not selected_candidates:
            print("\nKhông có mã nào đủ tiêu chuẩn giải ngân hôm nay.")
            self._cleanup()
            return

        print("\n>>> Tối ưu hóa Danh mục (Quant Max-Sharpe) cho các mã đã lọc...")
        # Chuyển list thành DataFrame
        df_selected = pd.DataFrame(selected_candidates)
        
        # Gọi danh mục đang cầm hiện tại ra
        active_portfolio = self.load_investment()
        has_new_buy = False
        
        try:
            from src.portfolio_quant import QuantPortfolioEngine
            engine_quant = QuantPortfolioEngine(price_dir=self.parquet_dir / 'price', output_dir=self.temp_dir)
            weights = engine_quant.optimize_portfolio(df_selected)
            if weights:
                for t, w in sorted(weights.items(), key=lambda item: item[1], reverse=True):
                    if w >= 0.01:
                        # CHỐNG MUA TRÙNG: Nếu đã có trong danh mục thì bỏ qua (Không nhồi lệnh)
                        if t in active_portfolio:
                            print(f"[*] Mã {t} đã có trong danh mục. Bỏ qua mua mới.")
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

                            # Ghi Log sự kiện MUA
                            self._log_trade(t, "BUY", price, f"Phân bổ {w*100:.1f}%")
                
                # Cập nhật sổ cái
                self.save_investment(active_portfolio)
                
        except ImportError as e:
            print(f"[!] Bỏ qua Quant Max-Sharpe (không load được thư viện scipy): {e}")
        except Exception as e:
            print(f"[!] Lỗi Scipy: {e}")

        self._cleanup()

    def _cleanup(self):
        try: 
            shutil.rmtree(self.temp_dir)
            # print("OK")
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