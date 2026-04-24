import os
import time
import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import concurrent.futures
import threading

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*DataFrameGroupBy.apply.*")

from vnstock_data import Listing, Company, Quote, Trading, Finance, Macro, CommodityPrice, Fund

class VNStockDataPipeline:
    def __init__(self, source='VCI', get_com=False, get_price=False, get_intra=False, get_board=False, get_fin=False, get_group=False, get_macro=False, get_foreign=False, get_prop=False, get_fund=False, get_share_group=False, get_index=False, get_pt=False):
        self.source = source.lower() if source else 'vci'
        self.get_com = get_com
        self.get_price = get_price
        self.get_intra = get_intra
        self.get_board = get_board
        self.get_fin = get_fin
        self.get_group = get_group if get_group else 'VN30'
        self.get_macro = get_macro
        self.get_foreign = get_foreign
        self.get_prop = get_prop
        self.get_fund = get_fund
        self.get_share_group = get_share_group
        self.get_index = get_index
        self.get_pt = get_pt

        # Cấu trúc thư mục lưu Parquet
        self.folders = {
            'parquet_base': Path('data/parquet'),
            'company': Path('data/parquet/company'),
            'price': Path('data/parquet/price'),
            'intraday': Path('data/parquet/intraday'),
            'current': Path('data/parquet/board'),
            'financial': Path('data/parquet/financial'),
            'macro': Path('data/parquet/macro')
        }
        for f in self.folders.values():
            f.mkdir(parents=True, exist_ok=True)

        self.listing = Listing(source=self.source)
        # self.macro = Macro(source='mbk')
        self.macro = Macro()
        # Mặc định KBS cho Trading
        self.trading = Trading(symbol='VCB', source='kbs')
        # NẠP SẴN DỮ LIỆU CŨ VÀO RAM ĐỂ MERGE
        self.old_prices = self._load_master_data('price', 'master_price.parquet')
        self.old_intraday = self._load_master_data('intraday', 'master_intraday.parquet')
        self.old_fins = self._load_master_data('financial', 'master_financial.parquet')
        self.old_coms = self._load_master_data('company', 'master_company.parquet')
        # BỘ ĐẾM RATE LIMITER THÔNG MINH
        self.request_timestamps = deque()
        self.rate_lock = threading.Lock()
        self.total_requests_sent = 0
        self.MAX_RPM = 200  # Ngưỡng an toàn tối đa (500 thay vì 600 để chừa hao phí cho thư viện)

    def _load_master_data(self, folder_key, file_name):
        """Đọc file Parquet cũ và chia thành Dictionary theo Ticker để truy xuất O(1)"""
        path = self.folders[folder_key] / file_name
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not df.empty and 'ticker' in df.columns:
                    # Chuyển Dataframe lớn thành Dict: { 'ACB': df_acb, 'HPG': df_hpg }
                    return {ticker: group for ticker, group in df.groupby('ticker')}
            except Exception as e:
                print(f"[!] Lỗi đọc file cũ {file_name}: {e}")
        return {}

    def _wait_for_token(self):
        """Thuật toán Sliding Window kiểm soát và đếm Request API"""
        with self.rate_lock:
            now = time.time()
            
            # 1. Quét và dọn dẹp các request đã cũ hơn 60 giây
            while self.request_timestamps and now - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()

            # 2. CHỐNG SPAM PHÚT (Giới hạn tổng 500 reqs / 60s)
            if len(self.request_timestamps) >= self.MAX_RPM:
                # Tính toán thời gian cần ngủ để đợi slot đầu tiên trong cửa sổ 60s được giải phóng
                sleep_duration = 60.0 - (now - self.request_timestamps[0])
                if sleep_duration > 0:
                    # print(f"\n[🛑] ĐẠT GIỚI HẠN AN TOÀN: Đã gửi {self.MAX_RPM} reqs trong 1 phút qua!")
                    # print(f"     -> Tạm dừng hệ thống {sleep_duration:.1f} giây để làm mát API...")
                    time.sleep(sleep_duration)
                    
                    # Làm mới lại thời gian sau khi ngủ
                    now = time.time()
                    while self.request_timestamps and now - self.request_timestamps[0] > 60:
                        self.request_timestamps.popleft()

            # 3. CHỐNG SPAM GIÂY (Tránh việc bắn 1 lúc 100 reqs trong 1 giây)
            if self.request_timestamps:
                time_since_last = now - self.request_timestamps[-1]
                if time_since_last < 0.1:  # Giới hạn tối đa ~10 reqs / giây
                    time.sleep(0.1 - time_since_last)
                    now = time.time()

            # 4. Cấp phép và Ghi nhận Request mới
            self.request_timestamps.append(now)
            self.total_requests_sent += 1
            current_rpm = len(self.request_timestamps)
            
            return self.total_requests_sent, current_rpm

    def _validate_schema(self, df, required_cols, item_name=""):
        """
        Trạm kiểm lâm: Kiểm tra, làm sạch cột trùng lặp và xác thực Schema.
        Trả về DataFrame (đã làm sạch) nếu hợp lệ. Trả về None nếu là RÁC.
        """
        if df is None or df.empty:
            return None

        dup_cols = df.columns[df.columns.duplicated()].unique()
        if len(dup_cols) > 0:
            print(f"   [!] CẢNH BÁO [{item_name}]: Lỗi trùng cột {list(dup_cols)}. Đang tự động gọt bỏ...")
            # Lọc giữ lại cột xuất hiện đầu tiên, vứt bỏ các cột duplicate phía sau
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Ghi log rõ ràng để anh biết API đang bị lỗi ở cột nào
            print(f"   [!] LỖI DỮ LIỆU [{item_name}]: Thiếu cột {missing_cols}. Đã tự động chặn!")
            return None
            
        return df

    # ==========================================
    # CÁC HÀM WORKER CHO ĐA LUỒNG (MULTI-THREADING)
    # ==========================================
    def _run_throttled_tasks(self, executor, worker_func, tickers):
        """
        Hàm điều phối đa luồng có Van Điều Áp (Throttling).
        Ép hệ thống chỉ submit 1 task mới sau mỗi 'delay' giây (0.12s ~ 500 req/min).
        """
        futures = {}
        results = []
        
        # Bơm việc từ từ thay vì ồ ạt
        for t in tickers:
            # Xin Quota và LẤY SỐ LIỆU ĐẾM (Sẽ tự động bị Block ở đây nếu quá tải)
            total_sent, current_rpm = self._wait_for_token()
            # IN LOG THEO DÕI: Cứ mỗi 20 mã sẽ báo cáo tốc độ 1 lần
            if total_sent % 20 == 0 or total_sent == 1:
                print(f"   [📡 API Tracker] Tổng reqs đã gọi: {total_sent} | Tốc độ hiện tại: {current_rpm}/{self.MAX_RPM} reqs/phút")

            future = executor.submit(worker_func, t)
            futures[future] = t
            
        # Thu thập kết quả khi các luồng hoàn thành
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"   [!] Luồng xử lý mã {futures[future]} văng lỗi: {e}")
                
        return results

    def fetch_macro_data(self):
        print("\n" + "="*50)
        print("     TẢI DỮ LIỆU VĨ MÔ")
        print("="*50)
        today_str = datetime.now().strftime('%Y-%m-%d')

        def _update_macro_file(fetch_func, file_name, default_start='2020-01-01'):
            file_path = self.folders['macro'] / file_name
            old_df = None
            start_d = default_start

            # 1. ĐỌC DỮ LIỆU CŨ VÀ TÌM NGÀY CUỐI CÙNG
            if file_path.exists():
                try:
                    old_df = pd.read_parquet(file_path)
                    if 'time' in old_df.columns:
                        # Chuẩn hóa time từ Epoch ms (nếu có) sang Datetime
                        if pd.api.types.is_numeric_dtype(old_df['time']):
                            old_df['time'] = pd.to_datetime(old_df['time'], unit='ms').dt.normalize()
                        else:
                            old_df['time'] = pd.to_datetime(old_df['time']).dt.normalize()
                        
                        # lùi lại 5 ngày vì số liệu có thể đc cập nhật lại
                        last_date = old_df['time'].max() - timedelta(days=5)
                        start_d = last_date.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"   [!] Lỗi đọc file {file_name} cũ: {e}")

            print(f"   -> Đang cập nhật {file_name} từ {start_d} đến nay...")
            
            # 2. TẢI DỮ LIỆU MỚI VÀ MERGE
            try:
                new_df = fetch_func(start_d, today_str)
                
                if new_df is not None and not new_df.empty:
                    if 'time' in new_df.columns:
                        # Chuẩn hóa time của data mới
                        if pd.api.types.is_numeric_dtype(new_df['time']):
                            new_df['time'] = pd.to_datetime(new_df['time'], unit='ms').dt.normalize()
                        else:
                            new_df['time'] = pd.to_datetime(new_df['time']).dt.normalize()
                        
                    # Gộp dữ liệu
                    if old_df is not None and not old_df.empty and 'time' in old_df.columns:
                        combined = pd.concat([old_df, new_df])
                        # Vì dữ liệu mới đã được lọc sạch rác ở hàm fetch_func bên dưới, 
                        # nên dùng keep='last' ở đây là hoàn toàn an toàn
                        combined = combined.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                    else:
                        combined = new_df
                        
                    combined.to_parquet(file_path, engine='pyarrow')
                    print(f"      [OK] Đã lưu tổng cộng {len(combined)} dòng vào {file_name}.")
                else:
                    print(f"      [-] Không có dữ liệu mới cho {file_name}.")
            except Exception as e:
                print(f"   [!] Lỗi khi lấy dữ liệu {file_name}: {e}")
        
        # 1. Tỷ giá USD/VND
        def fetch_fx(start, end):
            # df = self.macro.exchange_rate(start=start, end=end, period='day')
            df = self.macro.currency().exchange_rate(start=start, end=end, period='day')
            if df is not None and not df.empty:
                df = df.reset_index() # Hạ index xuống thành cột
                
                if 'report_time' in df.columns:
                    df = df.rename(columns={'report_time': 'time'})
                    
                # Lọc bỏ dòng null và rác
                if 'value' in df.columns:
                    df = df.dropna(subset=['value'])
                if 'name' in df.columns:
                    df = df[df['name'].str.contains("trung tâm", na=False, case=False)]

                required_cols = ['time', 'name', 'value']
                df = self._validate_schema(df, required_cols, item_name=f"FX USD/VND")
                if df is None: return None
            return df
            
        _update_macro_file(fetch_fx, "usd_vnd.parquet")

        # 2. Giá Dầu Thô
        def fetch_oil(start, end):
            commodity = CommodityPrice(start=start, end=end)
            df = commodity.oil_crude()
            if df is not None and not df.empty:
                df = df.reset_index()
                
                # Tùy version vnstock, cột time có thể tên là 'date', ta gom hết về 'time'
                if 'date' in df.columns and 'time' not in df.columns:
                    df = df.rename(columns={'date': 'time'})

                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                df = self._validate_schema(df, required_cols, item_name=f"CRUDE OIL")
                if df is None: return None
            return df
            
        _update_macro_file(fetch_oil, "crude_oil.parquet")

        # 3. Chỉ số VN-Index (Thị trường chung)
        def fetch_vnindex(start, end):
            quote = Quote(symbol='VNINDEX', source='vci')
            df = quote.history(start=start, end=end, interval='1D')
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            df = self._validate_schema(df, required_cols, item_name=f"VNINDEX")
            if df is None: return None
            if df is not None and not df.empty:
                # Kiểm tra nếu time đang là index thì hạ xuống thành cột
                if 'time' not in df.columns and df.index.name == 'time':
                    df = df.reset_index()
            return df
            
        # Lấy từ 2020 để dư sức tính EMA89 cho các năm sau
        _update_macro_file(fetch_vnindex, "vnindex.parquet")

    # XỬ LÝ UPSERT DỮ LIỆU GSO / SBV
    def _get_gso_start_month(self, file_name):
        """
        Tìm tháng cuối cùng trong file Parquet, lùi lại 2 tháng để vét Revision
        dành cho: GDP, CFI, IP, IE, Retail, MS, FDI
        """
        path = self.folders['macro'] / file_name
        if not path.exists(): 
            return '2020-01'
        try:
            df = pd.read_parquet(path)
            # Tương thích ngược: Đọc cả file cũ (report_time/date) và file mới (time)
            time_col = 'time' if 'time' in df.columns else ('report_time' if 'report_time' in df.columns else 'date')
            max_date = pd.to_datetime(df[time_col]).max()
            start_date = max_date - timedelta(days=90) # Lùi 90 ngày
            return start_date.strftime('%Y-%m')
        except: 
            return '2020-01'

    def _upsert_gso_macro(self, df_new, file_name, req_source=None, dedup_cols=None):
        """
        Động cơ Upsert thông minh:
        - Đổi tên cột thành 'time'
        - Lọc theo Nguồn (Source) chuẩn
        - Xóa trùng lặp theo tổ hợp Dimension (dedup_cols)
        - Ưu tiên giữ lại bản ghi có 'last_updated' mới nhất
        """
        if df_new is None or df_new.empty:
            print(f"      [-] Không có dữ liệu mới cho {file_name}.")
            return

        # 1. HẠ INDEX XUỐNG THÀNH CỘT
        df_new = df_new.reset_index()
        
        # Đổi tên cột thời gian về 'time'
        if 'report_time' in df_new.columns:
            df_new = df_new.rename(columns={'report_time': 'time'})

        required_cols = ['time', 'value', 'name', 'source', 'last_updated']
        df_new = self._validate_schema(df_new, required_cols, item_name=f"GSO Macro - {file_name}")
        if df_new is None: return

        # 2. CHUẨN HÓA THỜI GIAN
        if 'time' in df_new.columns:
            if pd.api.types.is_numeric_dtype(df_new['time']):
                # cho TH cột time là mili giây
                df_new['time'] = pd.to_datetime(df_new['time'], unit='ms').dt.normalize()
            else:
                # cho TH của GDP có ngày dạng '2025-06-30'
                df_new['time'] = pd.to_datetime(df_new['time'], errors='coerce').dt.normalize()

        # 3. BỘ LỌC NGUỒN CHÍNH THỐNG (Source Filter)
        if req_source and 'source' in df_new.columns:
            df_new = df_new[df_new['source'] == req_source].copy()
            if df_new.empty:
                print(f"      [-] Không có dữ liệu thuộc nguồn: {req_source}.")
                return

        # 4. ĐỌC DỮ LIỆU CŨ VÀ MERGE
        path = self.folders['macro'] / file_name
        if path.exists():
            df_old = pd.read_parquet(path)
            # Tương thích cấu trúc file cũ nếu lỡ lưu bằng report_time
            if 'report_time' in df_old.columns:
                df_old = df_old.rename(columns={'report_time': 'time'})
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # 5. XÓA TRÙNG LẶP DỰA TRÊN 'LAST_UPDATED'
        # Loại bỏ các cột không tồn tại để tránh văng lỗi code
        valid_dedup_cols = [c for c in dedup_cols if c in df_combined.columns]
        
        if 'last_updated' in df_combined.columns:
            # Sắp xếp theo [Nhóm Định danh] -> [last_updated]. Bản ghi update mới nhất sẽ nằm ở cuối (last)
            sort_order = valid_dedup_cols + ['last_updated']
            df_combined = df_combined.sort_values(sort_order)
        else:
            df_combined = df_combined.sort_values(valid_dedup_cols)

        # Xóa trùng lặp, giữ dòng cuối cùng (bản ghi có last_updated lớn nhất)
        df_combined = df_combined.drop_duplicates(subset=valid_dedup_cols, keep='last')

        # 6. LƯU PARQUET
        df_combined.to_parquet(path, engine='pyarrow')
        print(f"      [OK] Đã lưu tổng cộng {len(df_combined)} dòng vào {file_name}.")

    # 4. Tổng sản phẩm quốc nội (GDP)
    def fetch_macro_gdp(self):
        print("   -> Đang tải Tổng sản phẩm quốc nội (GDP)...")
        file_name = 'vn_gdp.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.gdp(start=start_m, end=end_m)
        df = self.macro.economy().gdp(start=start_m, end=end_m)
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Tổng cục thống kê",
            dedup_cols=['time', 'report_type', 'group_name', 'name']
        )


    # 5. Chỉ số giá tiêu dùng (CPI)
    def fetch_macro_cpi(self):
        print("   -> Đang tải Chỉ số giá tiêu dùng (CPI)...")
        file_name = 'vn_cpi.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.cpi(start=start_m, end=end_m, period='month')
        df = self.macro.economy().cpi(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Tổng cục thống kê",
            dedup_cols=['time', 'name']
        )

    # 6. Chỉ số sản xuất công nghiệp (IP)
    def fetch_macro_ip(self):
        print("   -> Đang tải Chỉ số sản xuất công nghiệp (IP)...")
        file_name = 'vn_ip.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.industry_prod(start=start_m, end=end_m, period='month')
        df = self.macro.economy().industry_prod(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source=None,  # Bỏ qua kiểm tra Source vì mặc định là null
            dedup_cols=['time', 'group_name', 'name']
        )

    # 7. Xuất nhập khẩu (IE)
    def fetch_macro_ie(self):
        print("   -> Đang tải Xuất nhập khẩu (IE)...")
        file_name = 'vn_ie.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.import_export(start=start_m, end=end_m, period='month')
        df = self.macro.economy().import_export(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Tổng cục thống kê",
            dedup_cols=['time', 'name']
        )

    # 8. Doanh thu bán lẻ (Retail)
    def fetch_macro_retail(self):
        print("   -> Đang tải Doanh thu bán lẻ (Retail)...")
        file_name = 'vn_retail.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.retail(start=start_m, end=end_m, period='month')
        df = self.macro.economy().retail(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Tổng cục thống kê",
            dedup_cols=['time', 'name']
        )

    # 9. Vốn đầu tư nước ngoài (FDI)
    def fetch_macro_fdi(self):
        print("   -> Đang tải Vốn đầu tư nước ngoài (FDI)...")
        file_name = 'vn_fdi.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.fdi(start=start_m, end=end_m, period='month')
        df = self.macro.economy().fdi(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Cục Đầu tư nước ngoài",
            dedup_cols=['time', 'group_name', 'name']
        )

    # 10. Cung tiền & Tín dụng (MS)
    def fetch_macro_ms(self):
        print("   -> Đang tải Cung tiền & Tín dụng (MS)...")
        file_name = 'vn_ms.parquet'
        start_m = self._get_gso_start_month(file_name)
        end_m = datetime.now().strftime('%Y-%m')
        # df = self.macro.money_supply(start=start_m, end=end_m, period='month')
        df = self.macro.economy().money_supply(start=start_m, end=end_m, period='month')
        self._upsert_gso_macro(
            df_new=df, file_name=file_name,
            req_source="Ngân hàng Nhà nước Việt Nam",
            dedup_cols=['time', 'name']
        )

    def fetch_foreign_flow(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🦈 TẢI GIA TĂNG DÒNG TIỀN KHỐI NGOẠI (MULTI-THREADING)")
        print("="*50)
        
        out_path = self.folders['macro'] / 'foreign_flow.parquet'
        existing_df = pd.DataFrame()
        last_dates = {}
        
        # 1. Đọc file cũ để lấy "Checkpoint" thời gian
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty and 'ticker' in existing_df.columns and 'time' in existing_df.columns:
                last_dates = existing_df.groupby('ticker')['time'].max().to_dict()
                
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        
        def _fetch_single_ticker(ticker):
            # Xác định Start Date
            if ticker in last_dates:
                start_date_obj = last_dates[ticker] - timedelta(days=2) # Overlap 2 ngày
                start_str = start_date_obj.strftime('%Y-%m-%d')
            else:
                start_str = '2020-01-01'
                
            try:
                trading = Trading(symbol=ticker, source='vci')
                df_foreign = trading.foreign_trade(start=start_str, end=current_date_str)
                required_cols = ['trading_date', 'fr_buy_volume_total', 'fr_buy_value_total', 'fr_sell_volume_total', 'fr_sell_value_total', 'fr_net_volume_total', 'fr_net_value_total']
                df_foreign = self._validate_schema(df_foreign, required_cols, item_name=f"Foreign Flow - {ticker}")
                if df_foreign is None: return None
                if df_foreign is not None and not df_foreign.empty:
                    df_foreign['ticker'] = ticker
                    return df_foreign
            except Exception:
                pass # Bỏ qua lỗi ngầm để không làm sập các luồng khác
            return None

        # CHIA NHỎ DANH SÁCH & CHẠY ĐA LUỒNG
        new_data_list = []
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_single_ticker, chunk)
                new_data_list.extend(res_list)

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        if not new_data_list:
            # print("[✓] Dữ liệu khối ngoại đã là mới nhất. Không cần cập nhật thêm.")
            return
            
        # 3. CHUẨN HÓA DỮ LIỆU MỚI
        print("[*] Đang gộp và chuẩn hóa dữ liệu...")
        df_new = pd.concat(new_data_list, ignore_index=True)
        df_new = df_new.rename(columns={
            'trading_date': 'time',
            'fr_buy_volume_total': 'foreign_buy_volume',
            'fr_buy_value_total': 'foreign_buy_value',
            'fr_sell_volume_total': 'foreign_sell_volume',
            'fr_sell_value_total': 'foreign_sell_value',
            'fr_net_volume_total': 'foreign_net_volume',
            'fr_net_value_total': 'foreign_net_value'
        })
        
        # Chuyển Epoch ms thành Datetime chuẩn
        if 'time' in df_new.columns:
            if pd.api.types.is_numeric_dtype(df_new['time']):
                df_new['time'] = pd.to_datetime(df_new['time'], unit='ms').dt.normalize()
            else:
                df_new['time'] = pd.to_datetime(df_new['time']).dt.normalize()
                
        # Lọc rác
        cols_to_keep = ['time', 'ticker', 'foreign_buy_volume', 'foreign_buy_value', 'foreign_sell_volume', 'foreign_sell_value', 'foreign_net_volume', 'foreign_net_value']
        df_new = df_new[[c for c in cols_to_keep if c in df_new.columns]]
        
        # 4. UPSERT VÀ KHỬ TRÙNG LẶP
        if not existing_df.empty:
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined = df_combined.sort_values(['ticker', 'time'])
        df_combined = df_combined.drop_duplicates(subset=['ticker', 'time'], keep='last')
        
        # Lưu ổ cứng
        df_combined.to_parquet(out_path, engine='pyarrow')
        print(f" [OK] Đã lưu {len(df_combined):,} dòng vào {out_path}")

    def fetch_prop_flow(self, tickers, max_workers=5):    
        print("\n" + "="*50)
        print(" 🦈 TẢI GIA TĂNG DÒNG TIỀN TỰ DOANH (MULTI-THREADING)")
        print("="*50)
        
        out_path = self.folders['macro'] / 'prop_flow.parquet'
        existing_df = pd.DataFrame()
        last_dates = {}
        
        # 1. Đọc file cũ để lấy "Checkpoint" thời gian
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty and 'ticker' in existing_df.columns and 'time' in existing_df.columns:
                last_dates = existing_df.groupby('ticker')['time'].max().to_dict()
                
        current_date_str = datetime.now().strftime('%Y-%m-%d')
        
        def _fetch_single_ticker(ticker):
            # Xác định Start Date
            if ticker in last_dates:
                start_date_obj = last_dates[ticker] - timedelta(days=2) # Overlap 2 ngày
                start_str = start_date_obj.strftime('%Y-%m-%d')
            else:
                start_str = '2020-01-01'
                
            try:
                trading = Trading(symbol=ticker, source='vci')
                df_prop = trading.prop_trade(start=start_str, end=current_date_str)
                required_cols = ['trading_date', 'total_buy_trade_volume', 'total_buy_trade_value', 'total_sell_trade_volume', 'total_sell_trade_value', 'total_trade_net_volume', 'total_trade_net_value']
                df_prop = self._validate_schema(df_prop, required_cols, item_name=f"Prop Flow - {ticker}")
                if df_prop is None: return None
                if df_prop is not None and not df_prop.empty:
                    df_prop['ticker'] = ticker
                    return df_prop
            except Exception:
                pass # Bỏ qua lỗi ngầm để không làm sập các luồng khác
            return None

        new_data_list = []
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_single_ticker, chunk)
                new_data_list.extend(res_list)

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        if not new_data_list:
            # print("[✓] Dữ liệu Tự doanh đã là mới nhất. Không cần cập nhật thêm.")
            return
            
        # 3. CHUẨN HÓA DỮ LIỆU MỚI
        print("[*] Đang gộp và chuẩn hóa dữ liệu Tự doanh...")
        df_new = pd.concat(new_data_list, ignore_index=True)
        
        # Đổi tên cột cho chuẩn với file JSON bạn gửi
        df_new = df_new.rename(columns={
            'trading_date': 'time',
            'total_buy_trade_volume': 'prop_buy_volume',
            'total_buy_trade_value': 'prop_buy_value',
            'total_sell_trade_volume': 'prop_sell_volume',
            'total_sell_trade_value': 'prop_sell_value',
            'total_trade_net_volume': 'prop_net_volume',
            'total_trade_net_value': 'prop_net_value'
        })
        
        # Chuyển Epoch ms thành Datetime chuẩn
        if 'time' in df_new.columns:
            if pd.api.types.is_numeric_dtype(df_new['time']):
                df_new['time'] = pd.to_datetime(df_new['time'], unit='ms').dt.normalize()
            else:
                df_new['time'] = pd.to_datetime(df_new['time']).dt.normalize()
                
        # Lọc rác, chỉ giữ lại các cột cốt lõi phục vụ tính toán
        cols_to_keep = ['time', 'ticker', 'prop_buy_volume', 'prop_buy_value', 'prop_sell_volume', 'prop_sell_value', 'prop_net_volume', 'prop_net_value']
        df_new = df_new[[c for c in cols_to_keep if c in df_new.columns]]
        
        # 4. UPSERT VÀ KHỬ TRÙNG LẶP
        if not existing_df.empty:
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined = df_combined.sort_values(['ticker', 'time'])
        df_combined = df_combined.drop_duplicates(subset=['ticker', 'time'], keep='last')
        
        # Lưu ổ cứng
        df_combined.to_parquet(out_path, engine='pyarrow')
        print(f"\n[OK] Đã lưu: {len(df_combined):,} dòng vào {out_path}")

    def save_parquet(self, data_list, folder_key, file_name, partition=False):
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            path = self.folders[folder_key] / file_name
            if partition:
                df.to_parquet(path, engine='pyarrow', partition_cols=['ticker'])
            else:
                df.to_parquet(path, engine='pyarrow')
            print(f" [OK] Đã lưu {len(df):,} dòng vào {folder_key.upper()}")

    def fetch_company_info(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🔄 TẢI GIA TĂNG THÔNG TIN DOANH NGHIỆP (MULTI-THREADING)")
        print("="*50)
        # all_companies = []
        updated_companies, updated_company_tickers = [], set()
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        def _fetch_company_worker(ticker):
            if len(ticker) > 3: return None # Bỏ qua quỹ ETF
            try:
                company = Company(symbol=ticker, source=self.source)
                df = company.overview()
                if df is not None and not df.empty:
                    df['ticker'] = ticker
                    return df
            except Exception: pass
            return None

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_company_worker, chunk)
                # all_companies.extend(res_list)
                for res in res_list:
                    updated_companies.append(res)
                    updated_company_tickers.add(res['ticker'].iloc[0])

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        if not updated_companies:
            return

        # self.save_parquet(all_companies, 'company', "master_company.parquet")
        self.merge_and_save(updated_companies, updated_company_tickers, tickers, self.old_coms, 'company', "master_company.parquet")

    
    def merge_and_save(self, updated_list, updated_tickers_set, tickers, old_dict, folder_key, file_name):
        final_list = list(updated_list) # Copy các mã đã cập nhật
        
        # Bổ sung lại các mã KHÔNG có sự thay đổi từ dữ liệu cũ
        for t, df_old in old_dict.items():
            if t not in updated_tickers_set:
                final_list.append(df_old)
                
        if final_list:
            master_df = pd.concat(final_list, ignore_index=True)
            path = self.folders[folder_key] / file_name
            master_df.to_parquet(path, engine='pyarrow') # Ghi đè file duy nhất
            print(f" [OK] Đã lưu {len(master_df):,} dòng vào {file_name}")

    def fetch_ohlcv(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🔄 TẢI GIA TĂNG LỊCH SỬ OHLCV (MULTI-THREADING)")
        print("="*50)
        updated_prices, updated_price_tickers = [], set()
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        def _fetch_price_worker(ticker):
            old_df = self.old_prices.get(ticker, None)
            start_date = '2020-01-01'
            today_str = datetime.now().strftime('%Y-%m-%d')

            # 1. KIỂM TRA LỊCH SỬ ĐỂ LẤY NGÀY BẮT ĐẦU
            if old_df is not None and not old_df.empty:
                if 'time' in old_df.columns:
                    # lấy lùi 5 ngày
                    last_date = pd.to_datetime(old_df['time']).max() - timedelta(days=5)
                    start_date = last_date.strftime('%Y-%m-%d')

            try:
                quote = Quote(symbol=ticker, source=self.source)
                new_df = quote.history(start=start_date, end=today_str, interval='1D')
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                new_df = self._validate_schema(new_df, required_cols, item_name=f"OHLCV - {ticker}")
                if new_df is None: return None
                if new_df is not None and not new_df.empty:
                    new_df['ticker'] = ticker
                    for col in ['open', 'high', 'low', 'close']:
                        if col in new_df.columns and new_df[col].max() < 1000:
                            new_df[col] = new_df[col] * 1000
                    
                    # 2. MERGE DỮ LIỆU CŨ VÀ MỚI
                    if old_df is not None:
                        old_df['time'] = pd.to_datetime(old_df['time'], unit='ms')
                        new_df['time'] = pd.to_datetime(new_df['time'], unit='ms')
                        
                        combined = pd.concat([old_df, new_df])
                        # Xóa trùng lặp dựa vào ngày (time), giữ lại bản cập nhật mới nhất (last)
                        combined = combined.sort_values('time').drop_duplicates(subset=['time'], keep='last')
                        return combined
                    
                    return new_df
            except Exception: pass
            return None

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_price_worker, chunk)
                for res in res_list:
                    updated_prices.append(res)
                    updated_price_tickers.add(res['ticker'].iloc[0])

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        self.merge_and_save(updated_prices, updated_price_tickers, tickers, self.old_prices, 'price', "master_price.parquet")

    def fetch_intra(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🔄 TẢI GIA TĂNG DỮ LIỆU KHỚP LỆNH (MULTI-THREADING)")
        print("="*50)
        updated_intraday, updated_intra_tickers = [], set()
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        
        def _fetch_intraday_worker(ticker):
            old_df = self.old_intraday.get(ticker, None)
            try:
                quote = Quote(symbol=ticker, source=self.source)
                new_df = quote.intraday()
                
                required_cols = ['time', 'price', 'volume', 'match_type', 'id']
                new_df = self._validate_schema(new_df, required_cols, item_name=f"INTRADAY - {ticker}")
                if new_df is None: return None

                if new_df is not None and not new_df.empty:
                    new_df['ticker'] = ticker
                    if 'price' in new_df.columns and new_df['price'].max() < 1000:
                        new_df['price'] = new_df['price'] * 1000

                    # ==========================================================
                    # LƯU LẠI LỊCH SỬ INTRADAY (ROLLING WINDOW)
                    # ==========================================================
                    if old_df is not None and not old_df.empty:
                        old_df['time'] = pd.to_datetime(old_df['time'], unit='ms' if pd.api.types.is_numeric_dtype(old_df['time']) else None)
                        new_df['time'] = pd.to_datetime(new_df['time'], unit='ms' if pd.api.types.is_numeric_dtype(new_df['time']) else None)
                        
                        # 1. Gộp TOÀN BỘ dữ liệu cũ và mới (Không vứt bỏ T-1 nữa)
                        combined = pd.concat([old_df, new_df])
                        
                        # 2. Xóa lệnh trùng (Giữ an toàn khi chạy Bot nhiều lần trong 1 ngày)
                        subset_cols = [c for c in ['time', 'price', 'volume', 'match_type', 'id'] if c in combined.columns]
                        combined = combined.drop_duplicates(subset=subset_cols, keep='last')
                        
                        combined = combined.sort_values('time')
                        
                        # 3. BẢO VỆ Ổ CỨNG & RAM: CHỈ GIỮ LẠI LỊCH SỬ 30 NGÀY GẦN NHẤT
                        unique_dates = combined['time'].dt.date.unique()
                        if len(unique_dates) > 30:
                            # Lấy mốc cắt là ngày thứ 30 tính từ dưới lên
                            cutoff_date = unique_dates[-30]
                            # Chuyển cutoff_date (datetime.date) sang pd.Timestamp để so sánh
                            cutoff_timestamp = pd.Timestamp(cutoff_date)
                            combined = combined[combined['time'] >= cutoff_timestamp]
                            
                        return combined

                    return new_df
            except Exception: pass
            return None

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_intraday_worker, chunk)
                for res in res_list:
                    updated_intraday.append(res)
                    updated_intra_tickers.add(res['ticker'].iloc[0])

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        self.merge_and_save(updated_intraday, updated_intra_tickers, tickers, self.old_intraday, 'intraday', "master_intraday.parquet")

    def fetch_fin_report(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🔄 TẢI GIA TĂNG BÁO CÁO TÀI CHÍNH (MULTI-THREADING)")
        print("="*50)
        # all_fin = []
        updated_fins, updated_fin_tickers = [], set()
        chunk_size = 100
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

        def _fetch_finance_worker(ticker):
            if len(ticker) > 3: return None # Bỏ qua quỹ ETF
            
            try:                
                # Ép dùng nguồn 'kbs' riêng cho Báo cáo tài chính
                fin = Finance(symbol=ticker, source='kbs') 
                df = fin.ratio(period='quarter')
                
                if df is not None and not df.empty:
                    # ==================================================
                    # 🛡️ KIỂM DUYỆT BÁO CÁO TÀI CHÍNH (TRẠM 2 LỚP)
                    # ==================================================
                    # Lớp 1: Kiểm tra Cấu trúc Cột cơ bản (Schema Validation)
                    required_cols = ['item', 'item_id'] 
                    df = self._validate_schema(df, required_cols, item_name=f"Finance Structure - {ticker}")
                    if df is None: return None
                        
                    # Lớp 2: Kiểm tra Nội dung Dòng (Data Values Validation)
                    # Soi vào cột 'item_id' xem có đủ các "chỉ số sống còn" của CANSLIM không
                    # Dựa theo file JSON mẫu của anh, tên các chỉ số là 'roe', 'roa', 'p_e', 'p_b'
                    required_metrics = ['roe', 'roa', 'p_e', 'p_b', 'trailing_eps']
                    
                    # Lấy danh sách các chỉ số mà API thực tế trả về
                    available_metrics = df['item_id'].tolist()
                    
                    missing_metrics = [m for m in required_metrics if m not in available_metrics]
                    
                    if missing_metrics:
                        print(f"   [!] LỖI DỮ LIỆU [Finance - {ticker}]: API trả thiếu chỉ số {missing_metrics}. Đã chặn!")
                        return None # Rác/Thiếu -> Bỏ qua mã này, không merge vào Parquet
                    # ==================================================
                    
                    df['ticker'] = ticker
                    return df
                
                if df is None: return None
            except Exception as e:
                print(f"   [!] Bỏ qua Finance {ticker}: {e}")
                return None

        print(f"[*] Đang khởi động {max_workers} luồng tải song song")
        for chunk_idx, chunk in enumerate(ticker_chunks):
            print(f">>> Đang xử lý Chunk {chunk_idx + 1}/{len(ticker_chunks)} ({len(chunk)} mã)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                res_list = self._run_throttled_tasks(executor, _fetch_finance_worker, chunk)
                # all_fin.extend(res_list)
                for res in res_list:
                    updated_fins.append(res)
                    updated_fin_tickers.add(res['ticker'].iloc[0])

            # Nghỉ ngơi giữa các chunk
            if chunk_idx < len(ticker_chunks) - 1:
                print("   [Zzz] Nghỉ 2 giây trước khi qua chunk mới...")
                time.sleep(2)

        if not updated_fins:
            return

        # self.save_parquet(all_fin, 'financial', "master_financial.parquet")
        self.merge_and_save(updated_fins, updated_fin_tickers, tickers, self.old_fins, 'financial', "master_financial.parquet")

    def fetch_funds(self):
        try:
            print("\n" + "="*50)
            print("     TẢI DANH SÁCH QUỸ TRÁI PHIẾU BOND")
            print("="*50)
            # Chỉ lấy Quỹ trái phiếu
            f = Fund()
            df_funds = f.listing(fund_type='BOND')
            if df_funds is not None and not df_funds.empty:
                fund_path = self.folders['macro'] / "bond_fund.parquet"
                df_funds.to_parquet(fund_path, engine='pyarrow')
                print(f" [OK] Đã lưu ds quỹ trái phiếu {len(df_funds)} mã.")
        except Exception as e:
            print(f" [!] Lỗi tải Quỹ trái phiếu: {e}")

    def fetch_index_components(self):
        print("\n" + "="*50)
        print(" TẢI DANH SÁCH RỔ CHỈ SỐ (INDEX COMPONENTS)")
        print("="*50)
        
        indices = ['HOSE', 'VN30', 'VNMidCap', 'VNSmallCap']
        all_components = []
        
        for idx in indices:
            try:
                # Gọi API lấy danh sách mã theo nhóm
                symbols_data = self.listing.symbols_by_group(group=idx)
                
                # Xử lý chuẩn hóa đầu ra (Series hoặc DataFrame)
                if isinstance(symbols_data, pd.Series): 
                    tickers = symbols_data.tolist()
                else: 
                    tickers = symbols_data['ticker'].tolist() if 'ticker' in symbols_data.columns else symbols_data.iloc[:, 0].tolist()
                
                if tickers:
                    # Tạo DataFrame Map: Tickers <-> Index_Code
                    df_idx = pd.DataFrame({'ticker': tickers, 'index_code': idx})
                    all_components.append(df_idx)
                    print(f" [OK] Đã tải rổ {idx:<10}: {len(tickers)} mã.")
            except Exception as e:
                print(f" [!] Lỗi tải rổ {idx}: {e}")

        if all_components:
            df_final = pd.concat(all_components, ignore_index=True)
            out_path = self.folders['macro'] / 'index_components.parquet'
            df_final.to_parquet(out_path, engine='pyarrow')
            print(f" [OK] Đã lưu Bản đồ phân rổ vào: {out_path}")

    def is_macro_update_window(self):
        # Kiểm tra xem hôm nay có nằm trong cửa sổ cập nhật Vĩ mô không (Từ ngày 26 đến mùng 5).
        today = datetime.now().day
        if today >= 26 or today <= 5:
            return True
        return False

    def get_ticker_list(self):
        index_path = self.folders['macro'] / 'index_components.parquet'
        if index_path.exists():
            try: 
                df_idx = pd.read_parquet(index_path)
                tickers = []
                if self.get_group in ['HOSE', 'VN30', 'VNMidCap', 'VNSmallCap']:
                    tickers = df_idx[df_idx['index_code'] == self.get_group]['ticker'].tolist()
                elif self.get_group == "VN100":
                    vn30_tickers = df_idx[df_idx['index_code'] == 'VN30']['ticker'].tolist()
                    mid_tickers = df_idx[df_idx['index_code'] == 'VNMidCap']['ticker'].tolist()
                    tickers = list(set(vn30_tickers + mid_tickers))
                return tickers
            except: 
                print(f"Could NOT read {index_path}")
                return []
        return []

    def fetch_put_through(self):
        print("\n" + "="*50)
        print("     TẢI DỮ LIỆU GIAO DỊCH THỎA THUẬN (PUT-THROUGH)")
        print("="*50)
        try:
            trading = Trading(source='VCI')
            df_pt = trading.put_through()
            required_cols = ['time', 'symbol', 'price', 'volume', 'match_value', 'change_percent']
            df_pt = self._validate_schema(df_pt, required_cols, item_name="Put-Through")

            if df_pt is not None and not df_pt.empty:
                # Chuẩn hóa thời gian từ mili-giây
                if pd.api.types.is_numeric_dtype(df_pt['time']):
                    df_pt['time'] = pd.to_datetime(df_pt['time'], unit='ms').dt.normalize()
                else:
                    df_pt['time'] = pd.to_datetime(df_pt['time']).dt.normalize()

                pt_path = self.folders['intraday'] / "master_put_through.parquet"
                if pt_path.exists():
                    old_df = pd.read_parquet(pt_path)
                    combined = pd.concat([old_df, df_pt], ignore_index=True)
                    # Xóa trùng lặp theo Ngày, Mã, Giá và Khối lượng
                    combined = combined.drop_duplicates(subset=['time', 'symbol', 'price', 'volume', 'change', 'change_percent', 'match_value'], keep='last')
                else:
                    combined = df_pt

                combined.to_parquet(pt_path, engine='pyarrow')
                print(f" [OK] Đã lưu {len(df_pt)} dòng vào master_put_through.parquet")
        except Exception as e:
            print(f" [!] Lỗi tải Giao dịch Thỏa thuận: {e}")

    # ==========================================
    # HÀM CHẠY CHÍNH (PIPELINE)
    # ==========================================
    def run_pipeline(self):
        # Chạy Vĩ Mô trước
        if self.get_macro:
            self.fetch_macro_data()

            # Lấy dữ liệu cho GDP, CFI, IP, IE, Retail, MS, FDI
            if self.is_macro_update_window():
                self.fetch_macro_gdp()
                self.fetch_macro_cpi()
                self.fetch_macro_ip()
                self.fetch_macro_ie()
                self.fetch_macro_retail()
                self.fetch_macro_fdi()
                self.fetch_macro_ms()

        # TẢI BẢN ĐỒ RỔ CHỈ SỐ (INDEX COMPONENTS)
        if getattr(self, 'get_index', False):
            self.fetch_index_components()

        # Tải ds mã theo nhóm ngành
        if self.get_share_group:
            print("\n" + "="*50)
            print("     TẢI DANH SÁCH MÃ THEO NGÀNH (ADAPTER MODE)")
            print("="*50)
            try:
                df_ind_raw = self.listing.symbols_by_industries()
                
                # KIỂM DUYỆT DANH SÁCH SÓNG NGÀNH (CẤU TRÚC MỚI)
                # Bắt theo đúng cấu trúc Long Format của vnstock_data mới
                required_cols = ['symbol', 'icb_level', 'icb_name'] 
                df_ind_raw = self._validate_schema(df_ind_raw, required_cols, item_name="Industry Groups")
                
                if df_ind_raw is None:
                    print("   [-] Dữ liệu Ngành bị lỗi cấu trúc. Giữ nguyên file cũ.")
                else:
                    # ADAPTER: Chuyển đổi từ Dọc (Long) sang Ngang (Wide)
                    # Gom nhóm theo 'symbol', xoay 'icb_level' thành cột, đưa 'icb_name' vào giá trị
                    df_ind = df_ind_raw.pivot_table(
                        index='symbol', 
                        columns='icb_level', 
                        values='icb_name', 
                        aggfunc='first'
                    ).reset_index()
                    
                    # Đổi tên các cột số (1, 2, 3, 4) thành tên cột truyền thống (icb_name1, icb_name2...)
                    # Sử dụng dict comprehension để bọc các level động (tránh lỗi nếu API thiếu level 4)
                    rename_dict = {level: f'icb_name{level}' for level in df_ind.columns if isinstance(level, (int, float))}
                    df_ind = df_ind.rename(columns=rename_dict)
                    
                    # Xóa tên cho cái trục cột (index name) sinh ra do hàm pivot
                    df_ind.columns.name = None 
                    
                    # Lưu file Parquet với cấu trúc Cũ (Tương thích ngược 100%)
                    group_path = self.folders['macro'] / "groups_by_industries.parquet"
                    df_ind.to_parquet(group_path, engine='pyarrow')
                    print(f" [OK] Đã bẻ ngang và lưu danh sách {len(df_ind)} mã vào groups_by_industries.parquet")
            except Exception as e:
                print(f" [!] Lỗi tải Danh sách theo ngành: {e}")

        # Mặc định lấy từ danh sách có sẵn trong index_components.parquet
        print("\n" + "="*50)
        print(f"     LẤY DANH SÁCH MÃ CHỨNG KHOÁN ĐỂ CHUẨN BỊ TẢI DỮ LIỆU")
        print("="*50)
        tickers = self.get_ticker_list()
        print(f" [OK] Đã chuẩn bị danh sách {len(tickers)} mã.")

        # TẢI DANH SÁCH QUỸ TRÁI PHIẾU
        if self.get_fund:
            self.fetch_funds()

        # TẢI BẢNG GIÁ (BATCH PROCESSING)
        if self.get_board:
            print("\n" + "="*50)
            print("     TẢI BẢNG GIÁ REALTIME")
            print("="*50)
            try:
                # Gửi 1 request duy nhất cho toàn bộ danh sách
                board_df = self.trading.price_board(symbols_list=tickers)
                if board_df is not None and not board_df.empty:
                    board_path = self.folders['current'] / "master_board.parquet"
                    board_df.to_parquet(board_path, engine='pyarrow')
                    print(f" [OK] Đã lưu bảng giá {len(board_df)} mã.")
            except Exception as e:
                print(f" [!] Lỗi tải Bảng giá: {e}")

        # TẢI DỮ LIỆU GIAO DỊCH THỎA THUẬN (PUT-THROUGH)
        if self.get_pt:
            self.fetch_put_through()

        # TẢI DỮ LIỆU CÔNG TY & CHẠY ĐA LUỒNG RIÊNG
        if self.get_com:
            self.fetch_company_info(tickers)

        # TẢI DỮ LIỆU FINANCE REPORT & CHẠY ĐA LUỒNG RIÊNG
        if self.get_fin:
            self.fetch_fin_report(tickers)

        # TẢI DỮ LIỆU OHLCV & CHẠY ĐA LUỒNG RIÊNG
        if self.get_price:
            self.fetch_ohlcv(tickers)

        # TẢI DỮ LIỆU INTRA & CHẠY ĐA LUỒNG RIÊNG
        if self.get_intra:
            self.fetch_intra(tickers)

        # TẢI DỮ LIỆU KHỐI NGOẠI & CHẠY ĐA LUỒNG RIÊNG
        if self.get_foreign:
            self.fetch_foreign_flow(tickers)

        # TẢI DỮ LIỆU KHỐI TỰ DOANH & CHẠY ĐA LUỒNG RIÊNG
        if self.get_prop:
            self.fetch_prop_flow(tickers)

if __name__ == "__main__":
    # Test chạy toàn bộ pipeline
    pipeline = VNStockDataPipeline(source='VCI', get_com=True, get_price=True, get_intra=True, get_board=True, get_fin=True)
    pipeline.run_pipeline()