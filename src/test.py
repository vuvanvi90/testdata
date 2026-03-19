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
    def __init__(self, source='VCI'):
        self.source = source.lower() if source else 'vci'

        # Cấu trúc thư mục lưu Parquet
        self.folders = {
            'parquet_base': Path('data/parquet'),
            'macro': Path('data/parquet/macro')
        }
        for f in self.folders.values():
            f.mkdir(parents=True, exist_ok=True)

        # BỘ ĐẾM RATE LIMITER THÔNG MINH
        self.request_timestamps = deque()
        self.rate_lock = threading.Lock()
        self.total_requests_sent = 0
        self.MAX_RPM = 500  # Ngưỡng an toàn tối đa (500 thay vì 600 để chừa hao phí cho thư viện)

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
                    print(f"\n[🛑] ĐẠT GIỚI HẠN AN TOÀN: Đã gửi {self.MAX_RPM} reqs trong 1 phút qua!")
                    print(f"     -> Tạm dừng hệ thống {sleep_duration:.1f} giây để làm mát API...")
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
                print(f"   [📡 API Tracker] Tổng số lệnh đã gọi: {total_sent} | Tốc độ hiện tại: {current_rpm}/{self.MAX_RPM} reqs/phút")

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

    def fetch_foreign_flow(self, tickers, max_workers=5):
        print("\n" + "="*50)
        print(" 🔄 TẢI GIA TĂNG DÒNG TIỀN KHỐI NGOẠI (MULTI-THREADING)")
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

    def run_pipeline(self):
        # Lấy danh sách mã
        print("--- 1. LẤY DANH SÁCH MÃ CHỨNG KHOÁN ---")
        try:
            symbols_data = self.listing.symbols_by_group(group=self.get_group)
            if isinstance(symbols_data, pd.Series): tickers = symbols_data.tolist()
            else: tickers = symbols_data['ticker'].tolist() if 'ticker' in symbols_data.columns else symbols_data.iloc[:, 0].tolist()
            print(f" [OK] Đã tải danh sách {len(tickers)} mã.")
        except Exception as e:
            print(f"[!] Lỗi lấy danh sách: {e}")
            return

        # TẢI DỮ LIỆU KHỐI NGOẠI & CHẠY ĐA LUỒNG RIÊNG
        self.fetch_foreign_flow(tickers)


if __name__ == "__main__":
    # Test chạy toàn bộ pipeline
    pipeline = VNStockDataPipeline(source='VCI')
    pipeline.run_pipeline()