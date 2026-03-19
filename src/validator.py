import pandas as pd
from pathlib import Path

class ValidatePipeline:
    def __init__(self, path_dir=None):
        self.path_dir = Path(path_dir) if path_dir else Path('data/parquet/price/master_price.parquet')

    # def validate_master_price(parquet_path):
    def validate_master_price(self):
        print("==================================================")
        print("    BẮT ĐẦU KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA QA)")
        print("==================================================")
        
        path = self.path_dir
        if not path.exists():
            print(f"[!] Không tìm thấy file tại: {self.path_dir}")
            return
        
        print(f"[*] Đang tải dữ liệu từ {self.path_dir}...")
        df = pd.read_parquet(path)
        print(f"[*] Tổng số dòng (Rows): {len(df):,}")
        print(f"[*] Tổng số mã cổ phiếu: {df['ticker'].nunique():,}")
        
        issues = []
        
        # 1. Kiểm tra Cột (Columns)
        required_cols = ['ticker', 'time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"❌ THIẾU CỘT NGHIÊM TRỌNG: {missing_cols}")
            print("\n".join(issues))
            return # Dừng luôn nếu thiếu cột cơ bản
            
        # 2. Kiểm tra Null / Rỗng
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"⚠️ DỮ LIỆU RỖNG (NULL):\n{null_counts[null_counts > 0].to_string()}")
            
        # 3. Kiểm tra logic OHLC (Giá High phải cao nhất, Low phải thấp nhất)
        # Lỗi 3a: High thấp hơn Open, Close hoặc Low
        invalid_high = df[(df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])]
        if not invalid_high.empty:
            sample_tickers = invalid_high['ticker'].unique()[:5]
            issues.append(f"❌ LỖI LOGIC NẾN (High không phải cao nhất) tại {len(invalid_high):,} dòng. Các mã dính lỗi: {list(sample_tickers)}")
            
        # Lỗi 3b: Low cao hơn Open hoặc Close
        invalid_low = df[(df['low'] > df['open']) | (df['low'] > df['close'])]
        if not invalid_low.empty:
            sample_tickers = invalid_low['ticker'].unique()[:5]
            issues.append(f"❌ LỖI LOGIC NẾN (Low không phải thấp nhất) tại {len(invalid_low):,} dòng. Các mã dính lỗi: {list(sample_tickers)}")
            
        # 4. Kiểm tra Volume âm
        invalid_vol = df[df['volume'] < 0]
        if not invalid_vol.empty:
            issues.append(f"❌ KHỐI LƯỢNG ÂM tại {len(invalid_vol):,} dòng.")
            
        # 5. Kiểm tra Trùng lặp (Duplicate Dates per Ticker)
        # Rất quan trọng để Backtest không bị mua/bán 2 lần trong cùng 1 ngày
        # Chuẩn hóa time trước khi check
        if pd.api.types.is_numeric_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], unit='ms').dt.normalize()
        else:
            df['time'] = pd.to_datetime(df['time']).dt.normalize()
            
        duplicates = df[df.duplicated(subset=['ticker', 'time'], keep=False)]
        if not duplicates.empty:
            sample_tickers = duplicates['ticker'].unique()[:5]
            issues.append(f"⚠️ TRÙNG LẶP NGÀY GIAO DỊCH tại {len(duplicates):,} dòng. Các mã dính lỗi: {list(sample_tickers)}")

        # 6. Kiểm tra các cú nhảy giá bất thường (Vượt quá 30% một phiên - Có thể do chưa chia cổ tức)
        # Tính biến động % so với phiên trước
        df_sorted = df.sort_values(['ticker', 'time'])
        df_sorted['prev_close'] = df_sorted.groupby('ticker')['close'].shift(1)
        df_sorted['pct_change'] = abs((df_sorted['close'] - df_sorted['prev_close']) / df_sorted['prev_close'])
        
        extreme_jumps = df_sorted[df_sorted['pct_change'] > 0.30] # Biến động > 30%
        if not extreme_jumps.empty:
            sample_tickers = extreme_jumps['ticker'].unique()[:5]
            issues.append(f"⚠️ NHẢY GIÁ BẤT THƯỜNG (>30%/phiên) tại {len(extreme_jumps):,} dòng. Có thể do chia cổ tức/chuyển sàn. VD: {list(sample_tickers)}")

        # =================================================
        # TỔNG KẾT
        # =================================================
        print("\n--- BÁO CÁO KẾT QUẢ ---")
        if issues:
            print("🚨 PHÁT HIỆN CÁC VẤN ĐỀ CẦN XỬ LÝ:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ DỮ LIỆU HOÀN HẢO! Không phát hiện lỗi Logic, Null hay Trùng lặp.")
            print("✅ Bạn đã sẵn sàng để chạy Backtester.")

if __name__ == "__main__":
    pipeline = ValidatePipeline()
    pipeline.validate_master_price()