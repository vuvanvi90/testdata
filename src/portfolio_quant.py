import pandas as pd
import numpy as np
import scipy.optimize as sco
from pathlib import Path
# from config import settings

class QuantPortfolioEngine:
    def __init__(self, price_dir=None, output_dir=None):
        # price_dir được truyền từ live.py
        self.price_dir = Path(price_dir) / 'master_price.parquet' if price_dir else Path('data/parquet/price/master_price.parquet')
        self.output_dir = Path(output_dir) if output_dir else Path('data/portfolio')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load giá lịch sử (để tính Covariance/Rủi ro)
        self.prices_df = self._load_historical_prices()

    def _load_historical_prices(self):
        """Load giá đóng cửa từ file Parquet và chuyển thành Ma trận giá"""
        if not self.price_dir.exists():
            print(f"[!] Lỗi: Không tìm thấy dữ liệu giá tại {self.price_dir}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.price_dir)
            if pd.api.types.is_numeric_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms').dt.date
            else:
                df['time'] = pd.to_datetime(df['time']).dt.date
                
            prices = df.pivot_table(index='time', columns='ticker', values='close')
            prices = prices.sort_index()

            # CHỈ LẤY LỊCH SỬ 1 NĂM GẦN NHẤT (252 PHIÊN)
            # Giúp các mã mới lên sàn (như NO1) không bị loại bỏ oan uổng
            prices = prices.tail(252) # Cắt lấy 252 ngày giao dịch cuối cùng
            
            # Làm sạch dữ liệu: Mã nào vắng mặt > 30% trong 1 năm qua mới bị loại
            threshold = len(prices) * 0.7  # 252 * 0.7 = ~176 phiên
            prices = prices.dropna(axis=1, thresh=int(threshold))
            
            # Điền khuyết dữ liệu cho các ngày nghỉ lễ / mất thanh khoản
            prices = prices.ffill().bfill()
            
            # Xóa các dòng (ngày) mà vẫn còn NaN (rất hiếm khi xảy ra sau khi ffill)
            prices = prices.dropna()
            
            return prices
        except Exception as e:
            print(f"[!] Lỗi xử lý ma trận giá: {e}")
            return pd.DataFrame()

    def get_expected_returns(self, forecast_df, valid_tickers):
        """
        Dịch tín hiệu Wyckoff thành Lợi nhuận kỳ vọng (Expected Returns)
        """
        expected_returns = pd.Series(0.05, index=valid_tickers) # Mặc định cơ sở 5%/năm

        for _, row in forecast_df.iterrows():
            ticker = row['Ticker']
            if ticker not in valid_tickers: continue
            
            signal = row['Signal']
            price = float(row['Price'])
            res = float(row.get('Resistance', price))
            
            # Định lượng kỳ vọng từ Wyckoff
            exp_ret = 0.05
            if signal == 'SOS':
                if res > price:
                    upside = (res - price) / price
                    exp_ret = max(upside, 0.07) # Tối thiểu 7%
                else:
                    exp_ret = 0.10 # Vượt đỉnh -> Kỳ vọng 10%
            elif signal == 'SPRING':
                exp_ret = 0.15 # Bắt đáy -> Kỳ vọng cao 15%
                
            expected_returns[ticker] = exp_ret
            
        return expected_returns

    def optimize_portfolio(self, forecast_data):
        """Quy trình chạy Tối ưu hóa Lượng tử (Max Sharpe bằng Scipy)"""
        # 1. Đọc dữ liệu
        if isinstance(forecast_data, str) or isinstance(forecast_data, Path):
            try: forecast_df = pd.read_csv(forecast_data)
            except: return None
        elif isinstance(forecast_data, pd.DataFrame):
            forecast_df = forecast_data
        else: return None

        if forecast_df.empty: return None

        # 2. Lọc danh sách mã hợp lệ
        selected_tickers = forecast_df['Ticker'].tolist()
        valid_tickers = [t for t in selected_tickers if t in self.prices_df.columns]

        if not valid_tickers:
            print("[ERROR] Các mã được chọn không đủ dữ liệu giá lịch sử để tính rủi ro.")
            return None

        prices_filtered = self.prices_df[valid_tickers]
        num_assets = len(valid_tickers)

        # 3. Tính Lợi nhuận kỳ vọng và Ma trận Hiệp phương sai
        # Tính tỷ suất lợi nhuận hàng ngày
        daily_returns = prices_filtered.pct_change().dropna()
        
        # Ma trận Covariance (Đã thường niên hóa: nhân với 252 ngày giao dịch)
        cov_matrix = daily_returns.cov() * 252 
        
        # Vector Lợi nhuận kỳ vọng từ Wyckoff
        expected_returns = self.get_expected_returns(forecast_df, valid_tickers).values

        # 4. THUẬT TOÁN TỐI ƯU HÓA BẰNG SCIPY (KHÔNG DÙNG CVXPY)
        # Hàm mục tiêu: Điểm Sharpe Âm (Vì scipy chỉ có hàm minimize, nên ta minimize Negative Sharpe)
        def calc_portfolio_perf(weights):
            port_return = np.sum(expected_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return port_return, port_volatility

        def negative_sharpe(weights):
            p_ret, p_vol = calc_portfolio_perf(weights)
            # Giả định Risk-free rate = 0.03 (3%/năm)
            return -(p_ret - 0.03) / p_vol if p_vol > 0 else 0

        # Điều kiện ràng buộc: Tổng tỷ trọng = 1 (100% vốn)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Giới hạn tỷ trọng mỗi mã: Từ 0% đến Max Weight
        max_weight = 0.35 if num_assets >= 3 else 1.0
        bounds = tuple((0.0, max_weight) for _ in range(num_assets))
        
        # Điểm bắt đầu (Khởi tạo tỷ trọng chia đều)
        init_guess = num_assets * [1. / num_assets,]

        try:
            # CHẠY BỘ GIẢI TOÁN SLSQP (Sequential Least Squares Programming)
            optimal_result = sco.minimize(
                negative_sharpe, 
                init_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )

            # 5. Xử lý kết quả đầu ra
            raw_weights = optimal_result.x
            
            # Làm sạch trọng số (Lọc bỏ các mã < 1%)
            cleaned_weights = {}
            for i, ticker in enumerate(valid_tickers):
                w = round(raw_weights[i], 4)
                if w >= 0.01:
                    cleaned_weights[ticker] = w
                    
            # Chuẩn hóa lại tổng = 100% sau khi cắt tỉa
            total_w = sum(cleaned_weights.values())
            cleaned_weights = {k: v/total_w for k, v in cleaned_weights.items()}

            print("\n" + "="*45)
            print(" 🧠 KHUYẾN NGHỊ PHÂN BỔ VỐN (QUANT MAX-SHARPE)")
            print("="*45)
            
            total_alloc = 0
            sorted_weights = sorted(cleaned_weights.items(), key=lambda item: item[1], reverse=True)
            
            for t, w in sorted_weights:
                print(f"   👉 {t}: {w*100:5.2f}%")
                total_alloc += w
            
            print("-" * 45)
            print(f"Tổng vốn sử dụng: {total_alloc*100:.2f}% (Tối đa/Mã: {max_weight*100:.0f}%)")
            
            # Xuất file
            pd.Series(cleaned_weights).to_csv(self.output_dir / "quant_allocation.csv")
            return cleaned_weights

        except Exception as e:
            print(f"[ERROR] Lỗi giải thuật tối ưu Scipy: {e}")
            return None