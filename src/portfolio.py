import pandas as pd
import numpy as np
import scipy.optimize as sco
from pathlib import Path

class QuantPortfolioEngine:
    def __init__(self, price_dir=None, output_dir=None):
        self.price_dir = Path(price_dir) / 'master_price.parquet' if price_dir else Path('data/parquet/price/master_price.parquet')
        self.output_dir = Path(output_dir) if output_dir else Path('data/portfolio')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load giá lịch sử (để tính Covariance/Rủi ro)
        self.prices_df = self._load_historical_prices()

    def _load_historical_prices(self):
        """Load giá đóng cửa và làm sạch chuẩn Quant (Chống Zero-Volatility Trap)"""
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
            prices = prices.sort_index().tail(252) # Cắt 1 năm (252 ngày giao dịch)
            
            # Lọc các mã sống sót > 70% thời gian (Chống nhiễu mã mới lên sàn quá ngắn)
            threshold = len(prices) * 0.7 
            prices = prices.dropna(axis=1, thresh=int(threshold))
            
            # Chỉ dùng ffill() để lấp ngày lễ. 
            # Tuyệt đối KHÔNG dùng bfill() để không tạo ra Volatility ảo (=0) ở quá khứ.
            prices = prices.ffill()
            
            return prices
        except Exception as e:
            print(f"[!] Lỗi xử lý ma trận giá: {e}")
            return pd.DataFrame()

    def get_expected_returns(self, forecast_df, valid_tickers):
        """
        Dịch tín hiệu Wyckoff & VPA thành Lợi nhuận kỳ vọng (Expected Returns)
        """
        expected_returns = pd.Series(0.05, index=valid_tickers) 

        for _, row in forecast_df.iterrows():
            ticker = row['Ticker']
            if ticker not in valid_tickers: continue
            
            signal = row['Signal']
            price = float(row['Price'])
            res = float(row.get('Resistance', price))
            
            exp_ret = 0.05
            if signal == 'SOS':
                if res > price:
                    upside = (res - price) / price
                    exp_ret = max(upside, 0.07) 
                else:
                    exp_ret = 0.10 
            elif signal == 'SPRING':
                exp_ret = 0.15 # Bắt đáy thành công -> Kỳ vọng cao 15%
            elif signal == 'TEST_CUNG':
                exp_ret = 0.12 # Rủi ro thấp, dư địa tới kháng cự lớn -> Kỳ vọng 12%
                
            expected_returns[ticker] = exp_ret
            
        return expected_returns

    def optimize_portfolio(self, forecast_data):
        """Quy trình chạy Tối ưu hóa Lượng tử (Max Sharpe bằng Scipy)"""
        # 1. Đọc dữ liệu
        if isinstance(forecast_data, (str, Path)):
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

        num_assets = len(valid_tickers)
        
        # Xử lý ngoại lệ an toàn khi chỉ có 1 mã (Fail-safe)
        if num_assets == 1:
            print("\n" + "="*45)
            print(" 🧠 KHUYẾN NGHỊ PHÂN BỔ VỐN (SINGLE ASSET)")
            print("="*45)
            single_ticker = valid_tickers[0]
            print(f"   👉 {single_ticker}: 100.00%")
            cleaned_weights = {single_ticker: 1.0}
            pd.Series(cleaned_weights).to_csv(self.output_dir / "quant_allocation.csv")
            return cleaned_weights

        prices_filtered = self.prices_df[valid_tickers]

        # 3. Tính Lợi nhuận kỳ vọng và Ma trận Hiệp phương sai
        # Dùng fillna(0) cho pct_change để đảm bảo Pandas không nuốt mất dòng khi có mã bị thiếu dữ liệu ở đầu chu kỳ
        daily_returns = prices_filtered.pct_change().fillna(0)
        
        cov_matrix = daily_returns.cov() * 252 
        expected_returns = self.get_expected_returns(forecast_df, valid_tickers).values

        # 4. THUẬT TOÁN TỐI ƯU HÓA BẰNG SCIPY
        def calc_portfolio_perf(weights):
            port_return = np.sum(expected_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return port_return, port_volatility

        def negative_sharpe(weights):
            p_ret, p_vol = calc_portfolio_perf(weights)
            # Risk-free rate = 3%/năm (Lãi suất tiết kiệm kỳ hạn ngắn)
            return -(p_ret - 0.03) / p_vol if p_vol > 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        max_weight = 0.35 if num_assets >= 3 else 1.0
        bounds = tuple((0.0, max_weight) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]

        try:
            optimal_result = sco.minimize(
                negative_sharpe, 
                init_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )

            # 5. Xử lý kết quả đầu ra
            raw_weights = optimal_result.x
            
            cleaned_weights = {}
            for i, ticker in enumerate(valid_tickers):
                w = round(raw_weights[i], 4)
                if w >= 0.01:
                    cleaned_weights[ticker] = w
                    
            # Chuẩn hóa lại tổng = 100%
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
            
            pd.Series(cleaned_weights).to_csv(self.output_dir / "quant_allocation.csv")
            return cleaned_weights

        except Exception as e:
            print(f"[ERROR] Lỗi giải thuật tối ưu Scipy: {e}")
            return None