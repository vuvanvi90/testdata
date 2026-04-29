import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class DarkPoolRadar:
    def __init__(self, data_dir='data/parquet'):
        self.data_dir = Path(data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Đài quan sát Dòng tiền Ngầm (Dark Pool Radar)...")
        
        self.df_intra = self._load_parquet_safe(self.data_dir / 'intraday/master_intraday.parquet')
        self.df_pt = self._load_parquet_safe(self.data_dir / 'intraday/master_put_through.parquet')
        self.df_price = self._load_parquet_safe(self.data_dir / 'price/master_price.parquet')
        self.df_idx = self._load_parquet_safe(self.data_dir / 'macro/index_components.parquet')
        
        self.t0_date = self._identify_system_t0()

        # Biến lưu trữ kết quả
        self.basket_stats = {}
        self.anomalies = []
        self.forecasts = []

    def _load_parquet_safe(self, path):
        """Đọc file và lột bỏ Timezone an toàn"""
        if path.exists():
            try: 
                df = pd.read_parquet(path)
                if 'time' in df.columns:
                    # Ép kiểu về datetime nếu API trả về dạng số/chuỗi
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        if pd.api.types.is_numeric_dtype(df['time']):
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                        else:
                            df['time'] = pd.to_datetime(df['time'])
                    
                    # Lột bỏ Timezone (Ép về Naive)
                    if getattr(df['time'].dt, 'tz', None) is not None:
                        df['time'] = df['time'].dt.tz_localize(None)
                return df
            except Exception as e: 
                print(f"[!] Lỗi đọc file {path.name}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _identify_system_t0(self):
        # """Xác định ngày giao dịch hiện tại thực tế nhất từ Intraday hoặc Price"""
        latest_date = None
        
        # Thử lấy từ Intraday trước (Dữ liệu mới nhất trong phiên)
        if not self.df_intra.empty and 'time' in self.df_intra.columns:
            latest_date = self.df_intra['time'].max()
            
        # Nếu Intraday rỗng, lấy từ Bảng giá
        elif not self.df_price.empty and 'time' in self.df_price.columns:
            latest_date = self.df_price['time'].max()
            
        # Nếu cả 2 đều rỗng, dùng ngày hiện tại của máy tính
        if pd.isna(latest_date) or latest_date is None:
            latest_date = pd.Timestamp.now()
            
        # Ép chuẩn về Timestamp và Normalize (Cắt giờ/phút/giây về 00:00:00)
        # Điều này đảm bảo nó có thể so sánh dấu "==" với bất kỳ cột 'time' nào trong DataFrame
        return pd.Timestamp(latest_date).normalize()

    def _prepare_data(self):
        """Xử lý dữ liệu nền tảng"""
        if self.df_pt.empty or self.df_price.empty or self.df_idx.empty:
            return False

        # 1. Xác định 5 phiên giao dịch gần nhất từ Bảng giá
        self.df_price['time'] = self.df_price['time'].dt.normalize()
        trading_days = sorted(self.df_price['time'].unique())
        if len(trading_days) >= 5:
            self.last_5_days = trading_days[-5:]
        else:
            self.last_5_days = trading_days

        # Lọc dữ liệu Thỏa thuận trong 5 ngày này
        self.df_pt['time'] = self.df_pt['time'].dt.normalize()
        self.pt_recent = self.df_pt[self.df_pt['time'].isin(self.last_5_days)].copy()

        # 2. Map Rổ chỉ số (VN30, VNMidCap, VNSmallCap) vào data Thỏa thuận
        # Tạo dictionary { 'ACB': 'VN30', ... }
        idx_dict = self.df_idx.set_index('ticker')['index_code'].to_dict()
        self.pt_recent['basket'] = self.pt_recent['symbol'].map(idx_dict).fillna('HOSE')

        # Chỉ giữ lại 3 rổ chính
        self.pt_recent = self.pt_recent[self.pt_recent['basket'].isin(['VN30', 'VNMidCap', 'VNSmallCap'])]

        # 3. Tính Khối lượng Khớp lệnh Trung bình 20 phiên (MA20 Volume) cho TẤT CẢ các mã
        # Lấy 20 ngày gần nhất
        recent_20_days = trading_days[-20:] if len(trading_days) >= 20 else trading_days
        df_price_20d = self.df_price[self.df_price['time'].isin(recent_20_days)]
        
        self.vol_ma20_dict = df_price_20d.groupby('ticker')['volume'].mean().to_dict()
        
        # Lưu lại Giá Đóng cửa cuối cùng để so sánh
        last_day = trading_days[-1]
        self.last_price_dict = self.df_price[self.df_price['time'] == last_day].set_index('ticker')['close'].to_dict()

        return True

    def _analyze_baskets(self):
        """Tầng 1: Góc nhìn Vĩ mô theo Rổ (Basket Heatmap)"""
        if self.pt_recent.empty: return

        # Nhóm theo rổ
        grouped = self.pt_recent.groupby('basket')
        
        for basket, group in grouped:
            total_val_bn = group['match_value'].sum() / 1_000_000_000
            
            # Tính Weighted Premium/Discount
            group['weighted_change'] = group['change_percent'] * group['match_value']
            avg_change = (group['weighted_change'].sum() / group['match_value'].sum()) * 100
            
            if avg_change > 1.0:
                status = f"🔥 PREMIUM (Lệch {avg_change:+.1f}%) -> Dòng tiền ngầm đang GOM ĐẮT!"
            elif avg_change < -1.0:
                status = f"🧊 DISCOUNT (Lệch {avg_change:+.1f}%) -> Áp lực phân phối/hoán đổi nội bộ."
            else:
                status = f"⚖️ NEUTRAL (Lệch {avg_change:+.1f}%) -> Sang tay quanh tham chiếu."
                
            self.basket_stats[basket] = {
                'total_val': total_val_bn,
                'status': status
            }

    def _detect_anomalies(self):
        """Tầng 2: Truy quét Dấu chân Cá mập (Anomaly Detection) - Bản nâng cấp V2"""
        if self.pt_recent.empty: return

        grouped = self.pt_recent.groupby('symbol')
        anomalies_temp = []

        for ticker, group in grouped:
            basket = group['basket'].iloc[0]
            total_val_bn = group['match_value'].sum() / 1_000_000_000
            
            # =================================================================
            # 🚀 THIẾT LẬP NGƯỠNG ĐỘNG (DYNAMIC THRESHOLDS) THEO TỪNG RỔ
            # =================================================================
            if basket == 'VN30':
                min_val = 50.0       # VN30: Dưới 50 Tỷ là nhiễu, bỏ qua
                extreme_val = 200.0  # Mốc Cá mập khổng lồ
                surge_ratio = 1.0    # Chỉ cần Thỏa thuận = 1x MA20 là đã rất khủng khiếp
            elif basket == 'VNMidCap':
                min_val = 15.0       # MidCap: Tối thiểu 15 Tỷ
                extreme_val = 50.0
                surge_ratio = 1.5    # Cần gấp rưỡi thanh khoản sàn mới gọi là đột biến
            else: # VNSmallCap
                min_val = 3.0        # SmallCap/Penny: 3 Tỷ đã là đáng chú ý
                extreme_val = 15.0
                surge_ratio = 2.5    # Phải gấp 2.5 lần thanh khoản bình thường
                
            if total_val_bn < min_val: continue 
            
            # Tính Cường độ (Liquidity Ratio)
            pt_vol = group['volume'].sum()
            ma20_vol = self.vol_ma20_dict.get(ticker, 0)
            liqd_ratio = (pt_vol / ma20_vol) if ma20_vol > 0 else 0
            
            # Tính Weighted Change
            group['weighted_change'] = group['change_percent'] * group['match_value']
            avg_change = (group['weighted_change'].sum() / group['match_value'].sum()) * 100
            
            # Khối lượng Thỏa thuận riêng phiên T0
            t0_group = group[group['time'] == self.t0_date]
            t0_val_bn = t0_group['match_value'].sum() / 1_000_000_000 if not t0_group.empty else 0
            
            # TÌM WHALE NODE
            whale_node = float(group.groupby('price')['volume'].sum().idxmax())
            
            # =================================================================
            # 🚀 LƯỚI LỌC CÁ MẬP ĐÃ CHUẨN HÓA (NORMALIZED STRICT FILTERS)
            # =================================================================
            is_vol_surge = liqd_ratio > surge_ratio # Bùng nổ khối lượng
            is_price_extreme = abs(avg_change) >= 2.0 and total_val_bn > (min_val * 1.5) # Ép giá lộ liễu
            is_true_whale = total_val_bn > extreme_val and liqd_ratio > (surge_ratio * 0.5) # Cá mập thực sự ra tay
            
            if is_vol_surge or is_price_extreme or is_true_whale:
                # Phân loại Ý đồ
                if avg_change > 1.0: intent = f"🔥 Premium ({avg_change:+.1f}%)"
                elif avg_change < -1.0: intent = f"🧊 Discount ({avg_change:+.1f}%)"
                else: intent = f"⚖️ Neutral ({avg_change:+.1f}%)"
                
                # 🚀 HỆ THỐNG CHẤM ĐIỂM ĐỘT BIẾN (NORMALIZED ANOMALY SCORE)
                # Loại bỏ việc cộng giá trị tuyệt đối (total_val_bn) để SmallCap không bị Bluechip đè bẹp.
                # Công thức chuẩn Quant: Score = (Tỷ lệ thanh khoản) + (Biên độ giá) + (Trọng số T0)
                t0_weight = (t0_val_bn / total_val_bn) * 10 if total_val_bn > 0 else 0
                
                anomaly_score = (liqd_ratio * 20) + (abs(avg_change) * 10) + t0_weight
                
                anomalies_temp.append({
                    'ticker': ticker,
                    'basket': basket,
                    'val_bn': total_val_bn,
                    't0_val_bn': t0_val_bn, 
                    'ratio': liqd_ratio,
                    'intent': intent,
                    'whale_node': whale_node,
                    'avg_change': avg_change,
                    'score': anomaly_score
                })
        
        # Sắp xếp theo Anomaly Score giảm dần và lấy Top 10
        self.anomalies = sorted(anomalies_temp, key=lambda x: x['score'], reverse=True)[:10]

    def _forecast_tactics(self):
        """Tầng 3: Động cơ Dự báo T+ (T-Day Forecaster)"""
        for an in self.anomalies:
            ticker = an['ticker']
            whale_node = an['whale_node']
            avg_change = an['avg_change']
            liqd_ratio = an['ratio']
            
            last_price = self.last_price_dict.get(ticker, 0)
            if last_price == 0: continue
            
            dist_to_node = (last_price - whale_node) / whale_node * 100
            forecast = ""
            
            # Đánh giá xem Node này có uy tín không (Dựa trên khối lượng)
            is_node_strong = liqd_ratio > 0.8
            
            # KỊCH BẢN 1: BỆ ĐỠ T+1 (Cần Support mạnh)
            if -2.0 <= dist_to_node <= 1.5 and avg_change > -1.0:
                if is_node_strong:
                    forecast = f"Giá ({last_price:,.0f}) test lại Whale Node ({whale_node:,.0f}) với thanh khoản ngầm ĐỦ LỚN. Lực đỡ uy tín. Khuyến nghị CANH BẮT ĐÁY."
                else:
                    forecast = f"Giá ({last_price:,.0f}) về Node ({whale_node:,.0f}) nhưng Khối lượng thỏa thuận yếu ({liqd_ratio:.1f}x). Bệ đỡ rỗng, dễ thủng. THEO DÕI."
            
            # KỊCH BẢN 2: MÁY BƠM T+3 (Premium)
            elif avg_change > 2.0:
                if dist_to_node > 5.0:
                    forecast = f"Thỏa thuận Premium mạnh nhưng giá sàn ({last_price:,.0f}) đã chạy xa Node ({whale_node:,.0f}). Chờ rũ bỏ T+1/T+2 mới có điểm vào."
                elif dist_to_node < -3.0:
                    forecast = f"⚠️ CÁ MẬP KẸP HÀNG: Mua Premium giá cao nhưng giá sàn rớt thảm. Động lượng cực xấu, cẩn thận vỡ deal. ĐỨNG NGOÀI."
                else:
                    forecast = f"🔥 Khát hàng: Nổ thỏa thuận Premium ngay nền. Deal đã chốt. Dự báo bứt phá T+1/T+2. Khuyến nghị GOM MẠNH."
                    
            # KỊCH BẢN 3: PHÂN PHỐI TỐI (Discount sâu)
            elif avg_change < -2.0:
                if dist_to_node > 4.0:
                    forecast = f"🩸 Giá sàn kéo thốc ({last_price:,.0f}) nhưng táng thỏa thuận Discount sâu. Dấu hiệu Phân phối ngầm / Bull-trap. T+2 TẮM MÁU. BÁN/ĐỨNG NGOÀI."
                else:
                    forecast = f"Sang tay Discount sâu vùng đáy. Lệnh trao tay nội bộ, không phản ánh cung cầu thật. THEO DÕI."
            
            # KỊCH BẢN 4: THAY MÁU CỔ ĐÔNG LỚN (Đột biến thanh khoản ngầm)
            elif liqd_ratio >= 5.0 and -2.0 <= avg_change <= 2.0:
                forecast = f"Thanh khoản ngầm SIÊU ĐỘT BIẾN ({liqd_ratio:.1f}x MA20) quanh tham chiếu. Dấu hiệu thay máu cổ đông lớn hoặc Deal thâu tóm (M&A). Cổ phiếu sắp có sóng lớn. ĐƯA VÀO TẦM NGẮM ĐẶC BIỆT."

            if forecast:
                self.forecasts.append(f"- Kèo [{ticker}]: {forecast}")

    def export_signals(self):
        """Xuất Mật lệnh (Signals) ra file JSON để live.py và sniper.py đọc"""
        out_path = self.data_dir.parent / 'live/darkpool_signals.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {}
        # Ép t0_date thành chuỗi chuẩn để làm Hạn sử dụng
        valid_date_str = self.t0_date.strftime('%Y-%m-%d')

        for an in self.anomalies:
            ticker = an['ticker']
            # Lấy câu text dự báo tương ứng với mã này
            forecast_text = next((f for f in self.forecasts if f"[{ticker}]" in f), "")
            
            # Gắn nhãn Hành động (Action) để Cỗ máy khác dễ hiểu
            action = "NONE"
            if "BẮT ĐÁY" in forecast_text or "GOM MẠNH" in forecast_text or "TẦM NGẮM ĐẶC BIỆT" in forecast_text:
                action = "BUY_TARGET"
            elif "BÁN/ĐỨNG NGOÀI" in forecast_text or "TẮM MÁU" in forecast_text or "CÁ MẬP KẸP HÀNG" in forecast_text:
                action = "DANGER"

            export_data[ticker] = {
                "basket": an['basket'],
                "val_bn": an['val_bn'],
                "t0_val_bn": an['t0_val_bn'],
                "ratio": an['ratio'],
                "intent": an['intent'],
                "forecast": forecast_text,
                "action": action,
                "valid_for_date": valid_date_str, # Gắn tem Hạn sử dụng
                "date_updated": datetime.now().strftime('%Y-%m-%d')
            }
            
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)
        print(f" [*] Đã phát tín hiệu (Broadcasted) {len(export_data)} mã sang Mạng lưới Bot.")

    def run_radar(self):
        """Thực thi Pipeline và In Báo Cáo"""
        if not self._prepare_data():
            print("[!] Thiếu dữ liệu lõi để chạy Dark Pool Radar.")
            return

        self._analyze_baskets()
        self._detect_anomalies()
        self._forecast_tactics()
        self.export_signals()

        today_str = datetime.now().strftime('%d/%m/%Y')
        print("\n" + "="*95)
        print(f" 🕵️ ĐÀI QUAN SÁT DÒNG TIỀN NGẦM (DARK POOL RADAR) | BÁO CÁO NGÀY: {today_str}")
        print("="*95)
        
        print("\n 📊 1. NHIỆT ĐỒ CÁC RỔ (TỔNG KẾT 5 PHIÊN GẦN NHẤT)")
        for basket in ['VN30', 'VNMidCap', 'VNSmallCap']:
            stats = self.basket_stats.get(basket)
            if stats:
                print(f"    - [{basket:<10}] : {stats['total_val']:>6,.0f} Tỷ | Trạng thái: {stats['status']}")
            else:
                print(f"    - [{basket:<10}] : Không có giao dịch đáng kể.")

        print("\n 🦈 2. TOP CÁ MẬP GIAO DỊCH BẤT THƯỜNG (ANOMALY DETECTED)")
        if self.anomalies:
            # Format lại bảng in cho chuyên nghiệp, bổ sung T0
            print(f"    {'Mã (Rổ)':<15} | {'Sang tay 5D':<15} | {'Riêng T0':<12} | {'Cường độ':<13} | {'Ý đồ Tay to':<20}")
            print("    " + "-"*85)
            for i, an in enumerate(self.anomalies, 1):
                ticker_basket = f"{i}. {an['ticker']} ({an['basket'][:4]})"
                val_5d = f"{an['val_bn']:>5.0f} Tỷ"
                val_t0 = f"{an['t0_val_bn']:>4.0f} Tỷ" if an['t0_val_bn'] > 0 else "  -  "
                ratio = f"{an['ratio']:.1f}x MA20"
                print(f"    {ticker_basket:<15} | {val_5d:<15} | {val_t0:<12} | {ratio:<13} | {an['intent']}")
        else:
            print("    - Không phát hiện hành vi Thỏa thuận đột biến nào thỏa mãn lưới lọc.")

        print("\n 🔮 3. DỰ BÁO CHIẾN THUẬT (T-DAY FORECAST)")
        if self.forecasts:
            for f in self.forecasts:
                print(f"    {f}")
        else:
            print("    - Chưa có Setup Thỏa thuận nào đủ mạnh để kích hoạt chiến thuật.")
            
        print("="*95 + "\n")

if __name__ == "__main__":
    radar = DarkPoolRadar()
    radar.run_radar()