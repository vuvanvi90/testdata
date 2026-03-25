import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*DataFrameGroupBy.apply.*")

class MarketTracker:
    def __init__(self, data_dir='data/parquet', verbose=True):
        """Khởi tạo Đài quan sát Vĩ mô & Dòng tiền"""
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Hệ thống Kiểm kê Thị trường (Market Tracker)...")
        
        self.price_path = f"{data_dir}/price/master_price.parquet"
        self.foreign_path = f"{data_dir}/macro/foreign_flow.parquet"
        self.prop_path = f"{data_dir}/macro/prop_flow.parquet"
        self.industry_path = f"{data_dir}/macro/groups_by_industries.parquet"
        self.verbose = verbose
        
        # Nạp dữ liệu vào RAM
        self.df_price = pd.read_parquet(self.price_path) if os.path.exists(self.price_path) else pd.DataFrame()
        self.df_foreign = pd.read_parquet(self.foreign_path) if os.path.exists(self.foreign_path) else pd.DataFrame()
        self.df_prop = pd.read_parquet(self.prop_path) if os.path.exists(self.prop_path) else pd.DataFrame()
        self.df_ind = pd.read_parquet(self.industry_path) if os.path.exists(self.industry_path) else pd.DataFrame()
        
        self._clean_data()

    def _clean_data(self):
        """Chuẩn hóa thời gian để Merge dữ liệu mượt mà"""
        if not self.df_price.empty:
            self.df_price['time'] = pd.to_datetime(self.df_price['time']).dt.normalize()
            self.df_price = self.df_price.sort_values(['ticker', 'time'])
            # Chỉ lấy các mã cơ sở (3 ký tự)
            self.df_price = self.df_price[self.df_price['ticker'].astype(str).str.len() == 3]
            
        for df in [self.df_foreign, self.df_prop]:
            if not df.empty and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()

    def _get_period_performance(self, lookback_days):
        """Tính toán Hiệu suất (Return) và Thanh khoản của toàn thị trường trong N ngày qua"""
        latest_date = self.df_price['time'].max()
        
        # Lấy danh sách các ngày giao dịch
        trading_days = sorted(self.df_price['time'].unique())
        if len(trading_days) < lookback_days + 1:
            return pd.DataFrame()
            
        start_date = trading_days[-(lookback_days + 1)]
        
        # Lọc dữ liệu trong khung thời gian
        df_period = self.df_price[(self.df_price['time'] >= start_date) & (self.df_price['time'] <= latest_date)]
        
        results = []
        for ticker, group in df_period.groupby('ticker'):
            if len(group) < 2: continue
            
            group = group.sort_values('time')
            start_price = group['close'].iloc[0]
            end_price = group['close'].iloc[-1]
            
            # Nếu giá đầu kỳ = 0 hoặc NaN thì bỏ qua
            if pd.isna(start_price) or start_price == 0: continue
                
            ret_pct = (end_price - start_price) / start_price * 100
            
            # Tính tổng giá trị giao dịch trong kỳ (Tỷ VNĐ)
            avg_price = group['close'].mean()
            total_vol = group['volume'].sum()
            total_val_bn = (avg_price * total_vol) / 1_000_000_000
            
            results.append({
                'ticker': ticker,
                'Return_%': ret_pct,
                'Total_Val_Bn': total_val_bn,
                'Close': end_price
            })
            
        return pd.DataFrame(results)

    def analyze_market_breadth(self, lookback_days=1, label="1 Ngày"):
        """1. ĐO LƯỜNG ĐỘ RỘNG THỊ TRƯỜNG (MARKET BREADTH)"""
        df_perf = self._get_period_performance(lookback_days)
        if df_perf.empty: return None
        
        # Loại bỏ các mã thanh khoản quá thấp (< 1 tỷ/kỳ) để tránh nhiễu
        df_perf = df_perf[df_perf['Total_Val_Bn'] >= 1.0]
        
        # Phân loại
        df_perf['Status'] = 'Tham chiếu'
        df_perf.loc[df_perf['Return_%'] > 0.5, 'Status'] = 'Tăng'  # Tăng > 0.5% mới tính
        df_perf.loc[df_perf['Return_%'] < -0.5, 'Status'] = 'Giảm' # Giảm > 0.5% mới tính
        
        summary = df_perf.groupby('Status').agg(
            Count=('ticker', 'count'),
            Value_Bn=('Total_Val_Bn', 'sum')
        ).reset_index()
        
        total_tickers = summary['Count'].sum()
        
        if self.verbose:
            print("\n" + "="*85)
            print(f" 📊 BÁO CÁO ĐỘ RỘNG THỊ TRƯỜNG (BREADTH) - KHUNG: {label}")
            print("="*85)
        for _, row in summary.iterrows():
            pct = (row['Count'] / total_tickers) * 100
            if self.verbose:
                print(f"  🔸 Số mã {row['Status']:<10}: {row['Count']:>4} mã ({pct:>5.1f}%) | Dòng tiền: {row['Value_Bn']:>8,.0f} Tỷ VNĐ")
            
        # CHẨN ĐOÁN RỦI RO (BULL TRAP / THẾ TRẬN TẤN CÔNG)
        advancers = summary[summary['Status'] == 'Tăng']
        decliners = summary[summary['Status'] == 'Giảm']
        if self.verbose:
            if not advancers.empty and not decliners.empty:
                adv_vol = advancers['Value_Bn'].values[0]
                dec_vol = decliners['Value_Bn'].values[0]
                adv_count = advancers['Count'].values[0]
                dec_count = decliners['Count'].values[0]
                print("-" * 85)
                if adv_count > dec_count and dec_vol > adv_vol * 1.2:
                    print("  🚨 CẢNH BÁO XANH VỎ ĐỎ LÒNG: Số mã tăng nhiều hơn nhưng Tiền lại tập trung bán tháo ở mã Giảm!")
                elif adv_count > total_tickers * 0.7 and adv_vol > dec_vol * 2:
                    print("  🚀 BÙNG NỔ THEO ĐÀ (FOLLOW-THROUGH): 70% thị trường đồng thuận tăng với thanh khoản áp đảo. TẤN CÔNG!")
                elif dec_count > total_tickers * 0.7:
                    print("  🩸 ÁP LỰC BÁN DIỆN RỘNG: Thị trường đang hoảng loạn diện rộng. Ưu tiên quản trị rủi ro.")
                else:
                    print("  ⚖️ TRẠNG THÁI CÂN BẰNG: Dòng tiền đang phân hóa, chọn lọc cổ phiếu kỹ lưỡng.")
            print("="*85)
        return df_perf

    def analyze_sector_rotation(self, df_perf, top_n=5):
        """2. TRUY TÌM DÒNG TIỀN LUÂN CHUYỂN NGÀNH (SECTOR ROTATION)"""
        if df_perf is None or self.df_ind.empty: return
        
        # Merge dữ liệu hiệu suất với Ngành cấp 2 (icb_name2) hoặc cấp 3
        sector_col = 'icb_name2' if 'icb_name2' in self.df_ind.columns else 'icb_name'
        df_merged = pd.merge(df_perf, self.df_ind[['symbol', sector_col]], left_on='ticker', right_on='symbol', how='inner')
        
        # Thống kê theo ngành
        sector_stats = df_merged.groupby(sector_col).agg(
            Avg_Return=('Return_%', 'mean'),
            Total_Val=('Total_Val_Bn', 'sum'),
            Stock_Count=('ticker', 'count'),
            Advancers=('Return_%', lambda x: (x > 0).sum())
        )
        
        # Lọc các ngành có ít nhất 5 mã để đảm bảo tính đại diện
        sector_stats = sector_stats[sector_stats['Stock_Count'] >= 5]
        sector_stats['Advance_Ratio_%'] = (sector_stats['Advancers'] / sector_stats['Stock_Count']) * 100
        
        # Xếp hạng Ngành Dẫn Dắt (Kết hợp giữa Tăng giá và Hút tiền)
        # Tạo điểm Score = Rank(Return) + Rank(Value)
        sector_stats['Score'] = sector_stats['Avg_Return'].rank() + sector_stats['Total_Val'].rank()
        leaders = sector_stats.sort_values('Score', ascending=False).head(top_n)
        if self.verbose:
            print("\n" + "="*85)
            print(f" 🏆 TOP {top_n} NGÀNH HÚT TIỀN & DẪN DẮT (SECTOR LEADERS)")
            print("="*85)
            print(f"{'TÊN NGÀNH':<30} | {'MỨC TĂNG TB':>12} | {'ĐỘ ĐỒNG THUẬN':>15} | {'DÒNG TIỀN (TỶ)':>15}")
            print("-" * 85)
            for index, row in leaders.iterrows():
                name = str(index)[:28]
                print(f"{name:<30} | {row['Avg_Return']:>11.2f}% | {row['Advance_Ratio_%']:>14.1f}% | {row['Total_Val']:>15,.0f}")
            print("="*85)
        
        return leaders.index.tolist()

    def analyze_flow_attribution(self, lookback_days=5):
        """3. GIẢI PHẪU DÒNG TIỀN TAY TO (FLOW ATTRIBUTION) TRÊN TOP 50 MÃ TĂNG"""
        df_perf = self._get_period_performance(lookback_days)
        if df_perf is None or df_perf.empty: return
        
        # Lấy Top 50 mã tăng mạnh nhất (Chỉ lấy mã thanh khoản > 5 tỷ để loại rác)
        df_top = df_perf[df_perf['Total_Val_Bn'] >= 5.0].nlargest(50, 'Return_%')
        top_tickers = df_top['ticker'].tolist()
        
        latest_date = self.df_price['time'].max()
        start_date = sorted(self.df_price['time'].unique())[-(lookback_days + 1)]
        
        # Lấy dữ liệu dòng tiền của Top 50 mã này
        f_net_total = 0
        p_net_total = 0
        
        if not self.df_foreign.empty:
            df_f = self.df_foreign[(self.df_foreign['time'] >= start_date) & (self.df_foreign['ticker'].isin(top_tickers))]
            f_net_total = df_f['foreign_net_value'].sum() / 1_000_000_000 # Đổi ra Tỷ VNĐ
            
        if not self.df_prop.empty:
            df_p = self.df_prop[(self.df_prop['time'] >= start_date) & (self.df_prop['ticker'].isin(top_tickers))]
            if 'prop_net_value' in df_p.columns:
                p_net_total = df_p['prop_net_value'].sum() / 1_000_000_000
                
        total_top_val = df_top['Total_Val_Bn'].sum()
        
        if self.verbose:
            print("\n" + "="*85)
            print(f" 🕵️ GIẢI PHẪU DÒNG TIỀN TRÊN TOP 50 MÃ TĂNG MẠNH NHẤT ({lookback_days} NGÀY)")
            print("="*85)
            print(f"  + Tổng Thanh khoản Top 50 : {total_top_val:,.0f} Tỷ VNĐ")
            print(f"  + Khối Ngoại Đóng góp     : {f_net_total:>+8.1f} Tỷ VNĐ")
            print(f"  + Tự Doanh Đóng góp       : {p_net_total:>+8.1f} Tỷ VNĐ")
            print("-" * 85)
        
        # Chẩn đoán bản chất sóng
        sm_net = f_net_total + p_net_total
        if self.verbose:
            if sm_net < 0:
                print("  🩸 BẢN CHẤT SÓNG: KÉO XẢ! Top 50 mã tăng mạnh nhất đang bị Tổ chức XẢ RÒNG.")
                print("  => Động lực kéo giá 100% đến từ Dòng tiền Ẩn (Lái nội) và FOMO của Nhỏ lẻ.")
            elif sm_net > total_top_val * 0.05: # Tay to chiếm > 5% thanh khoản
                print("  ✅ BẢN CHẤT SÓNG: UY TÍN! Sóng tăng được bảo kê bởi Dòng tiền Tổ chức (Gom ròng mạnh).")
            else:
                print("  ⚠️ BẢN CHẤT SÓNG: ĐẦU CƠ! Tay to đứng ngoài quan sát, thị trường tự chơi với nhau.")
            print("="*85)

    def analyze_full_intraday_macro(self, intraday_df):
        """
        CHẾ ĐỘ TOÀN TRI (OMNISCIENT MODE) - XỬ LÝ ORDER FLOW TOÀN THỊ TRƯỜNG
        Đánh giá X-Quang toàn bộ 1,600 mã để tìm ra Bản chất Dòng tiền ngay lúc 14h15.
        """
        if intraday_df is None or intraday_df.empty:
            print("[!] Thiếu dữ liệu Intraday trong master_intraday.parquet.")
            return None
            
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp và giải phẫu Order Flow Toàn Thị Trường...")
            df = intraday_df
            
            col_type = 'match_type' 
            col_vol = 'volume'
            col_price = 'price'
            
            if col_type not in df.columns:
                print(f"[!] Dữ liệu Intraday thiếu cột phân loại lệnh ({col_type}). Cột hiện có: {df.columns.tolist()}")
                return None
                
            # Tính Giá trị mỗi lệnh (Tỷ VNĐ)
            df['trade_val_bn'] = (df[col_price] * df[col_vol]) / 1_000_000_000
            
            # 🚀 Phân loại Mua/Bán chủ động bằng Vector (Bao phủ cả chữ 'Buy'/'Sell' và các mã viết tắt cũ)
            is_bu = df[col_type].isin(['Buy', 'BU', 'B'])
            is_sd = df[col_type].isin(['Sell', 'SD', 'S'])
            
            # 1. THỐNG KÊ VĨ MÔ TOÀN THỊ TRƯỜNG (MARKET-WIDE FLOW)
            total_bu_bn = df.loc[is_bu, 'trade_val_bn'].sum()
            total_sd_bn = df.loc[is_sd, 'trade_val_bn'].sum()
            market_net_active = total_bu_bn - total_sd_bn
            
            # Thống kê Cá Mập toàn thị trường (Lệnh > 1 Tỷ VNĐ)
            is_shark = df['trade_val_bn'] >= 1.0
            shark_bu_bn = df.loc[is_bu & is_shark, 'trade_val_bn'].sum()
            shark_sd_bn = df.loc[is_sd & is_shark, 'trade_val_bn'].sum()
            shark_net_active = shark_bu_bn - shark_sd_bn
            
            if self.verbose:
                print("\n" + "="*90)
                print(f" 🌐 BÁO CÁO ORDER FLOW TOÀN THỊ TRƯỜNG - {datetime.now().strftime('%H:%M')}")
                print("="*90)
                print(f" 1. TỔNG QUAN DÒNG TIỀN CHỦ ĐỘNG (RETAIL + INSTITUTION):")
                print(f"    🔸 Tổng MUA Chủ Động : {total_bu_bn:>8,.1f} Tỷ VNĐ")
                print(f"    🔸 Tổng BÁN Chủ Động : {total_sd_bn:>8,.1f} Tỷ VNĐ")
                print(f"    => CHÊNH LỆCH RÒNG   : {market_net_active:>+8,.1f} Tỷ VNĐ")
                print(f"\n 2. DẤU CHÂN CÁ MẬP (CÁC LỆNH KHỚP > 1 TỶ VNĐ):")
                print(f"    🔸 Cá Mập MUA C.Động : {shark_bu_bn:>8,.1f} Tỷ VNĐ")
                print(f"    🔸 Cá Mập BÁN C.Động : {shark_sd_bn:>8,.1f} Tỷ VNĐ")
                print(f"    => CÁ MẬP ĐANG       : {'🟢 GOM RÒNG' if shark_net_active > 0 else '🔴 XẢ RÒNG'} ({shark_net_active:>+8,.1f} Tỷ VNĐ)")
                print("-" * 90)
            
            # CHẨN ĐOÁN VĨ MÔ (MACRO VERDICT)
            if market_net_active > 0 and shark_net_active > 0:
                if self.verbose:
                    print("  🟢 ĐÈN XANH (FULL ATTACK): Tiền vào dứt khoát, Cá Mập dẫn đường. Tự tin giải ngân!")
                market_status = "GREEN"
            elif market_net_active < 0 and shark_net_active < 0:
                if self.verbose:
                    print("  🔴 ĐÈN ĐỎ (PANIC / DISTRIBUTION): Cá mập xả hàng, thị trường bán đuổi. ĐÓNG BĂNG MUA MỚI!")
                market_status = "RED"
            elif market_net_active > 0 and shark_net_active < 0:
                if self.verbose:
                    print("  🟡 ĐÈN VÀNG (BULL TRAP): Nhỏ lẻ hưng phấn Mua chủ động, nhưng Cá Mập đang âm thầm XẢ RÒNG. Rất rủi ro!")
                market_status = "YELLOW"
            else:
                if self.verbose:
                    print("  ⚪ TRẠNG THÁI CÂN BẰNG: Dòng tiền đang giằng co.")
                market_status = "NEUTRAL"
            print("="*90)
            
            if self.verbose:
                print(f"[*] Đang đóng gói dữ liệu Order Flow cho từng mã cổ phiếu...")
            
            # Hàm phụ trợ để tính an toàn cho từng group
            def calculate_ticker_flow(x):
                bu_val = x.loc[x[col_type].isin(['Buy', 'BU', 'B']), 'trade_val_bn'].sum()
                sd_val = x.loc[x[col_type].isin(['Sell', 'SD', 'S']), 'trade_val_bn'].sum()
                total_v = x[col_vol].sum()
                vwap_val = (x[col_price] * x[col_vol]).sum() / total_v if total_v > 0 else 0
                return pd.Series({
                    'bu_bn': bu_val,
                    'sd_bn': sd_val,
                    'vwap': vwap_val,
                    'last_price': x[col_price].iloc[-1] if not x.empty else 0
                })

            ticker_stats = df.groupby('ticker').apply(calculate_ticker_flow).reset_index()
            ticker_stats['net_active_bn'] = ticker_stats['bu_bn'] - ticker_stats['sd_bn']
            
            # Chuyển thành Dictionary để live.py tra cứu siêu tốc O(1)
            intraday_dict = ticker_stats.set_index('ticker').to_dict('index')
            
            return {
                'market_status': market_status,
                'market_net_active': market_net_active,
                'intraday_dict': intraday_dict
            }
                
        except Exception as e:
            print(f"[!] Lỗi phân tích Full Intraday Macro: {e}")
            return None

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    tracker = MarketTracker(data_dir='data/parquet')
    
    # 1. Quét Độ rộng thị trường (Đếm mã Tăng/Giảm) trong 1 Tuần qua (5 phiên)
    df_perf_1w = tracker.analyze_market_breadth(lookback_days=5, label="1 TUẦN QUA")
    
    # 2. Tìm Ngành Dẫn Dắt
    leading_sectors = tracker.analyze_sector_rotation(df_perf_1w, top_n=5)
    
    # 3. Giải phẫu xem ai đang đẩy Top 50 mã mạnh nhất
    tracker.analyze_flow_attribution(lookback_days=5)

    