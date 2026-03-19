import pandas as pd
import numpy as np
import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*DataFrameGroupBy.apply.*")

class SmartMoneyTracker:
    def __init__(self, df_price, df_foreign, df_prop):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp Sổ cái Dòng tiền & Lịch sử OHLCV...")
        self.df_price = df_price
        self.df_foreign = df_foreign
        self.df_prop = df_prop
        
        # Chuẩn hóa thời gian
        for df in [self.df_price, self.df_foreign, self.df_prop]:
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()

    def track_ticker(self, ticker, target_date=None, start_date=None):
        """
        X-Quang Dòng tiền: Bóc tách Tồn kho Tay to và Dòng tiền Ẩn (Shadow Flow)
        """
        print("\n" + "="*115)
        print(f" 🔍 X-QUANG DÒNG TIỀN ĐA CHIỀU (OHLCV + SMART MONEY): Ticker [ {ticker} ]")
        print("="*115)

        if self.df_price.empty:
            print("[!] Thiếu dữ liệu OHLCV trong master_price.parquet.")
            return

        # 1. Lọc dữ liệu theo Mã cổ phiếu
        df_price_t = self.df_price[self.df_price['ticker'] == ticker].copy()
        df_f_t = self.df_foreign[self.df_foreign['ticker'] == ticker].copy() if not self.df_foreign.empty else pd.DataFrame()
        df_p_t = self.df_prop[self.df_prop['ticker'] == ticker].copy() if not self.df_prop.empty else pd.DataFrame()

        if df_price_t.empty:
            print(f"[!] Không tìm thấy lịch sử giá của mã {ticker}.")
            return

        # 2. Tính Tổng Giá Trị Giao Dịch Của Thị Trường (Đơn vị: Tỷ VNĐ)
        DIVISOR = 1_000_000_000
        df_price_t['market_val_bn'] = (df_price_t['close'] * df_price_t['volume']) / DIVISOR

        # 3. Hợp nhất Dữ liệu bằng Left Join (Lấy OHLCV làm gốc)
        df_merged = pd.merge(df_price_t[['time', 'close', 'market_val_bn']], df_f_t, on='time', how='left')
        df_merged = pd.merge(df_merged, df_p_t, on='time', how='left').fillna(0)
        df_merged = df_merged.sort_values('time').reset_index(drop=True)

        # Lọc theo Khoảng thời gian (Cỗ máy thời gian)
        if start_date:
            df_merged = df_merged[df_merged['time'] >= pd.to_datetime(start_date)]
        if target_date:
            df_merged = df_merged[df_merged['time'] <= pd.to_datetime(target_date)]

        if df_merged.empty:
            print(f"[*] Không có giao dịch nào trong khoảng thời gian được chọn.")
            return

        # 4. TÍNH TOÁN BÓC TÁCH DÒNG TIỀN (SHADOW FLOW & DOMINANCE)
        # Giá trị Mua/Bán Ròng (Để tính Tồn kho)
        df_merged['f_net_bn'] = df_merged['foreign_net_value'] / DIVISOR
        df_merged['p_net_bn'] = df_merged['prop_net_value'] / DIVISOR
        df_merged['total_net_bn'] = df_merged['f_net_bn'] + df_merged['p_net_bn']

        # Tính Lũy kế (Cumulative Sum)
        df_merged['cum_total'] = df_merged['total_net_bn'].cumsum()

        # Tính Tỷ lệ Chi phối (Smart Money Dominance)
        # Tổng giá trị mua + bán của Tay To chia đôi (Để ra giá trị giao dịch 1 chiều tương đương với OHLCV)
        df_merged['f_gross_bn'] = (df_merged['foreign_buy_value'] + df_merged['foreign_sell_value']) / DIVISOR
        df_merged['p_gross_bn'] = (df_merged['prop_buy_value'] + df_merged['prop_sell_value']) / DIVISOR
        df_merged['sm_participation_bn'] = (df_merged['f_gross_bn'] + df_merged['p_gross_bn']) / 2

        # Dòng tiền Ẩn (Nhỏ lẻ & Đội lái nội) = Tổng TT - Khối lượng Tay to tham gia
        df_merged['shadow_flow_bn'] = df_merged['market_val_bn'] - df_merged['sm_participation_bn']
        df_merged['shadow_flow_bn'] = df_merged['shadow_flow_bn'].clip(lower=0) # Chống số âm do sai số API

        # Tỷ lệ chi phối (%)
        df_merged['sm_dominance_pct'] = np.where(
            df_merged['market_val_bn'] > 0,
            (df_merged['sm_participation_bn'] / df_merged['market_val_bn']) * 100,
            0
        )

        # 5. IN KẾT QUẢ RA TERMINAL
        print(f"{'NGÀY':<12} | {'GIÁ ĐÓNG':>10} | {'TỔNG TT (TỶ)':>14} | {'TAY TO NET (TỶ)':>17} | {'ẨN/LÁI NỘI (TỶ)':>17} | {'CHI PHỐI':>10} | {'LŨY KẾ (TỶ)':>13}")
        print("-" * 115)

        shadow_spikes = 0 # Đếm số phiên Dòng tiền ẩn bùng nổ

        for _, row in df_merged.iterrows():
            date_str = row['time'].strftime('%d-%m-%Y')
            close_p = row['close']
            mkt_val = row['market_val_bn']
            net_val = row['total_net_bn']
            shadow_val = row['shadow_flow_bn']
            dom_pct = row['sm_dominance_pct']
            cum_val = row['cum_total']

            # Đánh dấu các phiên Lái Nội / Nhỏ lẻ tự chơi (Chi phối < 5% nhưng Vol to)
            flag = ""
            if dom_pct < 5.0 and mkt_val > df_merged['market_val_bn'].mean() * 1.5:
                flag = " ⚠️ (Lái Nội)"
                shadow_spikes += 1
            elif dom_pct > 30.0:
                flag = " 🎯 (Tay To)"

            print(f"{date_str:<12} | {close_p:>10,.0f} | {mkt_val:>14.1f} | {net_val:>17.2f} | {shadow_val:>17.1f} | {dom_pct:>8.1f}% | {cum_val:>13.2f}{flag}")

        # 6. IN TỔNG KẾT (SUMMARY) TẠI NGÀY ĐÍCH
        final_row = df_merged.iloc[-1]
        start_date_str = df_merged.iloc[0]['time'].strftime('%d/%m/%Y')
        end_date_str = final_row['time'].strftime('%d/%m/%Y')
        
        avg_dominance = df_merged['sm_dominance_pct'].mean()

        print("\n" + "="*115)
        print(f" 🎯 TỔNG KẾT BÓC TÁCH DÒNG TIỀN ĐỐI VỚI [ {ticker} ]")
        print(f"    Giai đoạn: {start_date_str} -> {end_date_str} ({len(df_merged)} phiên giao dịch)")
        print("="*115)
        print(f" 🔹 TỔNG LƯỢNG TỒN KHO TAY TO : {final_row['cum_total']:>10.2f} Tỷ VNĐ")
        print(f" 🔹 Tỷ lệ Chi phối Trung bình   : {avg_dominance:>10.1f} % (Quyền lực của Smart Money)")
        print(f" 🔹 Số phiên 'Lái Nội' bùng nổ  : {shadow_spikes:>10} phiên")
        print("-" * 115)
        
        # CHẨN ĐOÁN HÀNH VI
        if shadow_spikes >= 3 and final_row['cum_total'] <= 0:
            print(" 🚨 KẾT LUẬN: Sóng Đầu Cơ Lái Nội! Tay to (Tây/Tự doanh) đứng ngoài hoặc đang xả hàng.")
            print("    => Rủi ro đu đỉnh cực cao nếu mua đuổi. Cẩn thận bẫy thanh khoản!")
        elif final_row['cum_total'] > 0 and avg_dominance > 15.0:
            print(" 🌟 KẾT LUẬN: Sóng Tổ Chức! Smart Money đang nắm quyền chi phối và thu gom hàng.")
            print("    => Bệ phóng an toàn, có thể tự tin đi lệnh lớn nếu nổ tín hiệu Wyckoff.")
        else:
            print(" ⚖️ KẾT LUẬN: Dòng tiền giằng co, chưa rõ phe nào kiểm soát hoàn toàn.")
        print("="*115)

        return df_merged

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    PRICE_PATH = 'data/parquet/price/master_price.parquet'
    FOREIGN_PATH = 'data/parquet/macro/foreign_flow.parquet'
    PROP_PATH = 'data/parquet/macro/prop_flow.parquet'
    
    tracker = SmartMoneyTracker(PRICE_PATH, FOREIGN_PATH, PROP_PATH)
    
    # Test: Bóc tách dòng tiền của 1 mã bất kỳ (Anh điền mã anh muốn soi vào đây)
    tracker.track_ticker(ticker='HPG')