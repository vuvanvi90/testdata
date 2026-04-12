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
                if pd.api.types.is_numeric_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], unit='ms').dt.normalize()
                else:
                    df['time'] = pd.to_datetime(df['time']).dt.normalize()

                df.sort_values(['ticker', 'time'])

        self.price_dict = dict(tuple(self.df_price.groupby('ticker'))) if not self.df_price.empty else {}
        self.foreign_dict = dict(tuple(self.df_foreign.groupby('ticker'))) if not self.df_foreign.empty else {}
        self.prop_dict = dict(tuple(self.df_prop.groupby('ticker'))) if not self.df_prop.empty else {}

    def track_ticker(self, ticker, target_date=None, start_date=None):
        """
        X-Quang Dòng tiền: Bóc tách Tồn kho Tay to, VWAP và Dòng tiền Ẩn
        """
        print("\n" + "="*135)
        print(f" 🔍 X-QUANG DÒNG TIỀN ĐA CHIỀU (OHLCV + SMART MONEY + VWAP): Ticker [ {ticker} ]")
        print("="*135)

        # Trích xuất O(1) từ Dictionary
        df_price_t = self.price_dict.get(ticker)
        df_f_t = self.foreign_dict.get(ticker, pd.DataFrame(columns=['time', 'foreign_net_value', 'foreign_buy_value', 'foreign_sell_value']))
        df_p_t = self.prop_dict.get(ticker, pd.DataFrame(columns=['time', 'prop_net_value', 'prop_buy_value', 'prop_sell_value']))

        if df_price_t is None or df_price_t.empty:
            print(f"[!] Không tìm thấy lịch sử giá của mã {ticker}.")
            return

        DIVISOR = 1_000_000_000
        df_price_t = df_price_t.copy()
        df_price_t['market_val_bn'] = (df_price_t['close'] * df_price_t['volume']) / DIVISOR

        # Hợp nhất Dữ liệu
        df_merged = pd.merge(df_price_t[['time', 'close', 'market_val_bn']], df_f_t, on='time', how='left')
        df_merged = pd.merge(df_merged, df_p_t, on='time', how='left').fillna(0)
        df_merged = df_merged.sort_values('time').reset_index(drop=True)

        if start_date: df_merged = df_merged[df_merged['time'] >= pd.to_datetime(start_date)]
        if target_date: df_merged = df_merged[df_merged['time'] <= pd.to_datetime(target_date)]

        if df_merged.empty:
            print(f"[*] Không có giao dịch nào trong khoảng thời gian được chọn.")
            return

        # 🚀 TÍNH TOÁN DÒNG TIỀN ẨN (SHADOW FLOW)
        df_merged['f_net_bn'] = df_merged['foreign_net_value'] / DIVISOR
        df_merged['p_net_bn'] = df_merged['prop_net_value'] / DIVISOR
        df_merged['total_net_bn'] = df_merged['f_net_bn'] + df_merged['p_net_bn']

        df_merged['f_gross_bn'] = (df_merged['foreign_buy_value'] + df_merged['foreign_sell_value']) / DIVISOR
        df_merged['p_gross_bn'] = (df_merged['prop_buy_value'] + df_merged['prop_sell_value']) / DIVISOR
        df_merged['sm_participation_bn'] = (df_merged['f_gross_bn'] + df_merged['p_gross_bn']) / 2

        df_merged['shadow_flow_bn'] = (df_merged['market_val_bn'] - df_merged['sm_participation_bn']).clip(lower=0)

        df_merged['sm_dominance_pct'] = np.where(
            df_merged['market_val_bn'] > 0,
            (df_merged['sm_participation_bn'] / df_merged['market_val_bn']) * 100, 0
        )

        # 🚀 TÍNH TOÁN LŨY KẾ & GIÁ VỐN LIÊN HOÀN (DYNAMIC VWAP)
        inventory_bn = 0
        current_vwap = 0.0
        
        for idx, row in df_merged.iterrows():
            net_val = row['total_net_bn']
            close_p = row['close']
            
            # Cập nhật tồn kho (Tiền)
            inventory_bn += net_val
            
            # Tính VWAP
            if net_val > 0:
                # Mua vào -> Tính lại trung bình giá
                # (Lưu ý: Để đơn giản hóa trong X-Quang, ta dùng Tổng Giá trị mua chia đều. 
                # Cách chuẩn xác nhất là quy đổi ra Volume cổ phiếu, nhưng dùng Tiền vẫn ra xu hướng đúng)
                if inventory_bn > 0:
                    old_weight = (inventory_bn - net_val) / inventory_bn
                    new_weight = net_val / inventory_bn
                    current_vwap = (current_vwap * old_weight) + (close_p * new_weight)
                else:
                    current_vwap = close_p
            elif net_val < 0:
                # Bán ra -> Giữ nguyên VWAP, chỉ giảm Inventory
                if inventory_bn <= 0:
                    inventory_bn = 0
                    current_vwap = 0.0
                    
            df_merged.at[idx, 'cum_total'] = inventory_bn
            df_merged.at[idx, 'dynamic_vwap'] = current_vwap

        # IN KẾT QUẢ
        print(f"{'NGÀY':<12} | {'GIÁ ĐÓNG':>10} | {'TỔNG TT (TỶ)':>14} | {'TAY TO NET (TỶ)':>17} | {'ẨN/LÁI (TỶ)':>15} | {'CHI PHỐI':>10} | {'LŨY KẾ (TỶ)':>13} | {'VWAP LÁI':>10}")
        print("-" * 135)

        shadow_spikes = 0 

        for _, row in df_merged.iterrows():
            date_str = row['time'].strftime('%d-%m-%Y')
            flag = ""
            if row['sm_dominance_pct'] < 5.0 and row['market_val_bn'] > df_merged['market_val_bn'].mean() * 1.5:
                flag = " ⚠️(Lái Nội)"
                shadow_spikes += 1
            elif row['sm_dominance_pct'] > 30.0:
                flag = " 🎯(Tay To)"

            print(f"{date_str:<12} | {row['close']:>10,.0f} | {row['market_val_bn']:>14.1f} | {row['total_net_bn']:>17.2f} | {row['shadow_flow_bn']:>15.1f} | {row['sm_dominance_pct']:>8.1f}% | {row['cum_total']:>13.2f} | {row['dynamic_vwap']:>10,.0f}{flag}")

        # TỔNG KẾT
        final_row = df_merged.iloc[-1]
        avg_dominance = df_merged['sm_dominance_pct'].mean()
        
        # Đánh giá xem Giá Hiện Tại đang Cao hay Thấp hơn Giá Vốn Tay To
        vwap = final_row['dynamic_vwap']
        current_price = final_row['close']
        vwap_status = ""
        if vwap > 0:
            diff_pct = (current_price - vwap) / vwap * 100
            vwap_status = f"(Tay to đang {'Lãi' if diff_pct > 0 else 'Lỗ'} {abs(diff_pct):.1f}%)"

        print("\n" + "="*135)
        print(f" 🎯 TỔNG KẾT BÓC TÁCH DÒNG TIỀN ĐỐI VỚI [ {ticker} ]")
        print(f"    Giai đoạn: {df_merged.iloc[0]['time'].strftime('%d/%m/%Y')} -> {final_row['time'].strftime('%d/%m/%Y')} ({len(df_merged)} phiên)")
        print("="*135)
        print(f" 🔹 TỔNG LƯỢNG TỒN KHO TAY TO : {final_row['cum_total']:>10.2f} Tỷ VNĐ")
        print(f" 🔹 GIÁ VỐN TRUNG BÌNH (VWAP) : {vwap:>10,.0f} đ {vwap_status}")
        print(f" 🔹 Tỷ lệ Chi phối Trung bình   : {avg_dominance:>10.1f}% (Quyền lực của Smart Money)")
        print(f" 🔹 Số phiên 'Lái Nội' bùng nổ  : {shadow_spikes:>10} phiên")
        print("-" * 135)
        
        if shadow_spikes >= 3 and final_row['cum_total'] <= 0:
            print(" 🚨 KẾT LUẬN: Sóng Đầu Cơ Lái Nội! Tay to (Tây/Tự doanh) đứng ngoài hoặc đang xả hàng.")
        elif final_row['cum_total'] > 0 and avg_dominance > 15.0:
            print(" 🌟 KẾT LUẬN: Sóng Tổ Chức! Smart Money đang nắm quyền chi phối và thu gom hàng.")
            if current_price < vwap:
                print("    => CƠ HỘI VÀNG: Giá hiện tại đang RẺ HƠN giá vốn của Tay to!")
        else:
            print(" ⚖️ KẾT LUẬN: Dòng tiền giằng co, chưa rõ phe nào kiểm soát hoàn toàn.")
        print("="*135)

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