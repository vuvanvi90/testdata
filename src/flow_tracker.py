import pandas as pd
import numpy as np
import os
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*DataFrameGroupBy.apply.*")

class SmartMoneyTracker:
    def __init__(self, df_price, df_foreign, df_prop, df_indx):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp Sổ cái Dòng tiền & Lịch sử OHLCV...")
        self.df_price = df_price
        self.df_foreign = df_foreign
        self.df_prop = df_prop
        self.df_indx = df_indx
        
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

        self.ticker_universe_map = {}
        try:
            if not self.df_indx.empty:
                for _, row in self.df_indx.iterrows():
                    ticker = row['ticker']
                    idx_code = row['index_code']
                    # Ưu tiên ghi đè: VN30 -> VNMidCap -> VNSmallCap
                    if ticker not in self.ticker_universe_map and idx_code in ['VN30', 'VNMidCap', 'VNSmallCap']:
                        self.ticker_universe_map[ticker] = idx_code
        except Exception as e:
            print(f"[!] Lỗi nạp Index Components: {e}")

    def _get_dynamic_thresholds(self, universe):
        """Thiết lập Ngưỡng phân tích dựa trên đặc thù của từng Rổ"""
        if universe == 'VN30':
            return {
                'dom_high': 20.0,      # Chi phối > 20% mới gọi là Tay To kiểm soát
                'dom_low': 5.0,        # Dưới 5% là vắng mặt
                'shadow_spike': 1.5,   # Lái nội nổ vol = 1.5x trung bình
            }
        elif universe == 'VNMidCap':
            return {
                'dom_high': 15.0,      # Midcap chỉ cần > 15% là uy tín
                'dom_low': 3.0,
                'shadow_spike': 2.0,   # Midcap hay bị lái nội thổi Vol x2
            }
        elif universe == 'VNSmallCap':
            return {
                'dom_high': 5.0,       # Penny mà Tây/TD cầm > 5% là CỰC KỲ BẤT THƯỜNG (Có biến)
                'dom_low': 1.0,
                'shadow_spike': 3.0,   # Penny Lái nội toàn thổi Vol x3 x4
            }
        else: # Default (HOSE)
            return {'dom_high': 15.0, 'dom_low': 3.0, 'shadow_spike': 2.0}

    def track_ticker(self, ticker, target_date=None, start_date=None):
        """
        X-Quang Dòng tiền: Bóc tách Tồn kho Tay to, VWAP và Dòng tiền Ẩn (ĐÃ TÍCH HỢP LỌC SANG TAY KÉP)
        """
        universe = self.ticker_universe_map.get(ticker, 'HOSE (Unknown)')
        thresh = self._get_dynamic_thresholds(universe)

        print("\n" + "="*145)
        print(f" 🔍 DÒNG TIỀN (OHLCV + SMART MONEY + VWAP): Mã [ {ticker} ] - Thuộc rổ: {universe}")
        print("="*145)

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

        # 🚀 LẤY THÊM OPEN, HIGH, LOW, VOLUME ĐỂ PHỤC VỤ BỘ LỌC SANG TAY
        df_merged = pd.merge(df_price_t[['time', 'open', 'high', 'low', 'close', 'volume', 'market_val_bn']], df_f_t, on='time', how='left').fillna(0)
        df_merged = pd.merge(df_merged, df_p_t, on='time', how='left').fillna(0)
        df_merged = df_merged.sort_values('time').reset_index(drop=True)

        if start_date: df_merged = df_merged[df_merged['time'] >= pd.to_datetime(start_date)]
        if target_date: df_merged = df_merged[df_merged['time'] <= pd.to_datetime(target_date)]

        if df_merged.empty:
            print(f"[*] Không có giao dịch nào trong khoảng thời gian được chọn.")
            return

        # 🚀 TÍNH TOÁN CÁC CHỈ SỐ THÔ
        df_merged['f_net_bn'] = df_merged['foreign_net_value'] / DIVISOR
        df_merged['p_net_bn'] = df_merged['prop_net_value'] / DIVISOR
        
        df_merged['f_gross_bn'] = (df_merged['foreign_buy_value'] + df_merged['foreign_sell_value']) / DIVISOR
        df_merged['p_gross_bn'] = (df_merged['prop_buy_value'] + df_merged['prop_sell_value']) / DIVISOR

        # 🚀 BỘ LỌC SANG TAY KÉP (DUAL PUT-THROUGH FILTER)
        df_merged['vol_ma20'] = df_merged['volume'].rolling(20, min_periods=1).mean()
        df_merged['price_spread_pct'] = (df_merged['high'] - df_merged['low']) / df_merged['low'] * 100
        
        cond_on_book = (df_merged['volume'] > df_merged['vol_ma20'] * 3) & (df_merged['price_spread_pct'] < 2.0)
        cond_off_book_f = df_merged['f_net_bn'].abs() > (df_merged['market_val_bn'] * 0.8)
        cond_off_book_p = df_merged['p_net_bn'].abs() > (df_merged['market_val_bn'] * 0.8)
        
        df_merged['is_pt_f'] = np.where(cond_on_book | cond_off_book_f, True, False)
        df_merged['is_pt_p'] = np.where(cond_on_book | cond_off_book_p, True, False)
        
        # 🚀 LÀM SẠCH DỮ LIỆU ĐỘC LẬP
        df_merged['f_net_adj'] = np.where(df_merged['is_pt_f'], 0, df_merged['f_net_bn'])
        df_merged['p_net_adj'] = np.where(df_merged['is_pt_p'], 0, df_merged['p_net_bn'])
        df_merged['total_net_adj'] = df_merged['f_net_adj'] + df_merged['p_net_adj']

        df_merged['f_gross_adj'] = np.where(df_merged['is_pt_f'], 0, df_merged['f_gross_bn'])
        df_merged['p_gross_adj'] = np.where(df_merged['is_pt_p'], 0, df_merged['p_gross_bn'])

        # 🚀 TÍNH TOÁN DÒNG TIỀN ẨN TRÊN DỮ LIỆU SẠCH
        df_merged['sm_participation_bn'] = (df_merged['f_gross_adj'] + df_merged['p_gross_adj']) / 2
        df_merged['shadow_flow_bn'] = (df_merged['market_val_bn'] - df_merged['f_net_adj'].abs() - df_merged['p_net_adj'].abs()).clip(lower=0)

        df_merged['sm_dominance_pct'] = np.where(
            df_merged['market_val_bn'] > 0,
            (df_merged['sm_participation_bn'] / df_merged['market_val_bn']) * 100, 0
        )

        # 🚀 TÍNH TOÁN LŨY KẾ & GIÁ VỐN LIÊN HOÀN (DYNAMIC VWAP)
        inventory_bn, f_inv_bn, p_inv_bn = 0, 0, 0
        current_vwap = 0.0
        
        for idx, row in df_merged.iterrows():
            # SỬ DỤNG DỮ LIỆU ĐÃ LÀM SẠCH (ADJUSTED) ĐỂ TÍNH VWAP
            net_val = row['total_net_adj']
            f_net_val = row['f_net_adj']
            p_net_val = row['p_net_adj']
            close_p = row['close']
            
            # Cập nhật tồn kho (Tiền)
            inventory_bn += net_val
            f_inv_bn += f_net_val
            p_inv_bn += p_net_val
            
            # Tính VWAP
            if net_val > 0:
                if inventory_bn > 0:
                    old_weight = (inventory_bn - net_val) / inventory_bn
                    new_weight = net_val / inventory_bn
                    current_vwap = (current_vwap * old_weight) + (close_p * new_weight)
                else:
                    current_vwap = close_p
            elif net_val < 0:
                if inventory_bn <= 0:
                    inventory_bn = 0
                    current_vwap = 0.0
                if f_inv_bn <= 0: f_inv_bn = 0
                if p_inv_bn <= 0: p_inv_bn = 0
                    
            df_merged.at[idx, 'cum_total'] = inventory_bn
            df_merged.at[idx, 'cum_f_total'] = f_inv_bn
            df_merged.at[idx, 'cum_p_total'] = p_inv_bn
            df_merged.at[idx, 'dynamic_vwap'] = current_vwap

        # IN KẾT QUẢ
        print(f"{'NGÀY':<10} | {'GIÁ ĐÓNG':>10} | {'TỔNG TT (TỶ)':>12} | {'NGOẠI (TỶ)':>10} | {'NỘI (TỶ)':>10} | {'ẨN/LÁI (TỶ)':>11} | {'CHI PHỐI':>8} | {'LŨY KẾ (TỶ)':>11} | {'VWAP LÁI':>10}")
        print("-" * 145)

        shadow_spikes = 0 
        avg_market_val = df_merged['market_val_bn'].mean()

        for _, row in df_merged.iterrows():
            date_str = row['time'].strftime('%d-%m-%Y')
            flag = ""
            
            # 🚀 GẮN CỜ SANG TAY NẾU CÓ
            if row['is_pt_f'] and row['is_pt_p']:
                flag += " ⚠️[SANG TAY NGOẠI & NỘI]"
            elif row['is_pt_f']:
                flag += " ⚠️[SANG TAY NGOẠI]"
            elif row['is_pt_p']:
                flag += " ⚠️[SANG TAY NỘI]"
            else:
                # Nếu không sang tay, xét tiếp cờ chi phối bình thường
                if row['sm_dominance_pct'] < thresh['dom_low'] and row['market_val_bn'] > avg_market_val * thresh['shadow_spike']:
                    flag = " ⚠️(Lái Nội)"
                    shadow_spikes += 1
                elif row['sm_dominance_pct'] > thresh['dom_high']:
                    flag = " 🎯(Tay To)"

            # IN BẰNG CỘT DỮ LIỆU ĐÃ LỌC (adj)
            print(f"{date_str:<10} | {row['close']:>10,.0f} | {row['market_val_bn']:>12.1f} | {row['f_net_adj']:>10.2f} | {row['p_net_adj']:>10.2f} | {row['shadow_flow_bn']:>11.1f} | {row['sm_dominance_pct']:>7.1f}% | {row['cum_total']:>11.2f} | {row['dynamic_vwap']:>10,.0f}{flag}")

        # TỔNG KẾT
        final_row = df_merged.iloc[-1]
        avg_dominance = df_merged['sm_dominance_pct'].mean()
        
        vwap = final_row['dynamic_vwap']
        current_price = final_row['close']
        vwap_status = ""
        if vwap > 0:
            diff_pct = (current_price - vwap) / vwap * 100
            vwap_status = f"(Tay to đang {'Lãi' if diff_pct > 0 else 'Lỗ'} {abs(diff_pct):.1f}%)"

        print("\n" + "="*145)
        print(f" 🎯 TỔNG KẾT BÓC TÁCH DÒNG TIỀN (SAU KHI LỌC SANG TAY) [ {ticker} - {universe} ]")
        print(f"    Giai đoạn: {df_merged.iloc[0]['time'].strftime('%d/%m/%Y')} -> {final_row['time'].strftime('%d/%m/%Y')} ({len(df_merged)} phiên)")
        print("="*145)
        print(f" 🔹 TỔNG LƯỢNG TỒN KHO TAY TO  : {final_row['cum_total']:>10.2f} Tỷ VNĐ")
        print(f" 🔹 TỔNG LƯỢNG TỒN KHO NGOẠI   : {final_row['cum_f_total']:>10.2f} Tỷ VNĐ")
        print(f" 🔹 TỔNG LƯỢNG TỒN KHO NỘI     : {final_row['cum_p_total']:>10.2f} Tỷ VNĐ")
        print(f" 🔹 GIÁ VỐN TRUNG BÌNH (VWAP)  : {vwap:>10,.0f} đ {vwap_status}")
        print(f" 🔹 Tỷ lệ Chi phối Trung bình  : {avg_dominance:>10.1f}% (Quyền lực của Smart Money)")
        print(f" 🔹 Số phiên 'Lái Nội' bùng nổ : {shadow_spikes:>10} phiên")
        print("-" * 145)
        
        # 🚀 CHẨN ĐOÁN THEO RỔ
        if universe == 'VNSmallCap' and avg_dominance > thresh['dom_high']:
            print(f" 🚨 ĐỘT BIẾN LỚN: {ticker} là Penny nhưng Tây/Tự doanh chi phối tới {avg_dominance:.1f}%! Rất có thể có tin nội gián (M&A).")
        elif shadow_spikes >= 3 and final_row['cum_total'] <= 0:
            print(" 🚨 KẾT LUẬN: Sóng Đầu Cơ Lái Nội! Tay to (Tây/Tự doanh) đang xả hoặc không quan tâm.")
        elif final_row['cum_total'] > 0 and avg_dominance >= thresh['dom_high']:
            print(" 🌟 KẾT LUẬN: Sóng Tổ Chức! Smart Money đang nắm quyền kiểm soát.")
            if current_price < vwap: print("    => CƠ HỘI VÀNG: Giá hiện tại đang RẺ HƠN giá vốn của Tay to!")
        else:
            print(" ⚖️ KẾT LUẬN: Dòng tiền giằng co, chưa rõ phe nào kiểm soát hoàn toàn.")
        print("="*145)

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