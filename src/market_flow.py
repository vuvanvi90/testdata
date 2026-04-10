import pandas as pd
import numpy as np
from datetime import datetime

class MarketFlowAnalyzer:
    """Cỗ máy X-Quang Dòng Tiền - Bản vá Đồng bộ Thời gian chuẩn (True-Time Window)"""
    def __init__(self):
        # Đồng bộ Bộ lọc Hạn sử dụng với Smart Money
        self.MAX_DELAY_DAYS = 15 

    def analyze_flow(self, ticker, df_p, df_f, df_pr, target_date_str=None, lookback_sessions=130):
        result = {
            "ticker": ticker,
            "anchor_date": None,
            "inventory": 0,
            "sm_vwap": 0.0,
            "adv_20": 0,
            "dtl_days": 0.0,
            "divergence": "NEUTRAL",
            "status": "NO_DATA",
            "correlation": 0.0
        }

        if df_p is None or df_p.empty: return result

        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()

        # =====================================================================
        # 1. TIỀN XỬ LÝ LƯỚI LỌC THỜI GIAN CHUẨN (TRUE-TIME WINDOW)
        # =====================================================================
        df_p = df_p[df_p['time'] <= target_date].sort_values('time').copy()
        if df_p.empty: return result

        # 🚀 CHẶN ĐỨNG LỖI "ZOMBIE STOCK"
        last_price_date = df_p['time'].iloc[-1]
        if (target_date - last_price_date).days > self.MAX_DELAY_DAYS:
            result["status"] = "OUTDATED_DATA"
            return result

        # 🚀 CẮT ĐÚNG N PHIÊN GIAO DỊCH THỰC TẾ (Mặc định 130 phiên ~ 6 tháng)
        df_p = df_p.tail(lookback_sessions).copy()
        if len(df_p) < 20:
            result["status"] = "INSUFFICIENT_DATA"
            return result
            
        result["adv_20"] = df_p.tail(20)['volume'].mean()
        df_flow = df_p[['time', 'close', 'volume']].copy()
        
        # Lấy ngày bắt đầu chính xác từ tập df_p đã cắt để khớp với Dòng tiền
        start_date = df_flow['time'].iloc[0] 

        # =====================================================================
        # 2. GHÉP NỐI DÒNG TIỀN (SMART MONEY)
        # =====================================================================
        if df_f is not None and not df_f.empty:
            df_f_valid = df_f[(df_f['time'] >= start_date) & (df_f['time'] <= target_date)]
            df_flow = pd.merge(df_flow, df_f_valid[['time', 'foreign_net_volume', 'foreign_net_value']].rename(columns={'foreign_net_volume': 'f_net'}), on='time', how='left')
        else:
            df_flow['f_net'] = 0
            df_flow['foreign_net_value'] = 0

        if df_pr is not None and not df_pr.empty:
            df_pr_valid = df_pr[(df_pr['time'] >= start_date) & (df_pr['time'] <= target_date)]
            df_flow = pd.merge(df_flow, df_pr_valid[['time', 'prop_net_volume', 'prop_net_value']].rename(columns={'prop_net_volume': 'p_net'}), on='time', how='left')
        else:
            df_flow['p_net'] = 0
            df_flow['prop_net_value'] = 0

        df_flow.fillna(0, inplace=True)
        df_flow['sm_net'] = df_flow['f_net'] + df_flow['p_net']

        # =====================================================================
        # 3. XÁC ĐỊNH ĐIỂM NEO (ANCHOR DATE) AN TOÀN
        # =====================================================================
        if df_flow['sm_net'].sum() == 0:
            result["status"] = "NO_SMART_MONEY"
            return result

        df_flow['cum_sm'] = df_flow['sm_net'].cumsum()
        
        valid_anchor_window = df_flow.iloc[:-10] 
        if not valid_anchor_window.empty:
            anchor_date = valid_anchor_window.loc[valid_anchor_window['cum_sm'].idxmin(), 'time']
        else:
            anchor_date = df_flow['time'].iloc[0]
            
        result["anchor_date"] = anchor_date.strftime('%Y-%m-%d')

        # =====================================================================
        # 4 & 5. KẾ TOÁN GIÁ VỐN LÁI (SMART MONEY COST BASIS ACCOUNTING)
        # =====================================================================
        df_flow['net_f'] = df_flow.get('f_net', 0)
        df_flow['net_p'] = df_flow.get('p_net', 0)
        df_flow['net_sm'] = df_flow.get('sm_net', 0)

        df_flow['val_f'] = df_flow.get('foreign_net_value', 0)
        df_flow['val_p'] = df_flow.get('prop_net_value', 0)
        df_flow['val_sm'] = df_flow['val_f'] + df_flow['val_p']
        
        # Để vẽ biểu đồ (Chart), chúng ta vẫn tính Cumsum
        df_flow['cum_sm'] = df_flow['net_sm'].cumsum()

        # 🚀 THUẬT TOÁN TÍNH VWAP THEO DÒNG CHẢY (FLOW-BASED VWAP)
        inventory = 0
        current_vwap = 0.0

        for idx, row in df_flow.iterrows():
            net_vol = row['net_sm']
            net_val = row['val_sm']

            if net_vol > 0:
                # MUA RÒNG: Tính lại trung bình giá vốn
                # Vốn cũ = tồn kho * giá vốn hiện tại. Vốn mới = vốn cũ + tiền vừa mua
                total_cost = (inventory * current_vwap) + net_val
                inventory += net_vol
                current_vwap = total_cost / inventory if inventory > 0 else 0
                
            elif net_vol < 0:
                # BÁN RÒNG: Giảm tồn kho, KHÔNG ĐỔI giá vốn của phần hàng còn lại
                inventory += net_vol
                if inventory <= 0:
                    # Nếu Lái bán sạch sành sanh kho (Washout/Phân phối hết), Reset game!
                    inventory = 0
                    current_vwap = 0.0
            
            # Lưu lại giá vốn theo từng ngày (Dùng để tính phân kỳ nếu cần)
            df_flow.at[idx, 'dynamic_vwap'] = current_vwap

        # Trả kết quả cuối cùng ra ngoài
        result["inventory"] = inventory
        result["sm_vwap"] = current_vwap if inventory > 0 else -1.0

        if result["adv_20"] > 0 and result["inventory"] > 0:
            result["dtl_days"] = result["inventory"] / result["adv_20"]

        # =====================================================================
        # 6. ĐO LƯỜNG PHÂN KỲ BẰNG HỆ SỐ TƯƠNG QUAN (15 PHIÊN)
        # =====================================================================
        # df_flow bây giờ đã chuẩn chỉnh 100% về mặt thời gian
        df_recent = df_flow.tail(15).copy()
        if len(df_recent) >= 10:
            # Kiểm tra Zero-Variance (Bản vá ở lượt trước)
            if df_recent['close'].nunique() <= 1 or df_recent['cum_sm'].nunique() <= 1:
                correlation = 0.0 
            else:
                correlation = df_recent['close'].corr(df_recent['cum_sm'])
                
            result["correlation"] = round(correlation, 2) if pd.notna(correlation) else 0.0
            
            price_trend_is_up = df_recent['close'].iloc[-1] > df_recent['close'].mean()

            if correlation < -0.4:
                if price_trend_is_up:
                    result["divergence"] = "BEARISH_TRAP" # Giá được neo cao nhưng SM đang âm thầm xả
                else:
                    result["divergence"] = "BULLISH_ACCUM" # Giá bị đè xuống đáy nhưng SM đang âm thầm gom
            elif correlation > 0.6:
                result["divergence"] = "TREND_CONFIRMED" # Giá và Dòng tiền đồng thuận

        result["status"] = "SUCCESS"
        return result