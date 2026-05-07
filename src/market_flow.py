import pandas as pd
import numpy as np
from datetime import datetime

class MarketFlowAnalyzer:
    """Cỗ máy X-Quang Dòng Tiền - Bản vá Đồng bộ Thời gian chuẩn (True-Time Window)"""
    def __init__(self):
        # Đồng bộ Bộ lọc Hạn sử dụng với Smart Money
        self.MAX_DELAY_DAYS = 15 

    def analyze_flow(self, ticker, df_p, df_pr, df_l2=None, target_date_str=None, lookback_sessions=130):
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

        # ÉP TARGET_DATE VỀ NAIVE (NẾU CÓ TIMEZONE)
        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        if getattr(target_date, 'tz', None) is not None:
            target_date = target_date.tz_localize(None)

        # =====================================================================
        # 1. TIỀN XỬ LÝ LƯỚI LỌC THỜI GIAN CHUẨN (TRUE-TIME WINDOW)
        # =====================================================================
        df_p = df_p.copy()
        
        # ÉP df_p VỀ NAIVE (BẢO HIỂM 2 LỚP)
        if getattr(df_p['time'].dt, 'tz', None) is not None:
            df_p['time'] = df_p['time'].dt.tz_localize(None)

        df_p = df_p[df_p['time'] <= target_date].sort_values('time')
        if df_p.empty: return result

        # CHẶN ĐỨNG LỖI "ZOMBIE STOCK"
        last_price_date = df_p['time'].iloc[-1]
        if (target_date - last_price_date).days > self.MAX_DELAY_DAYS:
            result["status"] = "OUTDATED_DATA"
            return result

        # CẮT ĐÚNG N PHIÊN GIAO DỊCH THỰC TẾ (Mặc định 130 phiên ~ 6 tháng)
        df_p = df_p.tail(lookback_sessions).copy()
        if len(df_p) < 20:
            result["status"] = "INSUFFICIENT_DATA"
            return result
            
        result["adv_20"] = df_p.tail(20)['volume'].mean()
        df_flow = df_p[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Lấy ngày bắt đầu chính xác từ tập df_p đã cắt để khớp với Dòng tiền
        start_date = df_flow['time'].iloc[0] 

        # =====================================================================
        # CHIẾT XUẤT DÒNG TIỀN "SẠCH" (MATCHED ONLY)
        # =====================================================================

        # 1. Dòng tiền Ngoại (Lấy từ L2)
        if df_l2 is not None and not df_l2.empty:
            l2_v = df_l2[(df_l2['time'] >= start_date) & (df_l2['time'] <= target_date)]
            df_flow = pd.merge(df_flow, l2_v[['time', 'fr_buy_value_matched', 'fr_sell_value_matched', 
                                             'fr_buy_volume_matched', 'fr_sell_volume_matched']], on='time', how='left')
            df_flow['val_f'] = df_flow['fr_buy_value_matched'] - df_flow['fr_sell_value_matched']
            df_flow['net_f'] = df_flow['fr_buy_volume_matched'] - df_flow['fr_sell_volume_matched']
        else:
            df_flow['val_f'] = 0; df_flow['net_f'] = 0

        # 2. Dòng tiền Tự doanh (Lấy từ Prop Premium)
        if df_pr is not None and not df_pr.empty:
            pr_v = df_pr[(df_pr['time'] >= start_date) & (df_pr['time'] <= target_date)]
            df_flow = pd.merge(df_flow, pr_v[['time', 'prop_net_val_matched', 'prop_net_vol_matched']], on='time', how='left')
            df_flow['val_p'] = df_flow['prop_net_val_matched']
            df_flow['net_p'] = df_flow['prop_net_vol_matched']
        else:
            df_flow['val_p'] = 0; df_flow['net_p'] = 0

        df_flow.fillna(0, inplace=True)

        # 3. Hợp nhất Dòng tiền Thông minh (SM)
        df_flow['net_sm'] = df_flow['net_f'] + df_flow['net_p']
        df_flow['val_sm'] = df_flow['val_f'] + df_flow['val_p']
        df_flow['cum_sm'] = df_flow['net_sm'].cumsum()

        # XÁC ĐỊNH ANCHOR DATE
        if df_flow['sm_net'].sum() if 'sm_net' in df_flow.columns else df_flow['net_sm'].sum() == 0:
            result["status"] = "NO_SMART_MONEY"
            return result
            
        valid_anchor_window = df_flow.iloc[:-10] 
        anchor_date = valid_anchor_window.loc[valid_anchor_window['cum_sm'].idxmin(), 'time'] if not valid_anchor_window.empty else df_flow['time'].iloc[0]
        result["anchor_date"] = anchor_date.strftime('%Y-%m-%d')

        # THUẬT TOÁN TÍNH VWAP THEO DÒNG CHẢY (FLOW-BASED VWAP)
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