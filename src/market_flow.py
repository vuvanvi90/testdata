import pandas as pd
import numpy as np
from datetime import datetime

class MarketFlowAnalyzer:
    """Cỗ máy X-Quang Dòng Tiền - Bản vá Đồng bộ Thời gian chuẩn (True-Time Window)"""
    def __init__(self):
        # Đồng bộ Bộ lọc Hạn sử dụng với Smart Money
        self.MAX_DELAY_DAYS = 15 

    def analyze_flow(self, ticker, df_pr, df_l2=None, target_date_str=None, lookback_sessions=130):
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

        if df_l2 is None or df_l2.empty: return result

        # ÉP TARGET_DATE VỀ NAIVE (NẾU CÓ TIMEZONE)
        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        if getattr(target_date, 'tz', None) is not None:
            target_date = target_date.tz_localize(None)

        # =====================================================================
        # 1. TIỀN XỬ LÝ LƯỚI LỌC THỜI GIAN CHUẨN (TRUE-TIME WINDOW)
        # =====================================================================
        
        # Lọc lấy các field cần thiết và clone data
        filterd_cols = [
            'time', 'ticker', 'close', 'volume', 'matched_volume',
            'fr_net_value_matched', 'fr_net_volume_matched'
        ]
        df_l2_v = df_l2[[c for c in filterd_cols if c in df_l2.columns]].copy()
        if 'matched_volume' in df_l2_v.columns and 'volume' not in df_l2_v.columns:
            df_l2_v = df_l2_v.rename(columns={'matched_volume': 'volume'})
        
        # ÉP df_l2_v VỀ NAIVE (BẢO HIỂM 2 LỚP)
        if getattr(df_l2_v['time'].dt, 'tz', None) is not None:
            df_l2_v['time'] = df_l2_v['time'].dt.tz_localize(None)

        df_l2_v = df_l2_v[df_l2_v['time'] <= target_date].sort_values('time')
        if df_l2_v.empty: return result

        # CHẶN ĐỨNG LỖI "ZOMBIE STOCK"
        last_price_date = df_l2_v['time'].iloc[-1]
        if (target_date - last_price_date).days > self.MAX_DELAY_DAYS:
            result["status"] = "OUTDATED_DATA"
            return result

        # CẮT ĐÚNG N PHIÊN GIAO DỊCH THỰC TẾ (Mặc định 130 phiên ~ 6 tháng)
        df_l2_v = df_l2_v.tail(lookback_sessions).copy()
        if len(df_l2_v) < 20:
            result["status"] = "INSUFFICIENT_DATA"
            return result

        result["adv_20"] = df_l2_v.tail(20)['volume'].mean()
        df_flow = df_l2_v[['time', 'close', 'volume', 'fr_net_value_matched', 'fr_net_volume_matched']].copy()
        
        # Lấy ngày bắt đầu chính xác từ tập df_p đã cắt để khớp với Dòng tiền
        start_date = df_flow['time'].iloc[0] 

        # =====================================================================
        # CHIẾT XUẤT DÒNG TIỀN "SẠCH" (MATCHED ONLY)
        # =====================================================================

        # 1. Dòng tiền Ngoại
        df_flow['val_f'] = df_flow['fr_net_value_matched']
        df_flow['net_f'] = df_flow['fr_net_volume_matched']

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

    def format_report(self, res):
        """In báo cáo đẹp mắt ra Terminal"""
        if res['status'] != 'SUCCESS':
            return f"[-] Không đủ dữ liệu phân tích cho {res['ticker']}."
            
        rpt = f"\n🔍 HỒ SƠ TẠO LẬP: {res['ticker']} (Tính đến {res['target_date']})\n"
        rpt += "-"*50 + "\n"
        rpt += f"📍 Ngày Khởi Thủy (Bắt đầu gom) : {res['anchor_date']}\n"
        rpt += f"📦 Tồn kho Cá Mập (Inventory)   : {res['inventory']:,.0f} cổ phiếu\n"
        rpt += f"💰 Giá vốn Trung bình (VWAP)    : {res['sm_vwap']:,.0f} đ\n"
        rpt += f"🌊 Thanh khoản TT (ADV 20)      : {res['adv_20']:,.0f} cổ/phiên\n"
        
        # Đánh giá rủi ro
        dtl = res['dtl_days']
        if dtl > 5:
            rpt += f"⏳ Sức ép Hấp thụ (DTL)         : {dtl:.1f} ngày 🟢 (An Toàn - Cá mập bị nhốt)\n"
        elif dtl < 1 and res['inventory'] > 0:
            rpt += f"⏳ Sức ép Hấp thụ (DTL)         : {dtl:.1f} ngày 🔴 (Nguy Hiểm - Xả cái một)\n"
        else:
            rpt += f"⏳ Sức ép Hấp thụ (DTL)         : {dtl:.1f} ngày 🟡 (Trung lập)\n"
            
        rpt += f"⚡ Phân kỳ Dòng tiền            : {res['divergence']}\n"
        rpt += "-"*50
        return rpt