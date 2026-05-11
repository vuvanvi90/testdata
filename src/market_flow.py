import pandas as pd
import numpy as np
from datetime import datetime

class MarketFlowAnalyzer:
    """
    Cỗ máy X-Quang Dòng Tiền - Bản vá Đồng bộ Thời gian chuẩn (True-Time Window)
    🚀 VERSION 3.0 (PHASE 4): 
       - Sửa lỗi Parameter Mapping tương thích 100% với sniper.py.
       - Tích hợp Chẩn đoán Vị thế Cá mập (Shark Position).
    """
    def __init__(self):
        # Đồng bộ Bộ lọc Hạn sử dụng với Smart Money
        self.MAX_DELAY_DAYS = 15 

    def analyze_flow(self, ticker, df_price, df_prop=None, target_date_str=None, lookback_sessions=130):
        """
        LƯU Ý THAM SỐ:
        - df_price: Chính là df_price_l2 (Đã chứa toàn bộ Price + Khớp lệnh L2) từ sniper.py truyền vào.
        - df_prop: Dữ liệu Tự doanh.
        """
        result = {
            "ticker": ticker,
            "anchor_date": None,
            "inventory": 0,
            "sm_vwap": 0.0,
            "adv_20": 0,
            "dtl_days": 0.0,
            "divergence": "NEUTRAL",
            "sm_status": "NEUTRAL", # Trạng thái Vị thế Cá mập
            "status": "NO_DATA",
            "correlation": 0.0
        }

        if df_price is None or df_price.empty: return result

        # ÉP TARGET_DATE VỀ NAIVE (NẾU CÓ TIMEZONE)
        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        if getattr(target_date, 'tz', None) is not None:
            target_date = target_date.tz_localize(None)

        # =====================================================================
        # 1. TIỀN XỬ LÝ LƯỚI LỌC THỜI GIAN CHUẨN (TRUE-TIME WINDOW)
        # =====================================================================
        df_p = df_price.copy()
        
        # Ép thời gian về Naive (Bảo hiểm 2 lớp)
        if getattr(df_p['time'].dt, 'tz', None) is not None:
            df_p['time'] = df_p['time'].dt.tz_localize(None)

        df_p = df_p[df_p['time'] <= target_date].sort_values('time')
        if df_p.empty: return result

        # Chặn đứng lỗi "Zombie Stock"
        last_price_date = df_p['time'].iloc[-1]
        if (target_date - last_price_date).days > self.MAX_DELAY_DAYS:
            result["status"] = "OUTDATED_DATA"
            return result

        # Cắt đúng 130 phiên giao dịch thực tế
        df_p = df_p.tail(lookback_sessions).copy()
        if len(df_p) < 20:
            result["status"] = "INSUFFICIENT_DATA"
            return result

        if 'matched_volume' in df_p.columns and 'volume' not in df_p.columns:
            df_p = df_p.rename(columns={'matched_volume': 'volume'})

        result["adv_20"] = df_p.tail(20)['volume'].mean()
        
        # Trích xuất các cột L2 trực tiếp từ df_price
        cols_to_keep = ['time', 'close', 'volume']
        for c in ['fr_net_value_matched', 'fr_net_volume_matched']:
            if c in df_p.columns: cols_to_keep.append(c)
                
        df_flow = df_p[cols_to_keep].copy()
        start_date = df_flow['time'].iloc[0] 

        # =====================================================================
        # CHIẾT XUẤT DÒNG TIỀN "SẠCH" (MATCHED ONLY)
        # =====================================================================
        # Dòng tiền Ngoại
        if 'fr_net_value_matched' in df_flow.columns:
            df_flow['val_f'] = df_flow['fr_net_value_matched']
            df_flow['net_f'] = df_flow['fr_net_volume_matched']
        else:
            df_flow['val_f'] = 0; df_flow['net_f'] = 0

        # Dòng tiền Tự doanh (Lấy từ df_prop Premium)
        if df_prop is not None and not df_prop.empty:
            pr_v = df_prop[(df_prop['time'] >= start_date) & (df_prop['time'] <= target_date)].copy()
            
            # Ưu tiên lấy matched data, nếu không có thì lấy data tổng
            if 'prop_net_val_matched' in pr_v.columns:
                pr_v['p_val'] = pr_v['prop_net_val_matched']
                pr_v['p_vol'] = pr_v['prop_net_vol_matched']
            elif 'prop_net_value' in pr_v.columns:
                pr_v['p_val'] = pr_v['prop_net_value']
                pr_v['p_vol'] = pr_v['prop_net_volume']
            else:
                pr_v['p_val'] = 0; pr_v['p_vol'] = 0
                
            df_flow = pd.merge(df_flow, pr_v[['time', 'p_val', 'p_vol']], on='time', how='left')
            df_flow['val_p'] = df_flow['p_val'].fillna(0)
            df_flow['net_p'] = df_flow['p_vol'].fillna(0)
        else:
            df_flow['val_p'] = 0; df_flow['net_p'] = 0

        df_flow.fillna(0, inplace=True)

        # 3. Hợp nhất Dòng tiền Thông minh (SM)
        df_flow['net_sm'] = df_flow['net_f'] + df_flow['net_p']
        df_flow['val_sm'] = df_flow['val_f'] + df_flow['val_p']
        df_flow['cum_sm'] = df_flow['net_sm'].cumsum()

        # Vá lỗi logic tìm Anchor Date
        if 'net_sm' not in df_flow.columns or df_flow['net_sm'].sum() == 0:
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
        # CHẨN ĐOÁN VỊ THẾ CÁ MẬP VÀ PHÂN KỲ
        # =====================================================================
        current_price = df_flow['close'].iloc[-1]
        
        # 🚀 ĐỌC VỊ TÂM LÝ CÁ MẬP (SHARK POSITION)
        if current_vwap > 0 and inventory > 0:
            if current_price > current_vwap * 1.05:
                result["sm_status"] = "PROFIT (Lái đang lãi >5%, rủi ro chốt lời trên đường lên)"
            elif current_price < current_vwap * 0.95:
                result["sm_status"] = "TRAPPED (Lái đang kẹp lỗ >5%, rủi ro Call Margin/Đạp lủng đáy)"
            elif current_price >= current_vwap:
                result["sm_status"] = "SAFE_ZONE (Giá chớm vượt VWAP EOD, Phe Bò đang làm chủ cục diện)"
            else:
                result["sm_status"] = "ACCUMULATING (Giá đè dưới VWAP EOD, Lái đang ép gom hàng)"

        # Đo lường phân kỳ 15 phiên
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
                    result["divergence"] = "BEARISH_TRAP (Kéo xả - Giá tăng nhưng Dòng tiền rút)" 
                else:
                    result["divergence"] = "BULLISH_ACCUM (Đạp gom - Giá giảm nhưng Dòng tiền vào)" 
            elif correlation > 0.6:
                result["divergence"] = "TREND_CONFIRMED (Đồng thuận - Giá và Dòng tiền cùng chiều)" 

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
        rpt += f"🎯 Vị thế Tay to (Shark Status) : {res['sm_status']}\n"
        rpt += "-"*50
        return rpt