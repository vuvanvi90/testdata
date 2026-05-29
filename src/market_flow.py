import pandas as pd
import numpy as np
from datetime import datetime

class MarketFlowAnalyzer:
    """
    CỖ MÁY X-QUANG DÒNG TIỀN (MARKET FLOW ANALYZER) - V4.0
    🚀 BẢN VÁ LƯỢNG TỬ (Khuyến nghị E):
       - Tích hợp Cảm biến Gãy Nền (Structural Breakdown Sensor).
       - Khả năng Reset Bộ nhớ Dòng tiền khi Lái buông trận (Markdown).
       - Phân biệt Gãy thật (Xả Vol lớn) vs Rũ bỏ (Spring Vol thấp).
    """
    def __init__(self):
        self.MAX_DELAY_DAYS = 15 

    def analyze_flow(self, ticker, df_price, df_prop=None, target_date_str=None, lookback_sessions=130):
        result = {
            "ticker": ticker,
            "anchor_date": None,
            "inventory": 0,
            "sm_vwap": 0.0,
            "adv_20": 0,
            "dtl_days": 0.0,
            "divergence": "NEUTRAL",
            "sm_status": "NEUTRAL",
            "status": "NO_DATA",
            "correlation": 0.0,
            "reset_count": 0
        }

        if df_price is None or df_price.empty: return result

        # ÉP TARGET_DATE VỀ NAIVE
        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        if getattr(target_date, 'tz', None) is not None:
            target_date = target_date.tz_localize(None)

        # 1. TIỀN XỬ LÝ & TẠO CẢM BIẾN
        df_p = df_price.copy()
        if getattr(df_p['time'].dt, 'tz', None) is not None:
            df_p['time'] = df_p['time'].dt.tz_localize(None)

        df_p = df_p[df_p['time'] <= target_date].sort_values('time')
        if df_p.empty: return result

        last_price_date = df_p['time'].iloc[-1]
        if (target_date - last_price_date).days > self.MAX_DELAY_DAYS:
            result["status"] = "OUTDATED_DATA"
            return result

        df_p = df_p.tail(lookback_sessions).copy()
        if len(df_p) < 20:
            result["status"] = "INSUFFICIENT_DATA"
            return result

        if 'matched_volume' in df_p.columns and 'volume' not in df_p.columns:
            df_p = df_p.rename(columns={'matched_volume': 'volume'})

        # 🚀 TẠO CÁC CHỈ BÁO ĐỘNG HỖ TRỢ CẢM BIẾN (Vectorized)
        if 'low' not in df_p.columns: df_p['low'] = df_p['close']
        df_p['support_20d'] = df_p['low'].shift(1).rolling(20).min()
        df_p['adv_20d'] = df_p['volume'].shift(1).rolling(20).mean()

        result["adv_20"] = df_p.tail(20)['volume'].mean()
        
        # Trích xuất dữ liệu L2 cần thiết
        cols_to_keep = ['time', 'close', 'low', 'volume', 'support_20d', 'adv_20d']
        for c in ['fr_net_value_matched', 'fr_net_volume_matched']:
            if c in df_p.columns: cols_to_keep.append(c)
                
        df_flow = df_p[cols_to_keep].copy()
        start_date = df_flow['time'].iloc[0] 

        # 2. CHIẾT XUẤT DÒNG TIỀN "SẠCH"
        if 'fr_net_value_matched' in df_flow.columns:
            df_flow['val_f'] = df_flow['fr_net_value_matched']
            df_flow['net_f'] = df_flow['fr_net_volume_matched']
        else:
            df_flow['val_f'] = 0; df_flow['net_f'] = 0

        if df_prop is not None and not df_prop.empty:
            pr_v = df_prop[(df_prop['time'] >= start_date) & (df_prop['time'] <= target_date)].copy()
            if 'prop_net_val_matched' in pr_v.columns:
                pr_v['p_val'] = pr_v['prop_net_val_matched']
                pr_v['p_vol'] = pr_v['prop_net_vol_matched']
            else:
                pr_v['p_val'] = 0; pr_v['p_vol'] = 0
                
            df_flow = pd.merge(df_flow, pr_v[['time', 'p_val', 'p_vol']], on='time', how='left')
            df_flow['val_p'] = df_flow['p_val'].fillna(0)
            df_flow['net_p'] = df_flow['p_vol'].fillna(0)
        else:
            df_flow['val_p'] = 0; df_flow['net_p'] = 0

        df_flow.fillna(0, inplace=True)

        df_flow['net_sm'] = df_flow['net_f'] + df_flow['net_p']
        df_flow['val_sm'] = df_flow['val_f'] + df_flow['val_p']

        # 3. THUẬT TOÁN TÍNH VWAP VÀ RESET BỘ NHỚ (DYNAMIC MEMORY)
        inventory = 0
        current_vwap = 0.0
        current_anchor = df_flow['time'].iloc[0]
        reset_count = 0

        for idx, row in df_flow.iterrows():
            net_vol = row['net_sm']
            net_val = row['val_sm']
            prev_inv = inventory

            if net_vol > 0:
                total_cost = (inventory * current_vwap) + net_val
                inventory += net_vol
                current_vwap = total_cost / inventory if inventory > 0 else 0
            elif net_vol < 0:
                inventory += net_vol
                if inventory <= 0:
                    inventory = 0
                    current_vwap = 0.0

            # 🚀 CẢM BIẾN GÃY NỀN (STRUCTURAL BREAKDOWN SENSOR)
            support = row['support_20d']
            adv = row['adv_20d']
            close_p = row['close']
            vol = row['volume']
            
            if pd.notna(support) and pd.notna(adv) and support > 0 and adv > 0:
                # Nếu giá đâm thủng Hỗ trợ 20 phiên > 2% VÀ Vol xả > 1.3x Trung bình
                if close_p < support * 0.98 and vol > adv * 1.3:
                    # GÃY NỀN THẬT! Lái buông trận -> Xóa sổ quá khứ
                    inventory = 0
                    current_vwap = 0.0
                    reset_count += 1
                    
            # LIÊN TỤC CẬP NHẬT NGÀY KHỞI THỦY (ANCHOR DATE)
            if inventory <= 0:
                current_anchor = row['time']
            elif prev_inv <= 0 and inventory > 0:
                current_anchor = row['time'] # Bắt đầu chu kỳ gom mới

        result["anchor_date"] = current_anchor.strftime('%Y-%m-%d')
        result["inventory"] = inventory
        result["sm_vwap"] = current_vwap if inventory > 0 else -1.0
        result["reset_count"] = reset_count

        if result["adv_20"] > 0 and result["inventory"] > 0:
            result["dtl_days"] = result["inventory"] / result["adv_20"]

        # 4. CHẨN ĐOÁN VỊ THẾ CÁ MẬP VÀ PHÂN KỲ
        current_price = df_flow['close'].iloc[-1]
        
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
        df_flow['cum_sm_dynamic'] = df_flow['net_sm'].cumsum() # Dùng tạm để tính corr
        df_recent = df_flow.tail(15).copy()
        
        if len(df_recent) >= 10:
            if df_recent['close'].nunique() <= 1 or df_recent['cum_sm_dynamic'].nunique() <= 1:
                correlation = 0.0 
            else:
                correlation = df_recent['close'].corr(df_recent['cum_sm_dynamic'])
                
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
        if res['status'] != 'SUCCESS':
            return f"[-] Không đủ dữ liệu phân tích cho {res['ticker']}."
            
        rpt = f"\n🔍 HỒ SƠ TẠO LẬP: {res['ticker']} (Tính đến {res['target_date']})\n"
        rpt += "-"*50 + "\n"
        rpt += f"📍 Ngày Khởi Thủy (Bắt đầu gom) : {res['anchor_date']}\n"
        
        if res.get('reset_count', 0) > 0:
            rpt += f"🔄 Số lần Reset Gãy nền        : {res['reset_count']} lần (Đã xóa bộ nhớ cũ)\n"
            
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