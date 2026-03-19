import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketFlowAnalyzer:
    """
    Cỗ máy X-Quang Dòng Tiền (Smart Money X-Ray Engine)
    Đo lường Tồn kho, Giá vốn và Sức ép thanh khoản của Nhà tạo lập.
    """
    def __init__(self, df_price, df_foreign, df_prop):
        """
        Khởi tạo với Master Data từ Parquet.
        :param df_price: DataFrame lịch sử giá (master_price)
        :param df_foreign: DataFrame khối ngoại (foreign_flow)
        :param df_prop: DataFrame tự doanh (prop_flow)
        """
        self.df_price = df_price
        self.df_foreign = df_foreign
        self.df_prop = df_prop

    def analyze_flow(self, ticker, target_date_str=None, lookback_months=6):
        """
        Phân tích dòng tiền của một mã cổ phiếu tính đến một ngày cụ thể.
        :param target_date_str: Ngày kết thúc phân tích (YYYY-MM-DD). Nếu None, lấy ngày hiện tại.
        """
        result = {
            "ticker": ticker,
            "target_date": None,
            "anchor_date": None,         # Ngày Khởi Thủy (Bắt đầu gom hàng)
            "inventory": 0,              # Tồn kho hiện tại (Cổ phiếu)
            "sm_vwap": 0.0,              # Giá vốn trung bình của Cá mập
            "adv_20": 0,                 # Thanh khoản trung bình 20 phiên
            "dtl_days": 0.0,             # Days To Liquidate (Số ngày cần để xả hết hàng)
            "divergence": "NEUTRAL",     # Phân kỳ (BULLISH/BEARISH)
            "status": "NO_DATA"
        }

        # 1. THIẾT LẬP POINT-IN-TIME (Chống nhìn trộm tương lai)
        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        result["target_date"] = target_date.strftime('%Y-%m-%d')
        start_date = target_date - pd.DateOffset(months=lookback_months)

        # Lọc Price Data
        df_p = self.df_price[(self.df_price['ticker'] == ticker) & (self.df_price['time'] <= target_date)].copy()
        if df_p.empty: return result

        # Tính ADV 20 phiên của thị trường
        df_p = df_p.sort_values('time')
        result["adv_20"] = df_p.tail(20)['volume'].mean()

        # Lọc Smart Money Data
        df_f = self.df_foreign[(self.df_foreign['ticker'] == ticker) & (self.df_foreign['time'] <= target_date)].copy()
        df_pr = self.df_prop[(self.df_prop['ticker'] == ticker) & (self.df_prop['time'] <= target_date)].copy()

        # 2. GỘP DÒNG TIỀN (Smart Money = Khối Ngoại + Tự Doanh)
        # Tạo khung thời gian chuẩn từ price để merge
        df_flow = df_p[['time', 'close', 'volume']].copy()
        
        # Đưa net_volume của Tây và Tự doanh vào
        if not df_f.empty:
            df_f_net = df_f[['time', 'foreign_net_volume']].rename(columns={'foreign_net_volume': 'f_net'})
            df_flow = pd.merge(df_flow, df_f_net, on='time', how='left')
        else:
            df_flow['f_net'] = 0
            
        if not df_pr.empty:
            df_pr_net = df_pr[['time', 'prop_net_volume']].rename(columns={'prop_net_volume': 'p_net'})
            df_flow = pd.merge(df_flow, df_pr_net, on='time', how='left')
        else:
            df_flow['p_net'] = 0

        # Điền NaN bằng 0 và tính Tổng Net Volume mỗi phiên
        df_flow.fillna(0, inplace=True)
        df_flow['sm_net'] = df_flow['f_net'] + df_flow['p_net']

        # 3. TÌM NGÀY KHỞI THỦY (ANCHOR DATE)
        # Chỉ xét trong khung thời gian lookback (VD: 6 tháng qua)
        df_lookback = df_flow[df_flow['time'] >= start_date].copy()
        if df_lookback.empty or df_lookback['sm_net'].sum() == 0:
            result["status"] = "NO_SMART_MONEY"
            return result

        # Tính cộng dồn (Cumulative Sum) để tìm đáy
        df_lookback['cum_sm'] = df_lookback['sm_net'].cumsum()
        
        # Đáy của đường cộng dồn chính là lúc họ ngừng bán và bắt đầu nhặt hàng (Anchor Date)
        min_idx = df_lookback['cum_sm'].idxmin()
        anchor_date = df_lookback.loc[min_idx, 'time']
        result["anchor_date"] = anchor_date.strftime('%Y-%m-%d')

        # 4. TÍNH TOÁN TỒN KHO & GIÁ VỐN (TỪ ANCHOR DATE ĐẾN TARGET DATE)
        df_accum = df_flow[df_flow['time'] >= anchor_date].copy()
        
        inventory = df_accum['sm_net'].sum()
        result["inventory"] = inventory

        # Tính VWAP (Chỉ tính những phiên họ Mua ròng để lấy giá vốn thực tế)
        df_buy_only = df_accum[df_accum['sm_net'] > 0].copy()
        if not df_buy_only.empty and inventory > 0:
            total_cost = (df_buy_only['close'] * df_buy_only['sm_net']).sum()
            total_buy_vol = df_buy_only['sm_net'].sum()
            result["sm_vwap"] = total_cost / total_buy_vol if total_buy_vol > 0 else 0

        # 5. TÍNH SỨC ÉP HẤP THỤ (LIQUIDITY SHOCK / DTL)
        if result["adv_20"] > 0 and inventory > 0:
            result["dtl_days"] = inventory / result["adv_20"]

        # 6. QUÉT PHÂN KỲ DÒNG TIỀN (BEARISH / BULLISH TRAP)
        # So sánh 10 phiên gần nhất
        df_recent = df_lookback.tail(10)
        if len(df_recent) >= 5:
            price_trend = df_recent['close'].iloc[-1] - df_recent['close'].iloc[0]
            flow_trend = df_recent['cum_sm'].iloc[-1] - df_recent['cum_sm'].iloc[0]
            
            if price_trend > 0 and flow_trend < 0:
                result["divergence"] = "BEARISH_TRAP (Giá Kéo - Dòng Tiền Xả)"
            elif price_trend < 0 and flow_trend > 0:
                result["divergence"] = "BULLISH_ACCUM (Giá Đè - Dòng Tiền Gom)"

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