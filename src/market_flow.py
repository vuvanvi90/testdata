import pandas as pd
from datetime import datetime

class MarketFlowAnalyzer:
    """Cỗ máy X-Quang Dòng Tiền - Tối ưu hóa O(1)"""
    def __init__(self):
        pass

    def analyze_flow(self, ticker, df_p, df_f, df_pr, target_date_str=None, lookback_months=6):
        result = {
            "ticker": ticker,
            "anchor_date": None,
            "inventory": 0,
            "sm_vwap": 0.0,
            "adv_20": 0,
            "dtl_days": 0.0,
            "divergence": "NEUTRAL",
            "status": "NO_DATA"
        }

        if df_p is None or df_p.empty: return result

        target_date = pd.to_datetime(target_date_str) if target_date_str else pd.Timestamp.now().normalize()
        start_date = target_date - pd.DateOffset(months=lookback_months)

        # Lọc dữ liệu theo ngày
        df_p = df_p[df_p['time'] <= target_date].sort_values('time')
        result["adv_20"] = df_p.tail(20)['volume'].mean()

        df_flow = df_p[['time', 'close', 'volume']].copy()

        # Ghép Khối ngoại
        if df_f is not None and not df_f.empty:
            df_f = df_f[df_f['time'] <= target_date]
            df_flow = pd.merge(df_flow, df_f[['time', 'foreign_net_volume']].rename(columns={'foreign_net_volume': 'f_net'}), on='time', how='left')
        else:
            df_flow['f_net'] = 0

        # Ghép Tự doanh
        if df_pr is not None and not df_pr.empty:
            df_pr = df_pr[df_pr['time'] <= target_date]
            df_flow = pd.merge(df_flow, df_pr[['time', 'prop_net_volume']].rename(columns={'prop_net_volume': 'p_net'}), on='time', how='left')
        else:
            df_flow['p_net'] = 0

        df_flow.fillna(0, inplace=True)
        df_flow['sm_net'] = df_flow['f_net'] + df_flow['p_net']

        df_lookback = df_flow[df_flow['time'] >= start_date].copy()
        if df_lookback.empty or df_lookback['sm_net'].sum() == 0:
            result["status"] = "NO_SMART_MONEY"
            return result

        # Tìm đáy cộng dồn
        df_lookback['cum_sm'] = df_lookback['sm_net'].cumsum()
        anchor_date = df_lookback.loc[df_lookback['cum_sm'].idxmin(), 'time']
        result["anchor_date"] = anchor_date.strftime('%Y-%m-%d')

        # Tính Tồn kho & VWAP
        df_accum = df_flow[df_flow['time'] >= anchor_date].copy()
        result["inventory"] = df_accum['sm_net'].sum()

        df_buy_only = df_accum[df_accum['sm_net'] > 0]
        if not df_buy_only.empty and result["inventory"] > 0:
            result["sm_vwap"] = (df_buy_only['close'] * df_buy_only['sm_net']).sum() / df_buy_only['sm_net'].sum()

        # Tính DTL
        if result["adv_20"] > 0 and result["inventory"] > 0:
            result["dtl_days"] = result["inventory"] / result["adv_20"]

        # Quét Phân kỳ 10 phiên
        df_recent = df_lookback.tail(10)
        if len(df_recent) >= 5:
            price_trend = df_recent['close'].iloc[-1] - df_recent['close'].iloc[0]
            flow_trend = df_recent['cum_sm'].iloc[-1] - df_recent['cum_sm'].iloc[0]
            if price_trend > 0 and flow_trend < 0:
                result["divergence"] = "BEARISH_TRAP"
            elif price_trend < 0 and flow_trend > 0:
                result["divergence"] = "BULLISH_ACCUM"

        result["status"] = "SUCCESS"
        return result