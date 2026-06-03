import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class DarkPoolEngine:
    """
    ĐỘNG CƠ DÒNG TIỀN NGẦM (DARK POOL ENGINE) V4.1
    🚀 Cập nhật V4.1 (Logic Phân luồng Dữ liệu):
       - Ưu tiên 1: Dùng price_l2 + prop_flow nếu ngày mục tiêu ĐÃ CHỐT EOD.
       - Ưu tiên 2: Chỉ dùng put_through nếu ngày mục tiêu LÀ LIVE T0 (Chưa có EOD).
    """
    # Đã loại bỏ df_board khỏi tham số đầu vào
    def __init__(self, df_l2=None, df_pt=None, df_prop=None, data_dir='data/parquet'):
        self.data_dir = Path(data_dir)
        
        # Dependency Injection để tối ưu RAM khi chạy cùng OmniMatrix
        self.df_price_l2 = df_l2 if df_l2 is not None else self._load_parquet_safe(self.data_dir / 'eod/master_price_l2.parquet')
        self.df_pt = df_pt if df_pt is not None else self._load_parquet_safe(self.data_dir / 't0/master_put_through.parquet')
        self.df_prop = df_prop if df_prop is not None else self._load_parquet_safe(self.data_dir / 'eod/prop_flow.parquet')
        
        if not self.df_price_l2.empty and 'matched_volume' in self.df_price_l2.columns and 'volume' not in self.df_price_l2.columns:
            self.df_price_l2 = self.df_price_l2.rename(columns={'matched_volume': 'volume'})

    def _load_parquet_safe(self, path):
        if path.exists():
            try: 
                df = pd.read_parquet(path)
                if 'time' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        df['time'] = pd.to_datetime(df['time'], unit='ms') if pd.api.types.is_numeric_dtype(df['time']) else pd.to_datetime(df['time'])
                    if getattr(df['time'].dt, 'tz', None) is not None:
                        df['time'] = df['time'].dt.tz_localize(None)
                return df
            except Exception as e: 
                print(f"[!] Lỗi đọc file {path.name}: {e}")
        return pd.DataFrame()

    def get_darkpool_context(self, ticker, target_date_str=None, lookback_days=5):
        """
        API Lõi cung cấp Cấu trúc Dòng tiền Ngầm cho OmniMatrix
        """
        target_date = pd.to_datetime(target_date_str).normalize() if target_date_str else pd.Timestamp.now().normalize()
        DIV = 1_000_000_000
        
        result = {
            'ticker': ticker,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'past_val_bn': 0.0,
            'past_intent': "NONE",
            't0_val_bn': 0.0,
            't0_intent': "NONE",
            'darkpool_vwap': 0.0,      
            'is_super_bullish': False, 
            'is_danger_dump': False
        }

        # Kiểm tra xem target_date đã có dữ liệu EOD (chốt phiên) chưa
        has_eod_for_target = False
        if not self.df_price_l2.empty:
            df_target_check = self.df_price_l2[(self.df_price_l2['ticker'] == ticker) & (self.df_price_l2['time'] == target_date)]
            has_eod_for_target = not df_target_check.empty

        # 1. TRÍCH XUẤT LỊCH SỬ TỪ EOD (price_l2 & prop_flow)
        past_dp_vol, past_dp_val = 0, 0
        s_buy, s_sell = 0, 0
        
        # Nếu đã có EOD cho target_date: Lấy dữ liệu gom đến <= target_date
        # Nếu chưa có EOD (Live T0): Lấy dữ liệu < target_date
        eod_filter = (self.df_price_l2['time'] <= target_date) if has_eod_for_target else (self.df_price_l2['time'] < target_date)
        
        if not self.df_price_l2.empty:
            df_l2 = self.df_price_l2[(self.df_price_l2['ticker'] == ticker) & eod_filter].sort_values('time').tail(lookback_days)
            
            if not df_l2.empty:
                past_val = df_l2['deal_value'].sum() / DIV if 'deal_value' in df_l2.columns else 0
                past_dp_vol = df_l2['deal_volume'].sum() if 'deal_volume' in df_l2.columns else 0
                past_dp_val = df_l2['deal_value'].sum() if 'deal_value' in df_l2.columns else 0
                
                f_buy = df_l2['fr_buy_value_deal'].sum() / DIV if 'fr_buy_value_deal' in df_l2.columns else 0
                f_sell = df_l2['fr_sell_value_deal'].sum() / DIV if 'fr_sell_value_deal' in df_l2.columns else 0
                
                p_buy, p_sell = 0, 0
                if not self.df_prop.empty:
                    df_p = self.df_prop[(self.df_prop['ticker'] == ticker) & (self.df_prop['time'] <= df_l2['time'].max())].tail(lookback_days)
                    p_buy = df_p['prop_buy_val_deal'].sum() / DIV if 'prop_buy_val_deal' in df_p.columns else 0
                    p_sell = df_p['prop_sell_val_deal'].sum() / DIV if 'prop_sell_val_deal' in df_p.columns else 0

                s_buy = max(0, past_val - f_buy - p_buy)
                s_sell = max(0, past_val - f_sell - p_sell)

                if s_buy > (past_val * 0.6): result['past_intent'] = "🥷 LÁI NỘI GOM NGẦM"
                elif s_sell > (past_val * 0.6): result['past_intent'] = "🩸 LÁI NỘI XẢ NGẦM"
                elif f_buy > (past_val * 0.6): result['past_intent'] = "🌍 TÂY LÔNG GOM NGẦM"
                elif f_sell > (past_val * 0.6): result['past_intent'] = "🏃 TÂY LÔNG XẢ NGẦM"
                else: result['past_intent'] = "🤝 TRAO TAY HỖN HỢP"
                
                result['past_val_bn'] = past_val

        # 2. XỬ LÝ DỮ LIỆU T0 (Fallback Logic)
        t0_dp_vol, t0_dp_val = 0, 0
        
        if has_eod_for_target:
            # ƯU TIÊN 1: Ngày này đã chốt phiên, không cần dùng file put_through
            # Dữ liệu T0 chính là dữ liệu của ngày target_date trong df_l2
            df_l2_today = df_l2[df_l2['time'] == target_date]
            if not df_l2_today.empty:
                result['t0_val_bn'] = df_l2_today['deal_value'].sum() / DIV if 'deal_value' in df_l2_today.columns else 0
                result['t0_intent'] = "✅ ĐÃ CHỐT EOD"
        else:
            # ƯU TIÊN 2: Ngày này chưa chốt phiên (Live), tiến hành quét put_through
            if not self.df_pt.empty:
                p_col = 'symbol' if 'symbol' in self.df_pt.columns else 'ticker'
                df_pt_t0 = self.df_pt[(self.df_pt[p_col] == ticker) & (self.df_pt['time'].dt.date == target_date.date())]
                
                if not df_pt_t0.empty:
                    t0_val = df_pt_t0['match_value'].sum() / DIV
                    t0_dp_vol = df_pt_t0['volume'].sum()
                    t0_dp_val = df_pt_t0['match_value'].sum()
                    
                    df_pt_t0['weighted_change'] = df_pt_t0['change_percent'] * df_pt_t0['match_value']
                    avg_change = (df_pt_t0['weighted_change'].sum() / df_pt_t0['match_value'].sum()) * 100 if t0_dp_val > 0 else 0

                    if avg_change > 1.0:
                        result['t0_intent'] = f"🔥 PREMIUM (+{avg_change:.1f}%)"
                        if "XẢ" not in result['past_intent']:
                            result['is_super_bullish'] = True
                    elif avg_change < -1.0:
                        result['t0_intent'] = f"🧊 DISCOUNT ({avg_change:.1f}%)"
                        if "XẢ" in result['past_intent'] or s_sell > s_buy * 2:
                            result['is_danger_dump'] = True
                    else:
                        result['t0_intent'] = "⚖️ THAM CHIẾU"
                        
                    result['t0_val_bn'] = t0_val

        # 3. CHẾ TẠO CHÉN THÁNH: Tính Tọa độ Giá Vốn VWAP của Lái Nội
        total_dp_vol = past_dp_vol + t0_dp_vol
        total_dp_val = past_dp_val + t0_dp_val
        if total_dp_vol > 0:
            result['darkpool_vwap'] = total_dp_val / total_dp_vol
            
        return result