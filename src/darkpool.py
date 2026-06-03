import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

class DarkPoolRadar:
    """
    ĐÀI QUAN SÁT DÒNG TIỀN NGẦM (DARK POOL RADAR) V3.1
    🚀 Áp dụng Kiến trúc Kép (Lambda):
       - Luồng EOD (T-5 -> T-1): Khấu trừ price_l2 & prop_flow để định danh Lái Nội.
       - Luồng T0 (Real-time): Quét put_through & board để bắt quả tang Premium/Discount.
    """
    def __init__(self, data_dir='data/parquet'):
        self.data_dir = Path(data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Đài quan sát Dòng tiền Ngầm (Dark Pool Radar)...")
        
        self.df_intra = self._load_parquet_safe(self.data_dir / 't0/master_intraday.parquet')
        self.df_pt = self._load_parquet_safe(self.data_dir / 't0/master_put_through.parquet')
        self.df_price_l2 = self._load_parquet_safe(self.data_dir / 'eod/master_price_l2.parquet')
        self.df_prop = self._load_parquet_safe(self.data_dir / 'eod/prop_flow.parquet')
        self.df_board = self._load_parquet_safe(self.data_dir / 't0/master_board.parquet')
        self.df_idx = self._load_parquet_safe(self.data_dir / 'macro/index_components.parquet')
        
        if not self.df_price_l2.empty:
            if 'matched_volume' in self.df_price_l2.columns and 'volume' not in self.df_price_l2.columns:
                self.df_price_l2 = self.df_price_l2.rename(columns={'matched_volume': 'volume'})

        self.t0_date, self.t1_date = self._identify_timelines()

        # Biến lưu trữ kết quả
        self.basket_stats = {}
        self.anomalies = []
        self.forecasts = []

    def _load_parquet_safe(self, path):
        """Đọc file và lột bỏ Timezone an toàn"""
        if path.exists():
            try: 
                df = pd.read_parquet(path)
                if 'time' in df.columns:
                    # Ép kiểu về datetime nếu API trả về dạng số/chuỗi
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        if pd.api.types.is_numeric_dtype(df['time']):
                            df['time'] = pd.to_datetime(df['time'], unit='ms')
                        else:
                            df['time'] = pd.to_datetime(df['time'])
                    
                    # Lột bỏ Timezone (Ép về Naive)
                    if getattr(df['time'].dt, 'tz', None) is not None:
                        df['time'] = df['time'].dt.tz_localize(None)
                return df
            except Exception as e: 
                print(f"[!] Lỗi đọc file {path.name}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _identify_timelines(self):
        """Xác định Ngày T0 (Live) và Ngày T-1 (EOD)"""
        dates = set()
        if not self.df_price_l2.empty and 'time' in self.df_price_l2.columns:
            dates.update(self.df_price_l2['time'].dt.date.unique())
        
        trading_dates = sorted(list(dates))
        if not trading_dates: return datetime.now().date(), datetime.now().date()
            
        t0 = datetime.now().date()
        if not self.df_pt.empty and 'time' in self.df_pt.columns:
            max_pt = self.df_pt['time'].dt.date.max()
            if max_pt > trading_dates[-1]: t0 = max_pt
            else: t0 = trading_dates[-1]

        past_dates = [d for d in trading_dates if d < t0]
        t1 = past_dates[-1] if past_dates else t0
        return t0, t1

    def _get_basket_mapping(self):
        idx_dict = self.df_idx.set_index('ticker')['index_code'].to_dict()
        return idx_dict

    def _analyze_historical_deals(self, ticker, df_l2_t, df_prop_t):
        """Bóc tách Kế toán Kép T-1 (Săn tìm Lái Nội)"""
        if df_l2_t.empty: return 0, 0, 0, 0, 0, "NO_DATA"

        DIV = 1_000_000_000
        total_deal_val = df_l2_t['deal_value'].sum() / DIV
        if total_deal_val == 0: return 0, 0, 0, 0, 0, "NONE"

        f_buy_deal = df_l2_t['fr_buy_value_deal'].sum() / DIV if 'fr_buy_value_deal' in df_l2_t.columns else 0
        f_sell_deal = df_l2_t['fr_sell_value_deal'].sum() / DIV if 'fr_sell_value_deal' in df_l2_t.columns else 0
        
        p_buy_deal, p_sell_deal = 0, 0
        if not df_prop_t.empty:
            p_buy_deal = df_prop_t['total_deal_buy_trade_value'].sum() / DIV if 'total_deal_buy_trade_value' in df_prop_t.columns else 0
            p_sell_deal = df_prop_t['total_deal_sell_trade_value'].sum() / DIV if 'total_deal_sell_trade_value' in df_prop_t.columns else 0

        # Phương trình Khấu trừ Bóng tối (Shadow Deduction)
        shadow_buy = max(0, total_deal_val - f_buy_deal - p_buy_deal)
        shadow_sell = max(0, total_deal_val - f_sell_deal - p_sell_deal)

        intent = "NEUTRAL"
        if shadow_buy > (total_deal_val * 0.6): intent = "🥷 LÁI NỘI GOM NGẦM"
        elif shadow_sell > (total_deal_val * 0.6): intent = "🩸 LÁI NỘI XẢ NGẦM"
        elif f_buy_deal > (total_deal_val * 0.6): intent = "🌍 TÂY LÔNG GOM NGẦM"
        elif f_sell_deal > (total_deal_val * 0.6): intent = "🏃 TÂY LÔNG XẢ NGẦM"
        elif p_buy_deal > (total_deal_val * 0.6): intent = "🏢 TỰ DOANH GOM NGẦM"
        elif p_sell_deal > (total_deal_val * 0.6): intent = "🏦 TỰ DOANH XẢ NGẦM"
        else: intent = "🤝 TRAO TAY HỖN HỢP"

        return total_deal_val, shadow_buy, shadow_sell, f_buy_deal, f_sell_deal, intent

    def _analyze_t0_deals(self, ticker, df_pt_t0):
        """Bóc tách Thỏa thuận T0 Real-time (Đo lường Khát máu)"""
        if df_pt_t0.empty: return 0, 0, "NONE"

        DIV = 1_000_000_000
        t0_val = df_pt_t0['match_value'].sum() / DIV
        
        # 1. Đo lường Premium/Discount so với Bảng điện
        board_price = 0
        if not self.df_board.empty:
            b_col = 'symbol' if 'symbol' in self.df_board.columns else 'ticker'
            b_df = self.df_board[self.df_board[b_col] == ticker]
            if not b_df.empty:
                board_price = b_df.iloc[0].get('close_price', 0)

        # 2. Đo lường với VWAP T0 Khớp lệnh
        vwap_t0 = board_price
        if not self.df_intra.empty:
            i_df = self.df_intra[(self.df_intra['ticker'] == ticker) & (self.df_intra['time'].dt.date == self.t0_date)]
            if not i_df.empty:
                total_vol = i_df['volume'].sum()
                vwap_t0 = (i_df['price'] * i_df['volume']).sum() / total_vol if total_vol > 0 else board_price

        # 3. Chẩn đoán Lệnh T0
        df_pt_t0['weighted_change'] = df_pt_t0['change_percent'] * df_pt_t0['match_value']
        avg_change = (df_pt_t0['weighted_change'].sum() / df_pt_t0['match_value'].sum()) * 100

        intent = "NEUTRAL"
        if avg_change > 1.0 or (vwap_t0 > 0 and df_pt_t0['price'].mean() > vwap_t0 * 1.01):
            intent = f"🔥 PREMIUM (+{avg_change:.1f}%)"
        elif avg_change < -1.0 or (vwap_t0 > 0 and df_pt_t0['price'].mean() < vwap_t0 * 0.99):
            intent = f"🧊 DISCOUNT ({avg_change:.1f}%)"
        else:
            intent = f"⚖️ THAM CHIẾU"

        return t0_val, avg_change, intent

    def run_radar(self):
        if self.df_price_l2.empty:
            print("[!] Thiếu dữ liệu L2.")

        print(f"[*] Phân tích Dòng tiền Ngầm: EOD (T-1: {self.t1_date}) và Live (T0: {self.t0_date})")
        
        idx_dict = self._get_basket_mapping()
        all_tickers = self.df_price_l2['ticker'].unique()
        
        # Lấy 5 ngày trước T0
        trading_days = sorted(self.df_price_l2['time'].dt.date.unique())
        past_days = [d for d in trading_days if d < self.t0_date]
        last_5d = past_days[-5:] if len(past_days) >= 5 else past_days

        anomalies_temp = []
        for ticker in all_tickers:
            basket = idx_dict.get(ticker, 'HOSE')
            # if basket not in ['VN30', 'VNMidCap', 'VNSmallCap']: continue
            if basket not in ['VN30', 'VNMidCap']: continue

            # --- DỮ LIỆU ---
            df_l2_5d = self.df_price_l2[(self.df_price_l2['ticker'] == ticker) & (self.df_price_l2['time'].dt.date.isin(last_5d))]
            df_prop_5d = self.df_prop[(self.df_prop['ticker'] == ticker) & (self.df_prop['time'].dt.date.isin(last_5d))] if not self.df_prop.empty else pd.DataFrame()
            
            p_col = 'symbol' if 'symbol' in self.df_pt.columns else 'ticker'
            df_pt_t0 = self.df_pt[(self.df_pt[p_col] == ticker) & (self.df_pt['time'].dt.date == self.t0_date)] if not self.df_pt.empty else pd.DataFrame()

            # --- TÍNH TOÁN QUÁ KHỨ 5D (T-5 -> T-1) ---
            val_5d, s_buy, s_sell, f_buy, f_sell, past_intent = self._analyze_historical_deals(ticker, df_l2_5d, df_prop_5d)
            
            # --- TÍNH TOÁN T0 LIVE ---
            val_t0, t0_change, t0_intent = self._analyze_t0_deals(ticker, df_pt_t0)

            total_val = val_5d + val_t0
            if total_val < (50 if basket == 'VN30' else 15 if basket == 'VNMidCap' else 3): continue

            # Thanh khoản trung bình khớp lệnh 20D
            df_20d = self.df_price_l2[(self.df_price_l2['ticker'] == ticker) & (self.df_price_l2['time'].dt.date.isin(past_days[-20:]))]
            ma20_vol = df_20d['volume'].mean() if not df_20d.empty else 0
            
            # Lấy Volume của riêng Deal 5D+T0
            pt_vol_5d = df_l2_5d['deal_volume'].sum() if 'deal_volume' in df_l2_5d.columns else 0
            pt_vol_t0 = df_pt_t0['volume'].sum() if not df_pt_t0.empty else 0
            liqd_ratio = ((pt_vol_5d + pt_vol_t0) / ma20_vol) if ma20_vol > 0 else 0

            # --- LƯỚI LỌC CÁ MẬP ---
            if liqd_ratio > (1.0 if basket == 'VN30' else 1.5 if basket == 'VNMidCap' else 2.5) or val_t0 > 10.0:
                score = (liqd_ratio * 20) + (abs(t0_change) * 10) + (val_t0 / total_val * 10 if total_val > 0 else 0)
                
                # Phán quyết gộp
                if val_t0 > 0: final_intent = f"T0: {t0_intent} | T-1: {past_intent}"
                else: final_intent = f"T-1: {past_intent}"

                anomalies_temp.append({
                    'ticker': ticker, 'basket': basket, 'val_bn': val_5d, 't0_val_bn': val_t0, 
                    'ratio': liqd_ratio, 'intent': final_intent, 'score': score, 
                    'shadow_buy': s_buy, 'shadow_sell': s_sell
                })

        self.anomalies = sorted(anomalies_temp, key=lambda x: x['score'], reverse=True)[:10]

        # 3. DỰ BÁO CHIẾN THUẬT
        for an in self.anomalies:
            ticker = an['ticker']
            t0_intent = an['intent'].split('|')[0]
            s_buy, s_sell = an['shadow_buy'], an['shadow_sell']
            
            action, msg = "NONE", ""
            if "PREMIUM" in t0_intent and "GOM" in an['intent']:
                action = "BUY_TARGET"; msg = f"🔥 T0 Khát hàng Premium + Lịch sử Gom ngầm. Siêu tín hiệu Bứt phá."
            elif "DISCOUNT" in t0_intent and ("XẢ" in an['intent'] or s_sell > s_buy * 2):
                action = "DANGER"; msg = f"🩸 T0 Tháo cống Discount + Lịch sử Xả ngầm. Cảnh báo Phân phối đỉnh/Đổ vỏ."
            elif s_buy > s_sell * 3:
                action = "BUY_TARGET"; msg = f"🥷 Lái nội Mua Gom áp đảo ({s_buy:.1f} Tỷ vs Xả {s_sell:.1f} Tỷ). Bệ đỡ giá an toàn."
            elif s_sell > s_buy * 3:
                action = "DANGER"; msg = f"🩸 Lái nội Xả Ngầm áp đảo ({s_sell:.1f} Tỷ vs Mua {s_buy:.1f} Tỷ). Rủi ro Tắm máu."
            else:
                msg = "🤝 Dòng tiền thỏa thuận giằng co, sang tay giữ nền."
                
            self.forecasts.append({'ticker': ticker, 'action': action, 'msg': f"[{ticker}]: {msg}"})

        self._export_signals()
        self._print_report()

    def _export_signals(self):
        out_path = self.data_dir.parent / 'live/darkpool_signals.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {}
        for an in self.anomalies:
            ticker = an['ticker']
            fc = next((f for f in self.forecasts if f['ticker'] == ticker), None)
            
            export_data[ticker] = {
                "basket": an['basket'], "val_bn": an['val_bn'], "t0_val_bn": an['t0_val_bn'],
                "ratio": an['ratio'], "intent": an['intent'],
                "forecast": fc['msg'] if fc else "", "action": fc['action'] if fc else "NONE",
                "valid_for_date": self.t0_date.strftime('%Y-%m-%d'), 
                "date_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)

    def _print_report(self):
        today_str = datetime.now().strftime('%d/%m/%Y')
        print("\n" + "="*95)
        print(f" 🕵️ ĐÀI QUAN SÁT DÒNG TIỀN NGẦM (DARK POOL RADAR V3.1) | BÁO CÁO NGÀY: {today_str}")
        print("="*95)
        
        print("\n 🦈 TOP CÁ MẬP GIAO DỊCH BẤT THƯỜNG (ANOMALY DETECTED)")
        if self.anomalies:
            print(f"    {'Mã (Rổ)':<15} | {'Sang tay 5D':<15} | {'Riêng T0':<12} | {'Cường độ':<13} | {'Bản Án Khấu Trừ V3.1':<20}")
            print("    " + "-"*85)
            for i, an in enumerate(self.anomalies, 1):
                ticker_basket = f"{i}. {an['ticker']} ({an['basket'][:4]})"
                val_5d = f"{an['val_bn']:>5.0f} Tỷ"
                val_t0 = f"{an['t0_val_bn']:>4.0f} Tỷ" if an['t0_val_bn'] > 0 else "  -  "
                ratio = f"{an['ratio']:.1f}x MA20"
                print(f"    {ticker_basket:<15} | {val_5d:<15} | {val_t0:<12} | {ratio:<13} | {an['intent']}")
        else:
            print("    - Không phát hiện hành vi Thỏa thuận đột biến nào.")

        print("\n 🔮 DỰ BÁO CHIẾN THUẬT BẮN TỈA T0")
        if self.forecasts:
            for f in self.forecasts: print(f"    {f['msg']}")
        else:
            print("    - Chưa có Setup Thỏa thuận nào đủ mạnh.")
        print("="*95 + "\n")

if __name__ == "__main__":
    radar = DarkPoolRadar()
    radar.run_radar()