import pandas as pd
import shutil
import os
from datetime import datetime
from pathlib import Path

from src.forecaster import WyckoffForecaster
from src.smart_money import SmartMoneyEngine
from src.shadow_profiler import ShadowProfiler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

class SignalInspector:
    def __init__(self, universe='VN30', df_price=None, df_foreign=None, df_prop=None, df_comp=None, df_idx=None, lookback_days=60):
        """
        Khởi tạo Hệ thống Kiểm toán Tín hiệu (Signal Inspector)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Hệ thống Kiểm toán (Signal Inspector)...")
        self.universe = universe
        self.df_price = df_price
        self.df_foreign = df_foreign
        self.df_prop = df_prop
        self.df_comp = df_comp
        self.df_idx = df_idx
        self.momentum_days = lookback_days

        self._load_and_filter_data()

    def _load_and_filter_data(self):
        print("[*] Đang nạp và Phân rổ Master Data vào RAM...")
        # 1. Lấy danh sách mã theo Rổ (Universe)
        valid_tickers = []
        if not self.df_idx.empty:
            if self.universe == "VN30": 
                valid_tickers = self.df_idx[self.df_idx['index_code'] == 'VN30']['ticker'].tolist()
            elif self.universe == "VNMidCap": 
                valid_tickers = self.df_idx[self.df_idx['index_code'] == 'VNMidCap']['ticker'].tolist()
            elif self.universe == "VNSmallCap": 
                valid_tickers = self.df_idx[self.df_idx['index_code'] == 'VNSmallCap']['ticker'].tolist()
            else: 
                valid_tickers = self.df_idx[self.df_idx['index_code'] == 'HOSE']['ticker'].tolist()

        # 2. Lọc dữ liệu
        if not self.df_price.empty and 'ticker' in self.df_price.columns:
            self.df_price = self.df_price[self.df_price['ticker'].isin(valid_tickers)].copy()

        if not self.df_foreign.empty and 'ticker' in self.df_foreign.columns:
            self.df_foreign = self.df_foreign[self.df_foreign['ticker'].isin(valid_tickers)].copy()
            
        if not self.df_prop.empty and 'ticker' in self.df_prop.columns:
            self.df_prop = self.df_prop[self.df_prop['ticker'].isin(valid_tickers)].copy()

        # Chuẩn hóa thời gian
        for df in [self.df_price, self.df_foreign, self.df_prop]:
            if not df.empty and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()

        if not self.df_price.empty:
            self.df_price = self.df_price.sort_values(by=['ticker', 'time'])

        # 3. Build Dictionary O(1) & Khởi tạo Engine
        self.price_dict = dict(tuple(self.df_price.groupby('ticker'))) if not self.df_price.empty else {}
        self.foreign_dict = dict(tuple(self.df_foreign.groupby('ticker'))) if not self.df_foreign.empty else {}
        self.prop_dict = dict(tuple(self.df_prop.groupby('ticker'))) if not self.df_prop.empty else {}
        
        out_shares_dict = {}
        if not self.df_comp.empty and 'ticker' in self.df_comp.columns and 'issue_share' in self.df_comp.columns:
            out_shares_dict = self.df_comp.set_index('ticker')['issue_share'].to_dict()

        try:
            self.sm_engine = SmartMoneyEngine(
                foreign_dict=self.foreign_dict, 
                prop_dict=self.prop_dict, 
                out_shares_dict=out_shares_dict,
                price_dict=self.price_dict,
                universe=self.universe
            )
        except Exception as e:
            print(f"[!] Lỗi khởi động Smart Money: {e}")

        try:
            self.profiler = ShadowProfiler(price_df=self.df_price, verbose=False)
        except Exception as e:
            print(f"[!] Lỗi khởi động Shadow Profiler: {e}")

    def scan_momentum(self, top_n=30):
        print("\n" + "="*75)
        print(f" 🔍 GIAI ĐOẠN 1: QUÉT ĐỘNG LƯỢNG (MOMENTUM SCANNER - {self.universe})")
        print("="*75)
        
        if self.df_price.empty: return None, None

        # Thiết lập Thanh khoản chuẩn (Tỷ VNĐ) tùy theo Rổ
        if self.universe == "VN30": min_liquidity_bn = 50.0
        elif self.universe == "VNMidCap": min_liquidity_bn = 15.0
        elif self.universe == "VNSmallCap": min_liquidity_bn = 2.0
        else: min_liquidity_bn = 5.0

        df_recent = self.df_price.groupby('ticker').tail(self.momentum_days + 5).copy()
        results = []
        
        for ticker, group in df_recent.groupby('ticker'):
            if len(group) < self.momentum_days + 1: continue 
                
            group = group.sort_values('time').reset_index(drop=True)
            avg_vol = group['volume'].tail(20).mean()
            avg_price = group['close'].tail(20).mean()
            liquidity_bn = (avg_vol * avg_price) / 1_000_000_000
            
            if liquidity_bn < min_liquidity_bn: continue
                
            current_close = group['close'].iloc[-1]
            close_1d = group['close'].iloc[-2]  
            close_1w = group['close'].iloc[-6]  
            close_1m = group['close'].iloc[-21] 
            close_custom = group['close'].iloc[-(self.momentum_days + 1)]

            ret_1d = (current_close - close_1d) / close_1d * 100
            ret_1w = (current_close - close_1w) / close_1w * 100
            ret_1m = (current_close - close_1m) / close_1m * 100
            ret_custom = (current_close - close_custom) / close_custom * 100
            
            results.append({
                'Ticker': ticker,
                'Close': current_close,
                'Liquidity_Bn': liquidity_bn,
                'Ret_1D': ret_1d, 'Ret_1W': ret_1w, 'Ret_1M': ret_1m,
                'Ret_Custom': ret_custom 
            })
            
        if not results:
            print("[!] Không có mã nào đạt tiêu chuẩn thanh khoản.")
            return None, None
            
        df_mom = pd.DataFrame(results)
        df_winners = df_mom.nlargest(top_n, 'Ret_Custom').reset_index(drop=True)
        df_losers = df_mom.nsmallest(top_n, 'Ret_Custom').reset_index(drop=True)

        print(f"\n🏆 TOP {top_n} SIÊU CỔ PHIẾU (WINNERS - {self.momentum_days}D):")
        print(f"{'MÃ CP':<8} | {'GIÁ HIỆN TẠI':<15} | {'1 NGÀY':>8} | {'1 TUẦN':>8} | {'1 THÁNG':>8} | {'N PHIÊN':>8}")
        print("-" * 75)
        for _, r in df_winners.head(15).iterrows():
            print(f"{r['Ticker']:<8} | {r['Close']:<15,.0f} | {r['Ret_1D']:>7.1f}% | {r['Ret_1W']:>7.1f}% | {r['Ret_1M']:>7.1f}% | {r['Ret_Custom']:>7.1f}%")
            
        print(f"\n🩸 TOP {top_n} TỘI ĐỒ (LOSERS - {self.momentum_days}D):")
        print("-" * 75)
        for _, r in df_losers.head(15).iterrows():
            print(f"{r['Ticker']:<8} | {r['Close']:<15,.0f} | {r['Ret_1D']:>7.1f}% | {r['Ret_1W']:>7.1f}% | {r['Ret_1M']:>7.1f}% | {r['Ret_Custom']:>7.1f}%")
            
        return df_winners, df_losers

    def _run_audit_loop(self, df_targets, desc="Audit"):
        """Hàm dùng chung để chạy Cỗ máy Thời gian (Bọc thép Pandas)"""
        if df_targets is None or df_targets.empty: return {}
        
        tickers_to_audit = df_targets['Ticker'].tolist()
        trading_days = sorted(self.df_price['time'].unique())
        audit_days = trading_days[-self.momentum_days:] if len(trading_days) >= self.momentum_days else trading_days

        audit_temp_dir = Path('data/temp_audit')
        audit_temp_dir.mkdir(parents=True, exist_ok=True)
        signals_dict = {t: [] for t in tickers_to_audit}

        for target_date in tqdm(audit_days, desc=desc):
            mask = (self.df_price['ticker'].isin(tickers_to_audit)) & (self.df_price['time'] <= target_date)
            df_slice = self.df_price[mask]
            
            if df_slice.empty: continue
            
            try:
                forecaster = WyckoffForecaster(data_input=df_slice, output_dir=audit_temp_dir, run_date=target_date)
                report = forecaster.run_forecast()
                
                if report is not None and isinstance(report, pd.DataFrame) and not report.empty:
                    buy_signals = report[report['Signal'].isin(['SOS', 'SPRING', 'TEST_CUNG'])]
                    for _, row in buy_signals.iterrows():
                        ticker = row['Ticker']
                        if ticker in signals_dict:
                            signals_dict[ticker].append({
                                'date': target_date.strftime('%Y-%m-%d'),
                                'price': row['Price'],
                                'signal': row['Signal'],
                                'vpa': row.get('VPA_Status', 'N/A')
                            })
            except Exception:
                pass
                
        try: shutil.rmtree(audit_temp_dir)
        except: pass
        
        return signals_dict

    def audit_winners(self, df_winners):
        print("\n" + "="*75)
        print(" 🕰️ GIAI ĐOẠN 2: CỖ MÁY THỜI GIAN (AUDIT WINNERS)")
        print("="*75)
        
        signals_dict = self._run_audit_loop(df_winners, desc="Audit Winners")
        if not signals_dict: return pd.DataFrame()

        audit_results = []
        for ticker in df_winners['Ticker']:
            ret_custom = df_winners[df_winners['Ticker'] == ticker]['Ret_Custom'].values[0]
            current_price = df_winners[df_winners['Ticker'] == ticker]['Close'].values[0]
            fired_list = signals_dict.get(ticker, [])
            
            if len(fired_list) > 0:
                best_signal = fired_list[0] 
                caught_price = best_signal['price']
                profit = (current_price - caught_price) / caught_price * 100 if caught_price > 0 else 0
                
                diagnosis = "🟢 CAUGHT (Bắt trúng chân sóng)"
                note = f"Nổ {best_signal['signal']} ({best_signal['date']}) | Lãi dự phóng: +{profit:.1f}% | VPA: {best_signal['vpa']}"
            else:
                diagnosis = "🔴 BLIND SPOT"
                note = "Không có cụm nến Wyckoff đủ chuẩn (Sóng thông tin/Kéo sốc)."
                
            audit_results.append({
                'Ticker': ticker, 'Ret_Custom': ret_custom,
                'Signals_Count': len(fired_list), 'Diagnosis': diagnosis, 'Note': note
            })
            
        df_audit = pd.DataFrame(audit_results)
        self._print_audit_report(df_audit, "BÁO CÁO KIỂM TOÁN TÍN HIỆU (TOP WINNERS)")
        return df_audit

    def audit_losers(self, df_losers):
        print("\n" + "="*75)
        print(" 🚨 GIAI ĐOẠN 3: RÀ SOÁT BẪY BẮT DAO RƠI (AUDIT LOSERS)")
        print("="*75)
        
        signals_dict = self._run_audit_loop(df_losers, desc="Audit Losers")
        if not signals_dict: return pd.DataFrame()

        audit_results = []
        for ticker in df_losers['Ticker']:
            ret_custom = df_losers[df_losers['Ticker'] == ticker]['Ret_Custom'].values[0]
            fired_list = signals_dict.get(ticker, [])
            
            if len(fired_list) == 0:
                diagnosis = "🛡️ SAFE (Miễn nhiễm)"
                note = "Hệ thống đứng ngoài thành công."
            else:
                worst_signal = fired_list[0]
                diagnosis = "🩸 TRAP (Dính bẫy)"
                note = f"Phát lệnh ảo {worst_signal['signal']} vào {worst_signal['date']} (Giá: {worst_signal['price']:,.0f}đ)"
                
            audit_results.append({
                'Ticker': ticker, 'Ret_Custom': ret_custom,
                'Trap_Count': len(fired_list), 'Diagnosis': diagnosis, 'Note': note
            })
            
        df_audit = pd.DataFrame(audit_results)
        self._print_audit_report(df_audit, "BÁO CÁO KIỂM TOÁN LỖI (TOP LOSERS)")
        return df_audit

    def _print_audit_report(self, df_audit, title):
        print(f"\n📊 {title}:")
        print(f"{'MÃ CP':<8} | {'HIỆU SUẤT':>9} | {'TÍN HIỆU':>10} | {'CHẨN ĐOÁN':<30}")
        print("-" * 75)
        for _, r in df_audit.iterrows():
            cnt = r.get('Signals_Count', r.get('Trap_Count', 0))
            print(f"{r['Ticker']:<8} | {r['Ret_Custom']:>8.1f}% | {cnt:>10} | {r['Diagnosis']:<30}")
            print(f"   └─ {r['Note']}")

    def cross_audit_traps(self, df_audit_losers):
        print("\n" + "="*80)
        print(" 🕵️ GIAI ĐOẠN 4: KIỂM TOÁN CHÉO (TÍCH HỢP ĐỘNG CƠ SMART MONEY)")
        print("="*80)

        df_traps = df_audit_losers[df_audit_losers['Diagnosis'].str.contains('TRAP')].copy()
        if df_traps.empty:
            print("[*] Không có bẫy nào để kiểm toán chéo. Hệ thống phòng thủ 100%!")
            return

        print(f"[*] Đối chiếu {len(df_traps)} lệnh TRAP với Động cơ SmartMoneyEngine (Rổ {self.universe})...")
        import re
        blocked_count = 0
        results = []

        for _, row in df_traps.iterrows():
            ticker = row['Ticker']
            note = row['Note']
            match = re.search(r'vào (\d{4}-\d{2}-\d{2})', note)
            if not match: continue
            
            trap_date_str = match.group(1)
            
            # 🚀 GỌI TRỰC TIẾP ENGINE THAY VÌ TÍNH TAY
            sm_result = self.sm_engine.analyze_ticker(ticker, board_info=None, target_date=trap_date_str)
            
            if sm_result.get("is_danger", False):
                status = "🛡️ ĐÃ BỊ CHẶN (Engine báo Đỏ)"
                reason = " | ".join(sm_result.get("warnings", []))
                blocked_count += 1
            elif sm_result.get("total_sm_score", 0) < 0:
                status = "🛡️ ĐÃ BỊ CHẶN (Dòng tiền Âm)"
                reason = "Điểm SM < 0"
                blocked_count += 1
            else:
                status = "⚠️ LỌT LƯỚI"
                reason = "Tay to trung lập/gom nhẹ"

            results.append({
                'Ticker': ticker, 'Trap_Date': trap_date_str, 
                'Status': status, 'Reason': reason
            })

        print("\n📊 BÁO CÁO ĐỐI CHẤT SMART MONEY:")
        print(f"{'MÃ CP':<8} | {'NGÀY DÍNH BẪY':<15} | {'KẾT QUẢ ĐỐI CHẤT TRONG LIVE.PY':<30} | {'LÝ DO BỊ CHẶN'}")
        print("-" * 80)
        for r in results:
            print(f"{r['Ticker']:<8} | {r['Trap_Date']:<15} | {r['Status']:<30} | {r['Reason']}")

        print("\n" + "="*80)
        if len(df_traps) > 0:
            print(f"🎯 KẾT LUẬN: Động cơ live.py TỰ ĐỘNG CHẶN ĐỨNG {blocked_count}/{len(df_traps)} lệnh TRAP ảo.")
        print("="*80)

    def inspect_single_ticker(self, ticker):
        print("\n" + "="*90)
        print(f" 🎯 CHẾ ĐỘ BẮN TỈA (SNIPER MODE): X-QUANG MÃ [ {ticker} ]")
        print("="*90)
        
        df_t = self.price_dict.get(ticker)
        if df_t is None or df_t.empty:
            print(f"[!] Không tìm thấy dữ liệu giá cho mã {ticker}.")
            return
            
        df_t = df_t.sort_values('time').reset_index(drop=True)
        
        if len(df_t) >= self.momentum_days + 1:
            current_close = df_t['close'].iloc[-1]
            close_historical = df_t['close'].iloc[-(self.momentum_days + 1)]
            ret_historical = (current_close - close_historical) / close_historical * 100
            print(f"[*] Bối cảnh {self.momentum_days} Ngày: Lãi/Lỗ: {ret_historical:+.2f}% | Giá hiện tại: {current_close:,.0f}đ")

        trading_days = sorted(df_t['time'].unique())
        audit_days = trading_days[-self.momentum_days:] if len(trading_days) >= self.momentum_days else trading_days
        
        signals_found = []
        for target_date in audit_days:
            df_slice = df_t[df_t['time'] <= target_date]
            try:
                forecaster = WyckoffForecaster(data_input=df_slice, output_dir=None, run_date=target_date)
                report = forecaster.run_forecast()
                if report is not None and isinstance(report, pd.DataFrame) and not report.empty:
                    buy_signals = report[report['Signal'].isin(['SOS', 'SPRING', 'TEST_CUNG'])]
                    if not buy_signals.empty:
                        row = buy_signals.iloc[0]
                        signals_found.append({'date': target_date.strftime('%Y-%m-%d'), 'price': row['Price'], 'signal': row['Signal']})
            except: pass
                
        if not signals_found:
            print(f"\n[🛡️] Trong {self.momentum_days} ngày qua, KHÔNG CÓ TÍN HIỆU WYCKOFF CHUẨN NÀO.")
            return
            
        print("\n📊 NHẬT KÝ TÍN HIỆU & ĐỐI CHẤT HỆ SINH THÁI (WYCKOFF + SMART MONEY + SHADOW):")
        print(f"{'NGÀY NỔ LỆNH':<15} | {'TÍN HIỆU':<10} | {'GIÁ NỔ':>10} | {'PHÁN QUYẾT TỪ LIVE.PY':<25} | {'LÝ DO/LÁI NỘI'}")
        print("-" * 100)
        
        for sig in signals_found:
            sig_date = sig['date']
            
            # 1. Gọi Smart Money
            sm_result = self.sm_engine.analyze_ticker(ticker, target_date=sig_date)
            
            # 2. Gọi Shadow Profiler
            shadow_msg = ""
            try:
                rules = self.profiler.build_criminal_profile([ticker], lookback_days=250)
                alerts = self.profiler.live_shadow_radar([ticker], rules, target_date=sig_date)
                if alerts and isinstance(alerts, list):
                    shadow_msg = alerts[0].get('Status', '')
            except: pass

            # 3. Tổng hợp Phán quyết
            if sm_result.get("is_danger", False):
                status = "🚫 BỊ CHẶN (Engine Đỏ)"
                reason = " | ".join(sm_result.get("warnings", []))
            elif sm_result.get("total_sm_score", 0) < 0:
                status = "⚠️ CẢNH BÁO (Điểm Âm)"
                reason = "Tay to không ủng hộ"
            else:
                status = "✅ UY TÍN (Cấp quyền mua)"
                reason = f"SM Điểm: {sm_result.get('total_sm_score')} | Lái nội: {shadow_msg[:20]}"
                
            print(f"{sig_date:<15} | {sig['signal']:<10} | {sig['price']:>10,.0f} | {status:<25} | {reason}")
        print("="*100)

# ==========================================
# KHỐI CHẠY KIỂM TOÁN TÍN HIỆU TOÀN DIỆN
# ==========================================
if __name__ == "__main__":
    DATA_DIR = 'data/parquet'
    inspector = SignalInspector(data_dir=DATA_DIR, universe='VNMidCap', lookback_days=60)
    
    # [Giai đoạn 1]: Lọc ra Top 30 Winners & Top 30 Losers
    winners, losers = inspector.scan_momentum(top_n=20)
    
    # [Giai đoạn 2]: Bật Cỗ máy thời gian kiểm toán Top Winners (Tấn công)
    if winners is not None:
        audit_win_report = inspector.audit_winners(winners)
        
    # [Giai đoạn 3]: Rà soát lỗi False Positives trên Top Losers (Phòng thủ)
    if losers is not None:
        audit_lose_report = inspector.audit_losers(losers)
        
        # [Giai đoạn 4]: Mang Tội đồ đi đối chất với Smart Money
        inspector.cross_audit_traps(audit_lose_report)

    # 🎯 KÍCH HOẠT CHẾ ĐỘ BẮN TỈA: Soi mã anh muốn kiểm tra
    inspector.inspect_single_ticker(ticker='FTS')

