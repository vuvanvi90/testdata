import pandas as pd
import shutil
import os
from datetime import datetime
from pathlib import Path

from src.forecaster import WyckoffForecaster

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

class SignalInspector:
    def __init__(self, df_price, df_foreign, df_prop):
        """
        Khởi tạo Hệ thống Kiểm toán Tín hiệu (Signal Inspector)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Khởi động Hệ thống Kiểm toán (Signal Inspector)...")
        self.df_price = df_price
        self.df_foreign = df_foreign
        self.df_prop = df_prop
        
        if not self.df_price.empty:
            self.df_price['time'] = pd.to_datetime(self.df_price['time']).dt.normalize()
            # Sắp xếp chuẩn để tính toán chuỗi thời gian
            self.df_price = self.df_price.sort_values(by=['ticker', 'time'])

    def scan_momentum(self, top_n=30, min_liquidity_bn=3.0):
        """
        Giai đoạn 1: Quét Động lượng (Momentum Scanner)
        Tìm Top Winners và Top Losers dựa trên động lượng 3 tháng qua.
        """
        print("\n" + "="*75)
        print(" 🔍 GIAI ĐOẠN 1: QUÉT ĐỘNG LƯỢNG (MOMENTUM SCANNER)")
        print("="*75)
        
        if self.df_price.empty:
            print("[!] Không có dữ liệu giá trong master_price.parquet.")
            return None, None

        latest_date = self.df_price['time'].max()
        print(f"[*] Ngày phân tích gần nhất: {latest_date.strftime('%Y-%m-%d')}")
        print(f"[*] Đang tính toán hiệu suất 1D, 1W, 1M, 3M cho toàn thị trường...")

        # Lấy 65 phiên gần nhất của mỗi mã (Đủ để tính lùi 60 phiên ~ 3 tháng)
        df_recent = self.df_price.groupby('ticker').tail(65).copy()
        
        results = []
        
        for ticker, group in df_recent.groupby('ticker'):
            if len(group) < 61:
                continue # Bỏ qua các mã mới lên sàn chưa đủ 3 tháng dữ liệu
                
            group = group.sort_values('time').reset_index(drop=True)
            
            # Tính thanh khoản trung bình 20 phiên gần nhất (Đơn vị: Tỷ VNĐ)
            avg_vol = group['volume'].tail(20).mean()
            avg_price = group['close'].tail(20).mean()
            liquidity_bn = (avg_vol * avg_price) / 1_000_000_000
            
            # Bộ lọc thanh khoản: Bỏ qua hàng rác, bo cung
            if liquidity_bn < min_liquidity_bn:
                continue
                
            # Trích xuất giá đóng cửa tại các mốc thời gian (Tính từ cuối lên)
            current_close = group['close'].iloc[-1]
            close_1d = group['close'].iloc[-2]   # 1 ngày trước
            close_1w = group['close'].iloc[-6]   # 1 tuần trước (5 phiên)
            close_1m = group['close'].iloc[-21]  # 1 tháng trước (20 phiên)
            close_3m = group['close'].iloc[-61]  # 3 tháng trước (60 phiên)
            
            # Tính Tỷ suất sinh lời (%)
            ret_1d = (current_close - close_1d) / close_1d * 100
            ret_1w = (current_close - close_1w) / close_1w * 100
            ret_1m = (current_close - close_1m) / close_1m * 100
            ret_3m = (current_close - close_3m) / close_3m * 100
            
            results.append({
                'Ticker': ticker,
                'Close': current_close,
                'Liquidity_Bn': liquidity_bn,
                'Ret_1D': ret_1d,
                'Ret_1W': ret_1w,
                'Ret_1M': ret_1m,
                'Ret_3M': ret_3m
            })
            
        df_mom = pd.DataFrame(results)
        
        # ---------------------------------------------------------------------
        # Phân loại Winners và Losers dựa trên hiệu suất 3 Tháng (Ret_3M)
        # ---------------------------------------------------------------------
        df_winners = df_mom.nlargest(top_n, 'Ret_3M').reset_index(drop=True)
        df_losers = df_mom.nsmallest(top_n, 'Ret_3M').reset_index(drop=True)
        
        # In Báo cáo ra Terminal
        print(f"\n🏆 TOP {top_n} SIÊU CỔ PHIẾU (WINNERS - Tăng mạnh nhất 3 Tháng qua):")
        print(f"{'MÃ CP':<8} | {'GIÁ HIỆN TẠI':<15} | {'1 NGÀY':>8} | {'1 TUẦN':>8} | {'1 THÁNG':>8} | {'3 THÁNG':>8}")
        print("-" * 75)
        for _, r in df_winners.head(30).iterrows(): # In 15 mã đại diện
            print(f"{r['Ticker']:<8} | {r['Close']:<15,.0f} | {r['Ret_1D']:>7.1f}% | {r['Ret_1W']:>7.1f}% | {r['Ret_1M']:>7.1f}% | {r['Ret_3M']:>7.1f}%")
            
        print(f"\n🩸 TOP {top_n} TỘI ĐỒ (LOSERS - Giảm mạnh nhất 3 Tháng qua):")
        print("-" * 75)
        for _, r in df_losers.head(30).iterrows():
            print(f"{r['Ticker']:<8} | {r['Close']:<15,.0f} | {r['Ret_1D']:>7.1f}% | {r['Ret_1W']:>7.1f}% | {r['Ret_1M']:>7.1f}% | {r['Ret_3M']:>7.1f}%")
            
        return df_winners, df_losers

    def audit_winners(self, df_winners, lookback_days=60):
        """
        Giai đoạn 2: Cỗ máy Thời gian (Walk-Forward Audit)
        Tích hợp WyckoffForecaster thật để quét ngược quá khứ 3 tháng.
        """
        print("\n" + "="*75)
        print(" 🕰️ GIAI ĐOẠN 2: CỖ MÁY THỜI GIAN (TÍCH HỢP WYCKOFF FORECASTER CHUẨN)")
        print("="*75)
        
        if df_winners is None or df_winners.empty:
            print("[!] Không có dữ liệu Winners để kiểm toán.")
            return
            
        tickers_to_audit = df_winners['Ticker'].tolist()
        
        # 1. Lấy danh sách N ngày giao dịch gần nhất
        trading_days = sorted(self.df_price['time'].unique())
        audit_days = trading_days[-lookback_days:] if len(trading_days) >= lookback_days else trading_days

        # Tạo không gian làm việc tạm thời
        audit_temp_dir = Path('data/temp_audit')
        audit_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_price_path = audit_temp_dir / 'audit_price.parquet'

        # Dictionary lưu trữ tín hiệu nổ của từng mã: { 'FTS': [{'date': '...', 'signal': 'SOS'}], ... }
        signals_dict = {t: [] for t in tickers_to_audit}

        print(f"[*] Kích hoạt Cỗ máy: Quét lùi {len(audit_days)} phiên giao dịch cho {len(tickers_to_audit)} Siêu cổ phiếu...")
        
        # 2. VÒNG LẶP DU HÀNH THỜI GIAN (WALK-FORWARD)
        for target_date in tqdm(audit_days, desc="Tiến trình Audit"):
            # Cắt dữ liệu từ mốc sơ khai đến đúng ngày target_date để Forecaster không "nhìn trộm" tương lai
            mask = (self.df_price['ticker'].isin(tickers_to_audit)) & (self.df_price['time'] <= target_date)
            df_slice = self.df_price[mask]
            
            if df_slice.empty: continue
            
            # Lưu ra Parquet tạm cho Forecaster đọc
            df_slice.to_parquet(temp_price_path, engine='pyarrow')
            
            # Khởi chạy Forecaster thật tại đúng mốc thời gian đó
            try:
                forecaster = WyckoffForecaster(
                    price_dir=temp_price_path, 
                    output_dir=audit_temp_dir, 
                    run_date=target_date
                )
                report = forecaster.run_forecast()
                
                # Nếu có báo cáo trả về, lọc tìm tín hiệu Mua
                if not report.empty:
                    buy_signals = report[report['Signal'].isin(['SOS', 'SPRING'])]
                    
                    for _, row in buy_signals.iterrows():
                        ticker = row['Ticker']
                        if ticker in signals_dict:
                            signals_dict[ticker].append({
                                'date': target_date.strftime('%Y-%m-%d'),
                                'price': row['Price'],
                                'signal': row['Signal'],
                                'vpa': row.get('VPA_Status', 'N/A')
                            })
            except Exception as e:
                # Bỏ qua lỗi ngầm nếu có 1 ngày dữ liệu bị hỏng
                pass
                
        # Dọn dẹp chiến trường
        try: shutil.rmtree(audit_temp_dir)
        except: pass

        # 3. CHẨN ĐOÁN LỖI (DIAGNOSIS) TỪ DỮ LIỆU ĐÃ QUÉT
        audit_results = []
        for ticker in tickers_to_audit:
            ret_3m = df_winners[df_winners['Ticker'] == ticker]['Ret_3M'].values[0]
            current_price = df_winners[df_winners['Ticker'] == ticker]['Close'].values[0]
            
            fired_list = signals_dict[ticker]
            
            if len(fired_list) > 0:
                best_signal = fired_list[0] # Bắt tín hiệu Mua ĐẦU TIÊN (Chân sóng)
                caught_price = best_signal['price']
                profit_if_caught = (current_price - caught_price) / caught_price * 100
                
                diagnosis = "🟢 CAUGHT (Bắt trúng chân sóng)"
                note = f"Nổ {best_signal['signal']} ngày {best_signal['date']} (Giá: {caught_price:,.0f} -> Biên lãi: +{profit_if_caught:.1f}%) | VPA: {best_signal['vpa']}"
            else:
                diagnosis = "🔴 BLIND SPOT (Điểm mù Wyckoff)"
                note = "Lái kéo thốc/Bo cung. Không có cụm nến Wyckoff hoặc VPA đủ chuẩn."
                
            audit_results.append({
                'Ticker': ticker,
                'Ret_3M': ret_3m,
                'Signals_Count': len(fired_list),
                'Diagnosis': diagnosis,
                'Note': note
            })
            
        # 4. TỔNG HỢP VÀ IN BÁO CÁO
        df_audit = pd.DataFrame(audit_results)
        
        print("\n📊 BÁO CÁO KIỂM TOÁN TÍN HIỆU (TOP WINNERS):")
        print(f"{'MÃ CP':<8} | {'TĂNG 3T':>8} | {'SỐ LẦN NỔ LỆNH':>14} | {'CHẨN ĐOÁN HỆ THỐNG':<30}")
        print("-" * 75)
        
        for _, r in df_audit.iterrows():
            print(f"{r['Ticker']:<8} | {r['Ret_3M']:>7.1f}% | {r['Signals_Count']:>14} | {r['Diagnosis']:<30}")
            print(f"   └─ Dấu vết: {r['Note']}")
            
        caught_count = len(df_audit[df_audit['Diagnosis'].str.contains('CAUGHT')])
        blind_count = len(df_audit[df_audit['Diagnosis'].str.contains('BLIND SPOT')])
        coverage_pct = (caught_count / len(df_audit)) * 100 if len(df_audit) > 0 else 0
        
        print("\n" + "="*75)
        print(f"🎯 ĐỘ PHỦ WYCKOFF: Thuật toán quét trúng {caught_count}/{len(df_audit)} siêu cổ phiếu ({coverage_pct:.1f}%).")
        if blind_count > 0:
            print(f"🙈 ĐIỂM MÙ KỸ THUẬT: Bỏ lỡ {blind_count} mã do lái đánh không thanh khoản hoặc lọt khe mẫu hình Wyckoff.")
        print("="*75)
        
        return df_audit

    def audit_losers(self, df_losers, lookback_days=60):
        """
        Giai đoạn 3: Rà soát Bẫy (False Positives Audit)
        Kiểm tra xem hệ thống có bắt đáy sai (bắt dao rơi) đối với các mã giảm sâu không.
        """
        import shutil
        from pathlib import Path
        from src.forecaster import WyckoffForecaster
        
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        print("\n" + "="*75)
        print(" 🚨 GIAI ĐOẠN 3: RÀ SOÁT BẪY KÉO XẢ (FALSE POSITIVES AUDIT)")
        print("="*75)
        
        if df_losers is None or df_losers.empty:
            print("[!] Không có dữ liệu Losers để kiểm toán.")
            return
            
        tickers_to_audit = df_losers['Ticker'].tolist()
        
        # 1. Lấy danh sách N ngày giao dịch gần nhất
        trading_days = sorted(self.df_price['time'].unique())
        audit_days = trading_days[-lookback_days:] if len(trading_days) >= lookback_days else trading_days

        # Không gian làm việc cho Losers
        audit_temp_dir = Path('data/temp_audit_losers')
        audit_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_price_path = audit_temp_dir / 'audit_price.parquet'

        signals_dict = {t: [] for t in tickers_to_audit}

        print(f"[*] Đang thử thách Cỗ máy: Quét lùi {len(audit_days)} phiên trên {len(tickers_to_audit)} Tội đồ (Losers)...")
        
        # 2. VÒNG LẶP DU HÀNH THỜI GIAN
        for target_date in tqdm(audit_days, desc="Tiến trình Audit Losers"):
            mask = (self.df_price['ticker'].isin(tickers_to_audit)) & (self.df_price['time'] <= target_date)
            df_slice = self.df_price[mask]
            
            if df_slice.empty: continue
            
            df_slice.to_parquet(temp_price_path, engine='pyarrow')
            
            try:
                forecaster = WyckoffForecaster(
                    price_dir=temp_price_path, 
                    output_dir=audit_temp_dir, 
                    run_date=target_date
                )
                report = forecaster.run_forecast()
                
                if not report.empty:
                    buy_signals = report[report['Signal'].isin(['SOS', 'SPRING'])]
                    for _, row in buy_signals.iterrows():
                        ticker = row['Ticker']
                        if ticker in signals_dict:
                            signals_dict[ticker].append({
                                'date': target_date.strftime('%Y-%m-%d'),
                                'price': row['Price'],
                                'signal': row['Signal']
                            })
            except Exception as e:
                pass
                
        try: shutil.rmtree(audit_temp_dir)
        except: pass

        # 3. CHẨN ĐOÁN LỖI BẮT DAO RƠI
        audit_results = []
        for ticker in tickers_to_audit:
            ret_3m = df_losers[df_losers['Ticker'] == ticker]['Ret_3M'].values[0]
            fired_list = signals_dict[ticker]
            
            if len(fired_list) == 0:
                diagnosis = "🛡️ SAFE (Miễn nhiễm)"
                note = "Hệ thống đứng ngoài thành công, không bị lừa bắt đáy."
            else:
                # Bị lừa ít nhất 1 lần
                worst_signal = fired_list[0]
                diagnosis = "🩸 TRAP (Dính bẫy bắt dao rơi)"
                note = f"Phát lệnh {worst_signal['signal']} sai lầm vào ngày {worst_signal['date']} (Giá: {worst_signal['price']:,.0f}đ)"
                
            audit_results.append({
                'Ticker': ticker,
                'Ret_3M': ret_3m,
                'Trap_Count': len(fired_list),
                'Diagnosis': diagnosis,
                'Note': note
            })
            
        df_audit = pd.DataFrame(audit_results)
        
        print("\n📊 BÁO CÁO KIỂM TOÁN LỖI (TOP LOSERS):")
        print(f"{'MÃ CP':<8} | {'GIẢM 3T':>8} | {'SỐ LẦN DÍNH BẪY':>15} | {'CHẨN ĐOÁN HỆ THỐNG':<30}")
        print("-" * 75)
        
        for _, r in df_audit.iterrows():
            print(f"{r['Ticker']:<8} | {r['Ret_3M']:>7.1f}% | {r['Trap_Count']:>15} | {r['Diagnosis']:<30}")
            print(f"   └─ Dấu vết: {r['Note']}")
            
        safe_count = len(df_audit[df_audit['Diagnosis'].str.contains('SAFE')])
        trap_count = len(df_audit[df_audit['Diagnosis'].str.contains('TRAP')])
        safe_pct = (safe_count / len(df_audit)) * 100 if len(df_audit) > 0 else 0
        
        print("\n" + "="*75)
        print(f"🛡️ ĐỘ PHÒNG THỦ: Cỗ máy né được {safe_count}/{len(df_audit)} bẫy giảm giá ({safe_pct:.1f}%).")
        if trap_count > 0:
            print(f"⚠️ CẢNH BÁO BẪY: Có {trap_count} mã khiến WyckoffForecaster phát lệnh ảo.")
        print("="*75)
        
        return df_audit

    def cross_audit_traps(self, df_audit_losers):
        """
        Giai đoạn 4: Kiểm toán Chéo (Cross-Audit Traps)
        Đối chiếu các lệnh TRAP với dữ liệu Dòng tiền Thông minh (Smart Money).
        """
        import re

        print("\n" + "="*80)
        print(" 🕵️ GIAI ĐOẠN 4: KIỂM TOÁN CHÉO (CROSS-AUDIT TRAPS VỚI SMART MONEY)")
        print("="*80)

        # Lọc ra các mã bị dính TRAP
        df_traps = df_audit_losers[df_audit_losers['Diagnosis'].str.contains('TRAP')].copy()

        if df_traps.empty:
            print("[*] Không có bẫy nào để kiểm toán chéo. Hệ thống phòng thủ 100%!")
            return

        print(f"[*] Đang đối chiếu {len(df_traps)} lệnh TRAP với Hành vi của Khối ngoại & Tự doanh...")

        # Nạp dữ liệu dòng tiền
        df_foreign = self.df_foreign
        df_prop = self.df_prop

        # Chuẩn hóa thời gian
        if not df_foreign.empty:
            df_foreign['time'] = pd.to_datetime(df_foreign['time']).dt.normalize()
        if not df_prop.empty:
            df_prop['time'] = pd.to_datetime(df_prop['time']).dt.normalize()

        blocked_count = 0
        results = []

        for _, row in df_traps.iterrows():
            ticker = row['Ticker']
            note = row['Note']

            # Dùng Regex để bóc tách ngày tháng từ chuỗi Note (Ví dụ: 2026-02-15)
            match = re.search(r'ngày (\d{4}-\d{2}-\d{2})', note)
            if not match:
                continue

            trap_date_str = match.group(1)
            trap_date = pd.to_datetime(trap_date_str)

            foreign_net = 0
            prop_net = 0

            # Lấy dữ liệu Khối ngoại ngày hôm đó
            if not df_foreign.empty:
                f_row = df_foreign[(df_foreign['ticker'] == ticker) & (df_foreign['time'] == trap_date)]
                if not f_row.empty:
                    # Lấy chính xác cột giá trị mua/bán ròng thực tế
                    foreign_net = f_row.iloc[0].get('foreign_net_value', 0)

            # Lấy dữ liệu Tự doanh ngày hôm đó
            if not df_prop.empty:
                p_row = df_prop[(df_prop['ticker'] == ticker) & (df_prop['time'] == trap_date)]
                if not p_row.empty:
                    prop_net = p_row.iloc[0].get('prop_net_value', 0)

            # Quy đổi ra Tỷ VNĐ (Giả định giá trị gốc là VNĐ)
            total_net = (foreign_net + prop_net) / 1_000_000_000 

            # Áp dụng logic của live.py: Nếu tay to táng hàng (Net < 0) -> Chặn!
            if total_net < 0:
                status = "🛡️ ĐÃ BỊ CHẶN (Tay to xả ròng)"
                blocked_count += 1
            else:
                status = "⚠️ LỌT LƯỚI (Chỉ báo nhiễu)"

            results.append({
                'Ticker': ticker,
                'Trap_Date': trap_date_str,
                'Total_Net_Bn': total_net,
                'Status': status
            })

        # In Báo cáo Đối chất
        print("\n📊 BÁO CÁO ĐỐI CHẤT DÒNG TIỀN (SMART MONEY VS TRAPS):")
        print(f"{'MÃ CP':<8} | {'NGÀY DÍNH BẪY':<15} | {'DÒNG TIỀN TAY TO (TỶ)':>21} | {'KẾT QUẢ ĐỐI CHẤT TRONG LIVE.PY':<30}")
        print("-" * 80)

        for r in results:
            # Format in màu đỏ nếu bị xả, xanh nếu được gom
            flow_str = f"{r['Total_Net_Bn']:>21.2f}"
            print(f"{r['Ticker']:<8} | {r['Trap_Date']:<15} | {flow_str} | {r['Status']:<30}")

        print("\n" + "="*80)
        print(f"🎯 KẾT LUẬN CUỐI CÙNG (FINAL VERDICT):")
        if len(df_traps) > 0:
            print(f"Trong {len(df_traps)} tín hiệu TRAP ảo, Cỗ máy live.py đã TỰ ĐỘNG CHẶN ĐỨNG {blocked_count} lệnh")
            print(f"nhờ phát hiện Tây/Tự doanh xả ròng (Tỷ lệ cản phá thêm: {(blocked_count/len(df_traps))*100:.1f}%).")
        print("="*80)

# ==========================================
# KHỐI CHẠY KIỂM TOÁN TÍN HIỆU TOÀN DIỆN
# ==========================================
if __name__ == "__main__":
    PRICE_PATH = 'data/parquet/price/master_price.parquet'
    FOREIGN_PATH = 'data/parquet/macro/foreign_flow.parquet'
    PROP_PATH = 'data/parquet/macro/prop_flow.parquet'
    
    inspector = SignalInspector(PRICE_PATH)
    
    # [Giai đoạn 1]: Lọc ra Top 30 Winners & Top 30 Losers
    winners, losers = inspector.scan_momentum(top_n=30, min_liquidity_bn=3.0)
    
    # [Giai đoạn 2]: Bật Cỗ máy thời gian kiểm toán Top Winners (Tấn công)
    if winners is not None:
        audit_win_report = inspector.audit_winners(winners, lookback_days=60)
        
    # [Giai đoạn 3]: Rà soát lỗi False Positives trên Top Losers (Phòng thủ)
    if losers is not None:
        audit_lose_report = inspector.audit_losers(losers, lookback_days=60)
        
        # [Giai đoạn 4]: Mang Tội đồ đi đối chất với Smart Money
        inspector.cross_audit_traps(audit_lose_report, FOREIGN_PATH, PROP_PATH)