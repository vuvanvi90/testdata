import pandas as pd
import os
from datetime import datetime

class CashFlowReporter:
    def __init__(self, foreign_df, prop_df):
        """
        Khởi tạo Reporter chỉ với 2 file Dòng tiền (Không cần Master Price nữa)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp dữ liệu Dòng tiền cho Reporter...")
        self.df_foreign = foreign_df 
        self.df_prop = prop_df
        
        # Chuẩn hóa thời gian
        for df in [self.df_foreign, self.df_prop]:
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()

    def generate_report(self, timeframe='week', target_date=None):
        """
        Tạo báo cáo Mua/Bán ròng dựa trên Value thực tế.
        :param timeframe: 'day' (1 phiên), 'week' (5 phiên), 'month' (20 phiên)
        :param target_date: Ngày kết thúc báo cáo (Format: 'YYYY-MM-DD'). Nếu None, lấy ngày mới nhất.
        """
        if self.df_foreign.empty and self.df_prop.empty:
            print("[!] Thiếu dữ liệu để tạo báo cáo.")
            return

        # 1. CẤU HÌNH THỜI GIAN
        if timeframe == 'day':
            days, title_prefix = 1, "BÁO CÁO DÒNG TIỀN: 1 PHIÊN"
        elif timeframe == 'week':
            days, title_prefix = 5, "BÁO CÁO DÒNG TIỀN: 1 TUẦN (5 PHIÊN)"
        elif timeframe == 'month':
            days, title_prefix = 20, "BÁO CÁO DÒNG TIỀN: 1 THÁNG (20 PHIÊN)"
        else:
            days, title_prefix = 5, "BÁO CÁO DÒNG TIỀN"

        # Lấy danh sách toàn bộ các ngày có giao dịch trên thị trường
        if not self.df_foreign.empty:
            trading_days = sorted(self.df_foreign['time'].dropna().unique())
        else:
            trading_days = sorted(self.df_prop['time'].dropna().unique())

        if not trading_days: return

        # ---------------------------------------------------------------------
        # 🕰️ XỬ LÝ "CỖ MÁY THỜI GIAN" (TARGET DATE)
        # ---------------------------------------------------------------------
        if target_date:
            target_dt = pd.to_datetime(target_date).normalize()
            # Lọc ra các ngày giao dịch xảy ra TRƯỚC hoặc ĐÚNG VÀO target_date
            valid_days = [d for d in trading_days if d <= target_dt]
            
            if not valid_days:
                print(f"\n[!] Không có dữ liệu giao dịch nào trước hoặc trong ngày {target_date}.")
                return
                
            latest_date = valid_days[-1] # Lấy ngày giao dịch hợp lệ gần nhất
            
            # Cảnh báo nếu ngày truyền vào là cuối tuần/ngày lễ
            if latest_date != target_dt:
                print(f"\n[*] Ngày {target_date} không có giao dịch. Tự động lùi về phiên gần nhất: {latest_date.strftime('%Y-%m-%d')}")
        else:
            latest_date = pd.to_datetime(trading_days[-1])

        # Tìm ngày bắt đầu (start_date) bằng cách lùi lại đúng số 'days' phiên giao dịch
        idx = trading_days.index(latest_date)
        start_idx = max(0, idx - days + 1)
        start_date = trading_days[start_idx]

        # 2. LỌC VÀ TÍNH TỔNG (SUM) THEO KHUNG THỜI GIAN ĐÃ CHỌN
        # Khóa chặn dữ liệu từ start_date đến latest_date
        df_f_agg = pd.DataFrame()
        if not self.df_foreign.empty:
            df_f = self.df_foreign[(self.df_foreign['time'] >= start_date) & (self.df_foreign['time'] <= latest_date)]
            df_f_agg = df_f.groupby('ticker')['foreign_net_value'].sum().reset_index()
        
        df_pr_agg = pd.DataFrame()
        if not self.df_prop.empty:
            df_pr = self.df_prop[(self.df_prop['time'] >= start_date) & (self.df_prop['time'] <= latest_date)]
            df_pr_agg = df_pr.groupby('ticker')['prop_net_value'].sum().reset_index()

        # 3. HỢP NHẤT DỮ LIỆU KHỐI NGOẠI & TỰ DOANH
        if not df_f_agg.empty and not df_pr_agg.empty:
            report_df = pd.merge(df_f_agg, df_pr_agg, on='ticker', how='outer').fillna(0)
        elif not df_f_agg.empty:
            report_df = df_f_agg.copy()
            report_df['prop_net_value'] = 0
        elif not df_pr_agg.empty:
            report_df = df_pr_agg.copy()
            report_df['foreign_net_value'] = 0
        else:
            return

        # 4. QUY ĐỔI RA ĐƠN VỊ TỶ VNĐ
        DIVISOR = 1_000_000_000 
        report_df['foreign_val_bn'] = report_df['foreign_net_value'] / DIVISOR
        report_df['prop_val_bn'] = report_df['prop_net_value'] / DIVISOR
        report_df['total_val_bn'] = report_df['foreign_val_bn'] + report_df['prop_val_bn']

        # 5. SẮP XẾP TỪ THẤP ĐẾN CAO
        report_df = report_df.sort_values(by='total_val_bn', ascending=True).reset_index(drop=True)
        report_df = report_df[report_df['total_val_bn'] != 0]

        # 6. IN RA TERMINAL
        print("\n" + "="*65)
        print(f" 📊 {title_prefix}")
        print(f"    Giai đoạn: {start_date.strftime('%d/%m/%Y')} -> {latest_date.strftime('%d/%m/%Y')}")
        print("    Sắp xếp  : Bán ròng mạnh nhất -> Mua ròng mạnh nhất")
        print("    Đơn vị   : Tỷ VNĐ (Dữ liệu Giá trị Thực tế)")
        print("="*65)
        print(f"{'MÃ CP':<8} | {'KHỐI NGOẠI':>12} | {'TỰ DOANH':>12} | {'TỔNG CỘNG':>12}")
        print("-" * 65)
        
        print("🔻 TOP BÁN RÒNG MẠNH NHẤT:")
        for _, row in report_df.head(15).iterrows():
            print(f"{row['ticker']:<8} | {row['foreign_val_bn']:>12.2f} | {row['prop_val_bn']:>12.2f} | {row['total_val_bn']:>12.2f}")
            
        print("-" * 65)
        print("... (Các mã ở giữa) ...")
        print("-" * 65)
        
        print("🟢 TOP MUA RÒNG MẠNH NHẤT:")
        for _, row in report_df.tail(15).iterrows():
            print(f"{row['ticker']:<8} | {row['foreign_val_bn']:>12.2f} | {row['prop_val_bn']:>12.2f} | {row['total_val_bn']:>12.2f}")
        print("="*65)

        # 7. XUẤT FILE CSV
        os.makedirs('data/reports', exist_ok=True)
        # Gắn thêm khoảng thời gian vào tên file để dễ quản lý
        out_csv = f"data/reports/cf_report_{timeframe}_{latest_date.strftime('%Y%m%d')}.csv"
        
        export_df = report_df.rename(columns={
            'ticker': 'Ma_CP',
            'foreign_val_bn': 'KhoiNgoai_TyDong',
            'prop_val_bn': 'TuDoanh_TyDong',
            'total_val_bn': 'Tong_TyDong'
        })[['Ma_CP', 'KhoiNgoai_TyDong', 'TuDoanh_TyDong', 'Tong_TyDong']]
        
        export_df = export_df.round(2)
        export_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"[💾] Đã xuất {len(export_df)} mã ra file: {out_csv}")

        return report_df

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    FOREIGN_PATH = 'data/parquet/macro/foreign_flow.parquet'
    PROP_PATH = 'data/parquet/macro/prop_flow.parquet'
    
    reporter = FlowReporter(FOREIGN_PATH, PROP_PATH)
    
    # Kịch bản 1: Không truyền target_date -> Mặc định lấy ngày MỚI NHẤT
    print("\n>>> TEST 1: Lấy dữ liệu mới nhất <<<")
    reporter.generate_report(timeframe='day')
    
    # Kịch bản 2: Truyền một ngày cụ thể trong quá khứ
    print("\n>>> TEST 2: Du hành thời gian về một ngày cụ thể <<<")
    # Ví dụ: Xem lại dòng tiền của 1 tuần tính đến ngày 26/02/2026
    reporter.generate_report(timeframe='week', target_date='2026-02-26')
    
    # Kịch bản 3: Truyền trúng ngày cuối tuần (Hệ thống sẽ tự lùi về Thứ 6)
    print("\n>>> TEST 3: Truyền ngày cuối tuần/ngày nghỉ <<<")
    # Ví dụ: Xem dòng tiền ngày Chủ Nhật 01/03/2026
    reporter.generate_report(timeframe='day', target_date='2026-03-01')