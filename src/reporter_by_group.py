import pandas as pd
import os
from datetime import datetime

class GroupCashFlowReporter:
    def __init__(self, foreign_df, prop_df, industry_df):
        """
        Khởi tạo Reporter với dữ liệu Dòng tiền và Danh mục Ngành (ICB)
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Đang nạp dữ liệu Dòng tiền và Sóng Ngành...")
        self.df_foreign = foreign_df
        self.df_prop = prop_df
        self.df_ind = industry_df
        
        # Chuẩn hóa thời gian
        for df in [self.df_foreign, self.df_prop]:
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()

    def generate_report(self, timeframe='week', target_date=None):
        if self.df_foreign.empty and self.df_prop.empty:
            print("[!] Thiếu dữ liệu để tạo báo cáo.")
            return

        # 1. CẤU HÌNH THỜI GIAN (Giữ nguyên logic "Cỗ máy thời gian" cũ)
        if timeframe == 'day': days, title_prefix = 1, "BÁO CÁO DÒNG TIỀN: 1 PHIÊN"
        elif timeframe == 'week': days, title_prefix = 5, "BÁO CÁO DÒNG TIỀN: 1 TUẦN (5 PHIÊN)"
        elif timeframe == 'month': days, title_prefix = 20, "BÁO CÁO DÒNG TIỀN: 1 THÁNG (20 PHIÊN)"
        else: days, title_prefix = 5, "BÁO CÁO DÒNG TIỀN"

        if not self.df_foreign.empty:
            trading_days = sorted(self.df_foreign['time'].dropna().unique())
        else:
            trading_days = sorted(self.df_prop['time'].dropna().unique())

        if not trading_days: return

        if target_date:
            target_dt = pd.to_datetime(target_date).normalize()
            valid_days = [d for d in trading_days if d <= target_dt]
            if not valid_days: return
            latest_date = valid_days[-1]
        else:
            latest_date = pd.to_datetime(trading_days[-1])

        idx = trading_days.index(latest_date)
        start_idx = max(0, idx - days + 1)
        start_date = trading_days[start_idx]

        # 2. LỌC VÀ TÍNH TỔNG (SUM) THEO KHUNG THỜI GIAN
        df_f_agg = pd.DataFrame()
        if not self.df_foreign.empty:
            df_f = self.df_foreign[(self.df_foreign['time'] >= start_date) & (self.df_foreign['time'] <= latest_date)]
            df_f_agg = df_f.groupby('ticker')['foreign_net_value'].sum().reset_index()
        
        df_pr_agg = pd.DataFrame()
        if not self.df_prop.empty:
            df_pr = self.df_prop[(self.df_prop['time'] >= start_date) & (self.df_prop['time'] <= latest_date)]
            df_pr_agg = df_pr.groupby('ticker')['prop_net_value'].sum().reset_index()

        # 3. HỢP NHẤT DỮ LIỆU
        if not df_f_agg.empty and not df_pr_agg.empty:
            report_df = pd.merge(df_f_agg, df_pr_agg, on='ticker', how='outer').fillna(0)
        elif not df_f_agg.empty:
            report_df = df_f_agg.copy(); report_df['prop_net_value'] = 0
        elif not df_pr_agg.empty:
            report_df = df_pr_agg.copy(); report_df['foreign_net_value'] = 0
        else: return

        DIVISOR = 1_000_000_000 
        report_df['foreign_val_bn'] = report_df['foreign_net_value'] / DIVISOR
        report_df['prop_val_bn'] = report_df['prop_net_value'] / DIVISOR
        report_df['total_val_bn'] = report_df['foreign_val_bn'] + report_df['prop_val_bn']

        # ---------------------------------------------------------------------
        # 🎯 MAP DỮ LIỆU NGÀNH (ICB MAPPING)
        # ---------------------------------------------------------------------
        if not self.df_ind.empty and 'symbol' in self.df_ind.columns:
            # Ưu tiên lấy ngành cấp 2 (icb_name2) để nhìn mảng lớn (Bất động sản, Ngân hàng...)
            # Nếu trống, lấy ngành cấp 3 hoặc 4
            self.df_ind['industry'] = self.df_ind['icb_name2'].fillna(self.df_ind['icb_name3']).fillna('Khác')
            ind_mapping = self.df_ind[['symbol', 'industry']].rename(columns={'symbol': 'ticker'})
            # Tránh duplicate
            ind_mapping = ind_mapping.drop_duplicates(subset=['ticker']) 
            
            report_df = pd.merge(report_df, ind_mapping, on='ticker', how='left')
            report_df['industry'] = report_df['industry'].fillna('Chưa phân loại')
        else:
            report_df['industry'] = 'Chưa phân loại'

        # ---------------------------------------------------------------------
        # 📊 TỔNG HỢP THEO SÓNG NGÀNH (SECTOR FLOW)
        # ---------------------------------------------------------------------
        sector_flow = report_df.groupby('industry').agg({
            'foreign_val_bn': 'sum',
            'prop_val_bn': 'sum',
            'total_val_bn': 'sum'
        }).reset_index()
        
        sector_flow = sector_flow.sort_values(by='total_val_bn', ascending=False).reset_index(drop=True)
        # Bỏ qua những ngành giao dịch quá bé (Gần 0)
        sector_flow = sector_flow[sector_flow['total_val_bn'].abs() > 0.5] 

        report_df = report_df.sort_values(by='total_val_bn', ascending=True).reset_index(drop=True)
        report_df = report_df[report_df['total_val_bn'] != 0]

        # ---------------------------------------------------------------------
        # 🖨️ IN BÁO CÁO CẤP CAO RA TERMINAL
        # ---------------------------------------------------------------------
        print("\n" + "="*75)
        print(f" 🏭 BỨC TRANH DÒNG TIỀN VĨ MÔ: {title_prefix}")
        print(f"    Giai đoạn: {start_date.strftime('%d/%m/%Y')} -> {latest_date.strftime('%d/%m/%Y')}")
        print("="*75)
        
        # IN TOP NGÀNH HÚT TIỀN (SÓNG NGÀNH)
        print("🌊 TOP NGÀNH ĐƯỢC TAY TO GOM RÒNG MẠNH NHẤT:")
        print(f"{'TÊN NGÀNH':<30} | {'KHỐI NGOẠI':>10} | {'TỰ DOANH':>10} | {'TỔNG CỘNG (TỶ)':>14}")
        print("-" * 75)
        for _, row in sector_flow.head(5).iterrows():
            if row['total_val_bn'] > 0:
                print(f"{str(row['industry'])[:28]:<30} | {row['foreign_val_bn']:>10.1f} | {row['prop_val_bn']:>10.1f} | {row['total_val_bn']:>14.1f}")
                
        # IN TOP NGÀNH BỊ RÚT TIỀN
        print("\n🩸 TOP NGÀNH BỊ TAY TO XẢ RÒNG RÁT NHẤT:")
        print("-" * 75)
        # Quét từ dưới lên của bảng sector_flow
        for _, row in sector_flow.tail(5).iloc[::-1].iterrows():
            if row['total_val_bn'] < 0:
                print(f"{str(row['industry'])[:28]:<30} | {row['foreign_val_bn']:>10.1f} | {row['prop_val_bn']:>10.1f} | {row['total_val_bn']:>14.1f}")

        print("\n" + "="*75)
        print(" 🎯 TOP CỔ PHIẾU TRỌNG ĐIỂM CHI PHỐI DÒNG TIỀN")
        print("="*75)
        print(f"{'MÃ CP':<6} | {'NGÀNH':<18} | {'TỔNG CỘNG (TỶ)':>14}")
        print("-" * 75)
        
        print("🔻 TOP BỊ BÁN RÒNG:")
        for _, row in report_df.head(10).iterrows():
            print(f"{row['ticker']:<6} | {str(row['industry'])[:16]:<18} | {row['total_val_bn']:>14.2f}")
            
        print("\n🟢 TOP ĐƯỢC MUA RÒNG:")
        for _, row in report_df.tail(10).iloc[::-1].iterrows(): # Sắp xếp lượng gom từ cao xuống thấp cho dễ đọc
            print(f"{row['ticker']:<6} | {str(row['industry'])[:16]:<18} | {row['total_val_bn']:>14.2f}")
        print("="*75)

        # 7. XUẤT FILE CSV
        os.makedirs('data/reports', exist_ok=True)
        out_csv = f"data/reports/cfg_report_{timeframe}_{latest_date.strftime('%Y%m%d')}.csv"
        
        export_df = report_df.rename(columns={
            'ticker': 'Ma_CP',
            'industry': 'Nganh_Nghe',
            'foreign_val_bn': 'KhoiNgoai_TyDong',
            'prop_val_bn': 'TuDoanh_TyDong',
            'total_val_bn': 'Tong_TyDong'
        })[['Ma_CP', 'Nganh_Nghe', 'KhoiNgoai_TyDong', 'TuDoanh_TyDong', 'Tong_TyDong']]
        
        export_df = export_df.sort_values('Tong_TyDong', ascending=False).round(2)
        export_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"[💾] Đã xuất chi tiết {len(export_df)} mã ra file: {out_csv}")

        return sector_flow, report_df

# ==========================================
# KHỐI CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    FOREIGN_PATH = 'data/parquet/macro/foreign_flow.parquet'
    PROP_PATH = 'data/parquet/macro/prop_flow.parquet'
    IND_PATH = 'data/parquet/macro/groups_by_industries.parquet' # File Sóng Ngành của anh
    
    reporter = GroupCashFlowReporter(FOREIGN_PATH, PROP_PATH, IND_PATH)
    
    # In báo cáo dòng tiền theo Tuần (5 phiên gần nhất)
    reporter.generate_report(timeframe='week')