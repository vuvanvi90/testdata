import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class BlacklistGuard:
    """
    Hệ thống Vệ binh Quản trị Danh Sách Đen (Blacklist)
    Được phân lớp theo Rổ Cổ Phiếu (Universe) và Kế thừa trí tuệ từ SmartMoneyEngine.
    """
    def __init__(self, universe="VN30", verbose=True):
        self.universe = universe
        self.verbose = verbose
        self.bl_path = Path(f"data/live/blacklist_{self.universe.lower()}.json")
        self.bl_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ĐIỀU CHỈNH LUẬT CHƠI THEO TỪNG RỔ (UNIVERSE STRATIFICATION)
        if self.universe == "VN30":
            self.pardon_threshold = 15.0  # Cần ít nhất 15 tỷ T0 để lật kèo ở VN30
        elif self.universe == "VNMidCap":
            self.pardon_threshold = 5.0   # Midcap chỉ cần 5 tỷ
        elif self.universe == "VNSmallCap":
            self.pardon_threshold = 2.0   # Penny chỉ cần 2 tỷ là đủ tạo sóng
        else:
            self.pardon_threshold = 20.0   # cần khóa với dòng tiền lớn
            
    def load(self):
        """Đọc danh sách Đen từ File"""
        if not self.bl_path.exists(): return {}
        try:
            with open(self.bl_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}

    def save(self, bl_dict):
        """Lưu danh sách Đen xuống File"""
        try:
            with open(self.bl_path, 'w', encoding='utf-8') as f: 
                json.dump(bl_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Lỗi khi lưu Blacklist {self.universe}: {e}")

    def evaluate_pardons(self, current_blacklist, board_info_dict, sm_info_dict, foreign_dict, current_date_str):
        """
        Lõi Logic: Quét và ra quyết định Ân Xá (Gỡ Blacklist)
        Sử dụng kết quả từ SmartMoneyEngine và Bảng điện T0.
        """
        tickers_to_remove = []
        current_date = pd.to_datetime(current_date_str).normalize() if current_date_str else pd.Timestamp.now().normalize()

        for t, info in current_blacklist.items():
            date_added = datetime.strptime(info['date_added'], '%Y-%m-%d')
            days_penalty = (datetime.now() - date_added).days
            
            # 1. ÂN XÁ MẶC ĐỊNH (Hết hạn 5 ngày)
            if days_penalty > 5:
                tickers_to_remove.append(t)
                if self.verbose:
                    print(f"   🔓 Hết hạn án treo: {t} gỡ khỏi Blacklist (Đã qua 5 ngày an toàn).")
                continue
                
            # 2. ÂN XÁ ĐẶC BIỆT: WASHOUT REVERSAL
            board_info = board_info_dict.get(t)
            sm_result = sm_info_dict.get(t, {})
            net_f_t0 = board_info.get('net_foreign', 0) if board_info else 0
            
            # Kiểm tra ngưỡng lật kèo theo chuẩn của từng rổ
            if net_f_t0 < self.pardon_threshold or sm_result.get("is_danger", False):
                continue
                
            # Tính toán Tổng Lượng Xả (Total Dump)
            total_dump_bn = 0
            df_f = foreign_dict.get(t)
            
            if df_f is not None and not df_f.empty:
                df_dump = df_f[(df_f['time'] >= pd.to_datetime(info['date_added'])) & (df_f['time'] < current_date)]
                
                if not df_dump.empty and 'foreign_net_value' in df_dump.columns:
                    total_dump_bn = df_dump['foreign_net_value'].sum() / 1_000_000_000
            
            # Kiểm định Tỷ lệ Phủ lấp (Coverage Ratio >= 50%)
            if total_dump_bn < -5.0: 
                coverage_ratio = net_f_t0 / abs(total_dump_bn)
                if coverage_ratio >= 0.5:
                    tickers_to_remove.append(t)
                    if self.verbose:
                        print(f"   👼 ĐẠI XÁ (WASHOUT REVERSAL): {t} gỡ Blacklist! Mua T0 ({net_f_t0:.1f} Tỷ) lấp được {coverage_ratio*100:.0f}% chuỗi xả.")
                else:
                    if self.verbose:
                        print(f"   ⚠️ TỪ CHỐI ÂN XÁ {t}: T0 mua {net_f_t0:.1f} Tỷ (Quá nhỏ so với ngưỡng xả {total_dump_bn:.1f} Tỷ).")
            else:
                tickers_to_remove.append(t)
                if self.verbose:
                    print(f"   👼 ĐẠI XÁ (FRESH BUY): {t} gỡ Blacklist! Tây gom đột biến {net_f_t0:.1f} Tỷ đẩy giá.")

        return tickers_to_remove