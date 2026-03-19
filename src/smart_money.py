import pandas as pd
from datetime import datetime

class SmartMoneyEngine:
    """
    Động cơ Phân tích Dòng tiền Thông minh (Smart Money) Đa Khung Thời Gian
    Bao gồm 4 Lớp: Dài hạn (Hấp thụ) -> Trung hạn (Động lượng) -> Ngắn hạn (Chiến thuật) -> Tức thời (Bảng điện).
    """
    def __init__(self, foreign_dict, prop_dict, out_shares_dict):
        self.foreign_dict = foreign_dict
        self.prop_dict = prop_dict
        self.out_shares_dict = out_shares_dict
        
        # Bộ lọc Hạn sử dụng: Chỉ tính điểm chiến thuật nếu có giao dịch trong 15 ngày qua
        self.MAX_DELAY_DAYS = 15
        self.current_date = pd.Timestamp.now().normalize()

    def analyze_ticker(self, ticker, board_info=None):
        """Phân tích toàn diện và trả về Bản án định lượng cho 1 mã cổ phiếu"""
        
        result = {
            "total_sm_score": 0,
            "is_danger": False,       # Cờ đỏ cảnh báo Xả hàng khẩn cấp
            "sm_details": [],         # Ghi chú để in ra màn hình
            "warnings": [],           # Lý do đưa vào Danger/Blacklist
            "valid_f": False,         # Khối ngoại còn "Hạn sử dụng" không
            "valid_p": False,         # Tự doanh còn "Hạn sử dụng" không
            "last_trade_date": None   # Ngày xả hàng cuối cùng (để tính Án treo)
        }

        df_f = self.foreign_dict.get(ticker)
        df_p = self.prop_dict.get(ticker)

        # Trích xuất số lượng cổ phiếu lưu hành để làm mẫu số
        shares_out = self.out_shares_dict.get(ticker, 0)

        # =====================================================================
        # LỚP 1: TẦM NHÌN DÀI HẠN (STRUCTURAL VIEW - 130 PHIÊN / 6 THÁNG)
        # Mục tiêu: Đo lường Độ kiệt quệ Free-Float (Float Absorption Rate)
        # =====================================================================
        total_net_f_6m = df_f['foreign_net_volume'].sum() if (df_f is not None and not df_f.empty) else 0
        total_net_p_6m = df_p['prop_net_volume'].sum() if (df_p is not None and not df_p.empty) else 0
        total_net_6m = total_net_f_6m + total_net_p_6m
        
        if shares_out > 0:
            absorption_rate = (total_net_6m / shares_out) * 100
            
            if absorption_rate >= 5.0:
                result["total_sm_score"] += 15
                result["sm_details"].append(f"Gom kiệt cung 6M (+{absorption_rate:.1f}%)")
            elif absorption_rate >= 2.0:
                result["total_sm_score"] += 10
                result["sm_details"].append(f"Gom ròng 6M (+{absorption_rate:.1f}%)")
            elif absorption_rate <= -5.0:
                result["total_sm_score"] -= 15
                result["sm_details"].append(f"Xả rát 6M ({absorption_rate:.1f}%)")
            elif absorption_rate <= -2.0:
                result["total_sm_score"] -= 10
                result["sm_details"].append(f"Xả ròng 6M ({absorption_rate:.1f}%)")

        # =====================================================================
        # LỚP 2 & 3: TẦM NHÌN TRUNG HẠN (20 PHIÊN) & NGẮN HẠN (3-5 PHIÊN)
        # =====================================================================
        foreign_base_positive, prop_base_positive = False, False
        foreign_base_negative, prop_base_negative = False, False

        # --- A. PHÂN TÍCH KHỐI NGOẠI ---
        if df_f is not None and not df_f.empty:
            last_date_f = pd.to_datetime(df_f['time'].max())
            if (self.current_date - last_date_f).days <= self.MAX_DELAY_DAYS:
                result["valid_f"] = True
                
                # Cắt lấy 20 phiên gần nhất (Trung hạn)
                df_f_20d = df_f.tail(20)
                total_net_f_20d = df_f_20d['foreign_net_volume'].sum()
                if total_net_f_20d > 0: foreign_base_positive = True
                elif total_net_f_20d < 0: foreign_base_negative = True

                # Cắt lấy 5 phiên gần nhất (Ngắn hạn)
                df_f_5d = df_f_20d.tail(5)
                buy_days_f = len(df_f_5d[df_f_5d['foreign_net_volume'] > 0])
                sell_days_f = len(df_f_5d[df_f_5d['foreign_net_volume'] < 0])
                
                # Chấm điểm Chiến thuật 5 phiên
                if buy_days_f >= 3:
                    result["total_sm_score"] += 5
                    result["sm_details"].append("Tây gom 3/5 phiên (+5)")
                elif sell_days_f >= 3:
                    result["total_sm_score"] -= 5
                    result["sm_details"].append("Tây xả 3/5 phiên (-5)")
                    
                # Cảnh báo Đỏ: Xả rát 3 phiên liên tiếp
                df_f_3d = df_f_20d.tail(3)
                if len(df_f_3d[df_f_3d['foreign_net_volume'] < 0]) == 3:
                    result["is_danger"] = True
                    result["warnings"].append("Tây xả 3 phiên")
                    result["last_trade_date"] = df_f_3d['time'].iloc[-1]
            else:
                result["sm_details"].append("(Tây lãng quên > 15 ngày)")

        # --- B. PHÂN TÍCH TỰ DOANH ---
        if df_p is not None and not df_p.empty:
            last_date_p = pd.to_datetime(df_p['time'].max())
            if (self.current_date - last_date_p).days <= self.MAX_DELAY_DAYS:
                result["valid_p"] = True
                
                # Cắt lấy 20 phiên gần nhất (Trung hạn)
                df_p_20d = df_p.tail(20)
                total_net_p_20d = df_p_20d['prop_net_volume'].sum()
                if total_net_p_20d > 0: prop_base_positive = True
                elif total_net_p_20d < 0: prop_base_negative = True

                # Cắt lấy 5 phiên gần nhất (Ngắn hạn)
                df_p_5d = df_p_20d.tail(5)
                buy_days_p = len(df_p_5d[df_p_5d['prop_net_volume'] > 0])
                sell_days_p = len(df_p_5d[df_p_5d['prop_net_volume'] < 0])
                
                # Chấm điểm Chiến thuật 5 phiên
                if buy_days_p >= 3:
                    result["total_sm_score"] += 5
                    result["sm_details"].append("Tự doanh gom 3/5 phiên (+5)")
                elif sell_days_p >= 3:
                    result["total_sm_score"] -= 5
                    result["sm_details"].append("Tự doanh xả 3/5 phiên (-5)")
                    
                # Cảnh báo Đỏ: Xả rát 3 phiên liên tiếp
                df_p_3d = df_p_20d.tail(3)
                if len(df_p_3d[df_p_3d['prop_net_volume'] < 0]) == 3:
                    result["is_danger"] = True
                    result["warnings"].append("Tự doanh xả 3 phiên")
                    # Lấy ngày xả gần nhất
                    p_last_time = df_p_3d['time'].iloc[-1]
                    if result["last_trade_date"] is None or result["last_trade_date"] < p_last_time:
                        result["last_trade_date"] = p_last_time
            else:
                if "(Tây lãng quên > 15 ngày)" not in result["sm_details"]: 
                    result["sm_details"].append("(Tự doanh lãng quên)")

        # --- C. ĐÁNH GIÁ CỘNG HƯỞNG TRUNG HẠN (DEADLY / SUPER COMBO) ---
        if foreign_base_positive and prop_base_positive:
            result["total_sm_score"] += 10
            result["sm_details"].append("🌟 SUPER COMBO 1M (+10)")
        elif foreign_base_negative and prop_base_negative:
            result["total_sm_score"] -= 10
            result["sm_details"].append("🚨 DEADLY COMBO 1M (-10)")
            # Đưa vào cờ nguy hiểm nếu cả 2 cùng xả ròng
            if result["valid_f"] and result["valid_p"]:
                result["is_danger"] = True
                result["warnings"].append("Deadly Combo Xả")
                result["last_trade_date"] = datetime.now()

        # =====================================================================
        # LỚP 4: TẦM NHÌN TỨC THỜI (BẢNG ĐIỆN INTRADAY)
        # =====================================================================
        if board_info:
            net_f_intraday = board_info.get('net_foreign', 0)
            if net_f_intraday > 0 and result["total_sm_score"] >= 0 and result["total_sm_score"] < 20:
                result["total_sm_score"] = min(result["total_sm_score"] + 5, 20)
                if "Tây Mua Tức Thời" not in " ".join(result["sm_details"]):
                    result["sm_details"].append("Tây Mua Tức Thời (+5)")
            elif net_f_intraday < 0 and result["total_sm_score"] <= 0 and result["total_sm_score"] > -20:
                result["total_sm_score"] = max(result["total_sm_score"] - 5, -20)
                if "Tây xả" not in " ".join(result["sm_details"]): 
                    result["sm_details"].append("Tây Bán Tức Thời (-5)")

        return result