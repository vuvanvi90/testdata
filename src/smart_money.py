import pandas as pd
from datetime import datetime

class SmartMoneyEngine:
    """
    Động cơ Phân tích Dòng tiền Thông minh (Smart Money) Đa Khung Thời Gian
    Bản Nâng Cấp: Tích hợp Đo lường Footprint % và Lái Nội Cân Tây.
    """
    def __init__(self, foreign_dict, prop_dict, out_shares_dict, price_dict=None):
        self.foreign_dict = foreign_dict
        self.prop_dict = prop_dict
        self.out_shares_dict = out_shares_dict
        self.price_dict = price_dict
        
        self.MAX_DELAY_DAYS = 15
        self.current_date = pd.Timestamp.now().normalize()

    def analyze_ticker(self, ticker, board_info=None, target_date=None):
        """Phân tích toàn diện và trả về Bản án định lượng cho 1 mã cổ phiếu"""
        # Dùng target_date của kịch bản Backtest làm "Hiện tại"
        if target_date:
            current_date = pd.to_datetime(target_date).normalize()
        else:
            current_date = pd.Timestamp.now().normalize()

        result = {
            "total_sm_score": 0,
            "is_danger": False,       
            "sm_details": [],         
            "warnings": [],           
            "valid_f": False,         
            "valid_p": False,         
            "last_trade_date": None   
        }

        df_f = self.foreign_dict.get(ticker)
        df_p = self.prop_dict.get(ticker)
        df_price = self.price_dict.get(ticker) if self.price_dict else None
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
                result["sm_details"].append(f"Gom kiệt cung 6M (+{absorption_rate:.1f}%) (+15)")
            elif absorption_rate >= 2.0:
                result["total_sm_score"] += 10
                result["sm_details"].append(f"Gom ròng 6M (+{absorption_rate:.1f}%) (+10)")
            elif absorption_rate <= -5.0:
                result["total_sm_score"] -= 15
                result["sm_details"].append(f"Xả rát 6M ({absorption_rate:.1f}%) (-15)")
            elif absorption_rate <= -2.0:
                result["total_sm_score"] -= 10
                result["sm_details"].append(f"Xả ròng 6M ({absorption_rate:.1f}%) (-10)")

        # =====================================================================
        # LỚP 2 & 3: TẦM NHÌN TRUNG HẠN (20 PHIÊN) & NGẮN HẠN (5 PHIÊN)
        # TÍCH HỢP TÍNH TOÁN FOOTPRINT (SỨC ẢNH HƯỞNG) LÊN THANH KHOẢN
        # =====================================================================
        foreign_base_positive, prop_base_positive = False, False
        foreign_base_negative, prop_base_negative = False, False
        
        # Lấy tổng thanh khoản 20 phiên để làm thước đo
        total_liq_20d_bn = 0
        if df_price is not None and not df_price.empty:
            df_price_20d = df_price.tail(20)
            total_liq_20d_bn = (df_price_20d['close'] * df_price_20d['volume']).sum() / 1_000_000_000

        # --- A. PHÂN TÍCH KHỐI NGOẠI ---
        if df_f is not None and not df_f.empty:
            # Cắt bỏ tương lai nếu có target_date
            df_f = df_f[df_f['time'] <= current_date]
            if df_f.empty: pass
            else:
                last_date_f = pd.to_datetime(df_f['time'].max())
                
                # CHỈ ĐÁNH GIÁ NẾU CÒN HẠN SỬ DỤNG
                if (current_date - last_date_f).days <= self.MAX_DELAY_DAYS:
                    result["valid_f"] = True
                    
                    df_f_20d = df_f.tail(20)
                    net_val_f_20d_bn = df_f_20d['foreign_net_value'].sum() / 1_000_000_000
                    f_footprint = (net_val_f_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if f_footprint >= 5.0: foreign_base_positive = True
                    elif f_footprint <= -5.0: foreign_base_negative = True

                    # Ngắn hạn 5 phiên
                    df_f_5d = df_f_20d.tail(5)
                    net_val_f_5d_bn = df_f_5d['foreign_net_value'].sum() / 1_000_000_000
                    
                    if net_val_f_5d_bn > 15.0:
                        result["total_sm_score"] += 5
                        result["sm_details"].append(f"Tây vồ mồi 5D (+{net_val_f_5d_bn:.1f} Tỷ) (+5)")
                    elif net_val_f_5d_bn < -15.0:
                        result["total_sm_score"] -= 5
                        result["sm_details"].append(f"Tây táng mạnh 5D ({net_val_f_5d_bn:.1f} Tỷ) (-5)")
                        
                    # KIỂM TOÁN ĐỘNG LƯỢNG (MOMENTUM DECAY)
                    if foreign_base_positive:
                        # 1. Bắt Gia Tốc: 5 ngày cuối chiếm > 50% tổng lực 20 ngày?
                        if net_val_f_5d_bn > (net_val_f_20d_bn * 0.5) and net_val_f_5d_bn > 10:
                            result["total_sm_score"] += 10
                            result["sm_details"].append("🔥 Gia tốc Tây gom TĂNG MẠNH (Tiền cực nóng) (+10)")
                            
                        # 2. Bắt Độ Trễ (Nguội lạnh): Tìm ngày gom mạnh gần nhất (> 10 Tỷ)
                        major_buys = df_f_20d[df_f_20d['foreign_net_value'] > 10_000_000_000]
                        if not major_buys.empty:
                            last_big_buy_date = pd.to_datetime(major_buys['time'].max())
                            days_since_big_buy = (current_date - last_big_buy_date).days
                            
                            # Nếu gom 20 ngày dương, nhưng lệnh lớn cuối cùng cách đây > 10 ngày và 5 ngày qua đi ngang
                            if days_since_big_buy >= 10 and net_val_f_5d_bn <= 0:
                                result["total_sm_score"] -= 10 # Phạt nặng vì tiền đã nguội
                                result["sm_details"].append(f"❄️ Tây ngừng gom > {days_since_big_buy} ngày (Dòng tiền nguội) (-10)")

                    # Cảnh báo Đỏ... (Phần này giữ nguyên)
                    df_f_3d = df_f_20d.tail(3)
                    if (df_f_3d['foreign_net_value'].sum() / 1_000_000_000) < -50.0:
                        result["is_danger"] = True
                        result["warnings"].append("Tây tháo cống 3D")
                        result["last_trade_date"] = df_f_3d['time'].iloc[-1]
                else:
                    # Nếu vượt MAX_DELAY_DAYS tính từ last_date đến current_date
                    result["sm_details"].append(f"(Tây bỏ rơi mã này {(current_date - last_date_f).days} ngày)")

        # --- B. PHÂN TÍCH TỰ DOANH ---
        if df_p is not None and not df_p.empty:
            # 🚀 CẮT BỎ TƯƠNG LAI (Chống Time-Travel Bug)
            df_p = df_p[df_p['time'] <= current_date]
            
            if not df_p.empty:
                last_date_p = pd.to_datetime(df_p['time'].max())
                
                # CHỈ ĐÁNH GIÁ NẾU CÒN HẠN SỬ DỤNG
                if (current_date - last_date_p).days <= self.MAX_DELAY_DAYS:
                    result["valid_p"] = True
                    
                    df_p_20d = df_p.tail(20)
                    if 'prop_net_value' in df_p_20d.columns:
                        net_val_p_20d_bn = df_p_20d['prop_net_value'].sum() / 1_000_000_000
                        p_footprint = (net_val_p_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                        
                        if p_footprint >= 5.0: prop_base_positive = True
                        elif p_footprint <= -5.0: prop_base_negative = True

                        # Ngắn hạn 5 phiên (Ngưỡng 10 tỷ - Nhỏ hơn khối ngoại 15 tỷ)
                        df_p_5d = df_p_20d.tail(5)
                        net_val_p_5d_bn = df_p_5d['prop_net_value'].sum() / 1_000_000_000
                        
                        if net_val_p_5d_bn > 10.0:
                            result["total_sm_score"] += 5
                            result["sm_details"].append(f"Tự doanh gom 5D (+{net_val_p_5d_bn:.1f} Tỷ) (+5)")
                        elif net_val_p_5d_bn < -10.0:
                            result["total_sm_score"] -= 5
                            result["sm_details"].append(f"Tự doanh xả 5D ({net_val_p_5d_bn:.1f} Tỷ) (-5)")
                            
                        # KIỂM TOÁN ĐỘNG LƯỢNG (MOMENTUM DECAY)
                        if prop_base_positive:
                            # 1. Bắt Gia Tốc: 5 ngày cuối chiếm > 50% tổng lực 20 ngày? (Ngưỡng 8 Tỷ)
                            if net_val_p_5d_bn > (net_val_p_20d_bn * 0.5) and net_val_p_5d_bn > 8.0:
                                result["total_sm_score"] += 10
                                result["sm_details"].append("🔥 Gia tốc Tự doanh gom TĂNG MẠNH (Đánh hơi tin tức) (+10)")
                                
                            # 2. Bắt Độ Trễ (Nguội lạnh): Lệnh gom mạnh gần nhất (> 8 Tỷ)
                            major_buys_p = df_p_20d[df_p_20d['prop_net_value'] > 8_000_000_000]
                            if not major_buys_p.empty:
                                last_big_buy_date_p = pd.to_datetime(major_buys_p['time'].max())
                                days_since_big_buy_p = (current_date - last_big_buy_date_p).days
                                
                                # Nếu lặn mất tăm > 10 ngày và 5 ngày qua không màng mua thêm
                                if days_since_big_buy_p >= 10 and net_val_p_5d_bn <= 0:
                                    result["total_sm_score"] -= 10
                                    result["sm_details"].append(f"❄️ Tự doanh ngừng gom > {days_since_big_buy_p} ngày (Dòng tiền nguội) (-10)")

                        # Cảnh báo Đỏ: Tự doanh xả > 30 tỷ trong 3 phiên
                        df_p_3d = df_p_20d.tail(3)
                        if (df_p_3d['prop_net_value'].sum() / 1_000_000_000) < -30.0:
                            result["is_danger"] = True
                            result["warnings"].append("Tự doanh thoát hàng 3D")
                            p_last_time = df_p_3d['time'].iloc[-1]
                            if result["last_trade_date"] is None or result["last_trade_date"] < p_last_time:
                                result["last_trade_date"] = p_last_time
                else:
                    # Rà soát xem Khối ngoại có chung số phận bị lãng quên không để tránh in log trùng lặp
                    if not any("bỏ rơi" in text for text in result["sm_details"]): 
                        result["sm_details"].append(f"(Tự doanh bỏ rơi {(current_date - last_date_p).days} ngày)")

        # --- C. ĐÁNH GIÁ CỘNG HƯỞNG TRUNG HẠN (DEADLY / SUPER COMBO) ---
        if foreign_base_positive and prop_base_positive:
            result["total_sm_score"] += 15
            result["sm_details"].append("🌟 SUPER COMBO 1M (Ngoại + TD đồng thuận gom) (+15)")
        elif foreign_base_negative and prop_base_negative:
            result["total_sm_score"] -= 15
            result["sm_details"].append("🚨 DEADLY COMBO 1M (Ngoại + TD tháo cống) (-15)")
            if result["valid_f"] and result["valid_p"]:
                result["is_danger"] = True
                result["warnings"].append("Deadly Combo Xả")
                result["last_trade_date"] = datetime.now()

        # =====================================================================
        # LỚP 4: TẦM NHÌN TỨC THỜI (BẢNG ĐIỆN INTRADAY)
        # =====================================================================
        if board_info:
            net_f_intraday = board_info.get('net_foreign', 0)
            if net_f_intraday > 5.0: # Trong phiên mua ròng > 5 tỷ mới có ý nghĩa
                result["total_sm_score"] = min(result["total_sm_score"] + 10, 20)
                result["sm_details"].append(f"Tây Mua Tức Thời (+{net_f_intraday:.1f} Tỷ) (+10)")
            elif net_f_intraday < -5.0:
                result["total_sm_score"] = max(result["total_sm_score"] - 10, -20)
                result["sm_details"].append(f"Tây Bán Tức Thời ({net_f_intraday:.1f} Tỷ) (-10)")

        return result