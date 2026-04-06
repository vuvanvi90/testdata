import pandas as pd
from datetime import datetime

class SmartMoneyEngine:
    """
    Động cơ Phân tích Dòng tiền Thông minh (Smart Money) Đa Khung Thời Gian
    Bản Nâng Cấp: Đồng bộ Mốc Thời Gian Chuẩn (True-Time Windows) để chống ảo giác dữ liệu.
    """
    def __init__(self, foreign_dict, prop_dict, out_shares_dict, price_dict=None):
        self.foreign_dict = foreign_dict
        self.prop_dict = prop_dict
        self.out_shares_dict = out_shares_dict
        self.price_dict = price_dict
        
        self.MAX_DELAY_DAYS = 15

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
        # 🚀 BƯỚC 0: TẠO THƯỚC ĐO THỜI GIAN CHUẨN (TRUE-TIME WINDOWS)
        # =====================================================================
        # Khởi tạo mốc thời gian mặc định (Dùng Lịch thường nếu không có Dữ liệu Giá)
        cutoff_6m = current_date - pd.DateOffset(months=6)
        cutoff_1m = current_date - pd.Timedelta(days=30)
        cutoff_1w = current_date - pd.Timedelta(days=7)
        cutoff_3d = current_date - pd.Timedelta(days=4)
        
        total_liq_20d_bn = 0

        # Nếu có df_price, ta sẽ ĐẾM LÙI CHÍNH XÁC SỐ PHIÊN GIAO DỊCH THỰC TẾ
        if df_price is not None and not df_price.empty:
            df_price_valid = df_price[df_price['time'] <= current_date]
            if not df_price_valid.empty:
                trading_dates = df_price_valid['time'].sort_values().unique()
                
                # Cắt chính xác mốc thời gian dựa trên lịch giao dịch
                if len(trading_dates) >= 130: cutoff_6m = trading_dates[-130]
                if len(trading_dates) >= 20:  cutoff_1m = trading_dates[-20]
                if len(trading_dates) >= 5:   cutoff_1w = trading_dates[-5]
                if len(trading_dates) >= 3:   cutoff_3d = trading_dates[-3]
                
                # Tính tổng thanh khoản 20 phiên thực tế
                df_price_20d = df_price_valid[df_price_valid['time'] >= cutoff_1m]
                total_liq_20d_bn = (df_price_20d['close'] * df_price_20d['volume']).sum() / 1_000_000_000

        # Cắt DataFrame Dòng tiền sao cho không vượt qua current_date (Chống rò rỉ tương lai)
        df_f_valid = df_f[df_f['time'] <= current_date] if (df_f is not None and not df_f.empty) else pd.DataFrame()
        df_p_valid = df_p[df_p['time'] <= current_date] if (df_p is not None and not df_p.empty) else pd.DataFrame()

        # =====================================================================
        # LỚP 1: TẦM NHÌN DÀI HẠN (STRUCTURAL VIEW - ĐÚNG 130 PHIÊN GẦN NHẤT)
        # =====================================================================
        total_net_f_6m = total_net_p_6m = 0
        
        if not df_f_valid.empty:
            # CHỈ LẤY CÁC LỆNH NẰM TRONG 6 THÁNG GẦN NHẤT
            df_f_6m = df_f_valid[df_f_valid['time'] >= cutoff_6m]
            col_vol_f = 'foreign_net_volume' if 'foreign_net_volume' in df_f_6m.columns else 'foreign_net_vol'
            total_net_f_6m = df_f_6m[col_vol_f].sum() if col_vol_f in df_f_6m.columns else 0
            
        if not df_p_valid.empty:
            df_p_6m = df_p_valid[df_p_valid['time'] >= cutoff_6m]
            col_vol_p = 'prop_net_volume' if 'prop_net_volume' in df_p_6m.columns else 'prop_net_vol'
            total_net_p_6m = df_p_6m[col_vol_p].sum() if col_vol_p in df_p_6m.columns else 0
            
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
        # =====================================================================
        foreign_base_positive, prop_base_positive = False, False
        foreign_base_negative, prop_base_negative = False, False

        # --- A. PHÂN TÍCH KHỐI NGOẠI ---
        if not df_f_valid.empty:
            last_date_f = pd.to_datetime(df_f_valid['time'].max())
            
            # CHỈ ĐÁNH GIÁ CHIẾN THUẬT NẾU LỆNH GẦN NHẤT < 15 NGÀY (CÒN HẠN SỬ DỤNG)
            if (current_date - last_date_f).days <= self.MAX_DELAY_DAYS:
                result["valid_f"] = True
                
                # 🚀 Dùng màng lọc chuẩn 20 phiên thay vì .tail(20)
                df_f_20d = df_f_valid[df_f_valid['time'] >= cutoff_1m]
                net_val_f_20d_bn = df_f_20d['foreign_net_value'].sum() / 1_000_000_000 if 'foreign_net_value' in df_f_20d.columns else 0
                f_footprint = (net_val_f_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                
                if f_footprint >= 5.0: foreign_base_positive = True
                elif f_footprint <= -5.0: foreign_base_negative = True

                # 🚀 Dùng màng lọc chuẩn 5 phiên
                df_f_5d = df_f_valid[df_f_valid['time'] >= cutoff_1w]
                net_val_f_5d_bn = df_f_5d['foreign_net_value'].sum() / 1_000_000_000 if 'foreign_net_value' in df_f_5d.columns else 0
                
                if net_val_f_5d_bn > 15.0:
                    result["total_sm_score"] += 5
                    result["sm_details"].append(f"Tây vồ mồi 5D (+{net_val_f_5d_bn:.1f} Tỷ) (+5)")
                elif net_val_f_5d_bn < -15.0:
                    result["total_sm_score"] -= 5
                    result["sm_details"].append(f"Tây táng mạnh 5D ({net_val_f_5d_bn:.1f} Tỷ) (-5)")
                    
                # KIỂM TOÁN ĐỘNG LƯỢNG (MOMENTUM DECAY)
                if foreign_base_positive:
                    if net_val_f_5d_bn > (net_val_f_20d_bn * 0.5) and net_val_f_5d_bn > 10:
                        result["total_sm_score"] += 10
                        result["sm_details"].append("🔥 Gia tốc Tây gom TĂNG MẠNH (Tiền cực nóng) (+10)")
                        
                    major_buys = df_f_20d[df_f_20d['foreign_net_value'] > 10_000_000_000] if 'foreign_net_value' in df_f_20d.columns else pd.DataFrame()
                    if not major_buys.empty:
                        last_big_buy_date = pd.to_datetime(major_buys['time'].max())
                        days_since_big_buy = (current_date - last_big_buy_date).days
                        if days_since_big_buy >= 10 and net_val_f_5d_bn <= 0:
                            result["total_sm_score"] -= 10
                            result["sm_details"].append(f"❄️ Tây ngừng gom > {days_since_big_buy} ngày (Dòng tiền nguội) (-10)")

                # Cảnh báo Đỏ 3 Phiên chuẩn xác
                df_f_3d = df_f_valid[df_f_valid['time'] >= cutoff_3d]
                if not df_f_3d.empty and 'foreign_net_value' in df_f_3d.columns:
                    if (df_f_3d['foreign_net_value'].sum() / 1_000_000_000) < -50.0:
                        result["is_danger"] = True
                        result["warnings"].append("Tây tháo cống 3D")
                        result["last_trade_date"] = df_f_3d['time'].iloc[-1]
            else:
                result["sm_details"].append(f"(Tây bỏ rơi mã này {(current_date - last_date_f).days} ngày)")

        # --- B. PHÂN TÍCH TỰ DOANH ---
        if not df_p_valid.empty:
            last_date_p = pd.to_datetime(df_p_valid['time'].max())
            
            if (current_date - last_date_p).days <= self.MAX_DELAY_DAYS:
                result["valid_p"] = True
                
                # 🚀 Dùng màng lọc chuẩn 20 phiên
                df_p_20d = df_p_valid[df_p_valid['time'] >= cutoff_1m]
                if 'prop_net_value' in df_p_20d.columns:
                    net_val_p_20d_bn = df_p_20d['prop_net_value'].sum() / 1_000_000_000
                    p_footprint = (net_val_p_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if p_footprint >= 5.0: prop_base_positive = True
                    elif p_footprint <= -5.0: prop_base_negative = True

                    # 🚀 Dùng màng lọc chuẩn 5 phiên
                    df_p_5d = df_p_valid[df_p_valid['time'] >= cutoff_1w]
                    net_val_p_5d_bn = df_p_5d['prop_net_value'].sum() / 1_000_000_000
                    
                    if net_val_p_5d_bn > 10.0:
                        result["total_sm_score"] += 5
                        result["sm_details"].append(f"Tự doanh gom 5D (+{net_val_p_5d_bn:.1f} Tỷ) (+5)")
                    elif net_val_p_5d_bn < -10.0:
                        result["total_sm_score"] -= 5
                        result["sm_details"].append(f"Tự doanh xả 5D ({net_val_p_5d_bn:.1f} Tỷ) (-5)")
                        
                    # KIỂM TOÁN ĐỘNG LƯỢNG (MOMENTUM DECAY)
                    if prop_base_positive:
                        if net_val_p_5d_bn > (net_val_p_20d_bn * 0.5) and net_val_p_5d_bn > 8.0:
                            result["total_sm_score"] += 10
                            result["sm_details"].append("🔥 Gia tốc Tự doanh gom TĂNG MẠNH (Đánh hơi tin tức) (+10)")
                            
                        major_buys_p = df_p_20d[df_p_20d['prop_net_value'] > 8_000_000_000]
                        if not major_buys_p.empty:
                            last_big_buy_date_p = pd.to_datetime(major_buys_p['time'].max())
                            days_since_big_buy_p = (current_date - last_big_buy_date_p).days
                            if days_since_big_buy_p >= 10 and net_val_p_5d_bn <= 0:
                                result["total_sm_score"] -= 10
                                result["sm_details"].append(f"❄️ Tự doanh ngừng gom > {days_since_big_buy_p} ngày (Dòng tiền nguội) (-10)")

                    # Cảnh báo Đỏ 3 Phiên
                    df_p_3d = df_p_valid[df_p_valid['time'] >= cutoff_3d]
                    if not df_p_3d.empty and (df_p_3d['prop_net_value'].sum() / 1_000_000_000) < -30.0:
                        result["is_danger"] = True
                        result["warnings"].append("Tự doanh thoát hàng 3D")
                        p_last_time = df_p_3d['time'].iloc[-1]
                        if result["last_trade_date"] is None or result["last_trade_date"] < p_last_time:
                            result["last_trade_date"] = p_last_time
            else:
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
            if net_f_intraday > 5.0: 
                result["total_sm_score"] = min(result["total_sm_score"] + 10, 20)
                result["sm_details"].append(f"Tây Mua Tức Thời (+{net_f_intraday:.1f} Tỷ) (+10)")
            elif net_f_intraday < -5.0:
                result["total_sm_score"] = max(result["total_sm_score"] - 10, -20)
                result["sm_details"].append(f"Tây Bán Tức Thời ({net_f_intraday:.1f} Tỷ) (-10)")

        return result