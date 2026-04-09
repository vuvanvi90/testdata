import pandas as pd
from datetime import datetime

class SmartMoneyEngine:
    """
    Động cơ Phân tích Dòng tiền Thông minh (Smart Money) Đa Khung Thời Gian
    Bản Nâng Cấp: Tra cứu Phiên gần nhất bằng Ngày Giao Dịch Tuyệt Đối (Absolute Timestamp).
    """
    def __init__(self, foreign_dict, prop_dict, out_shares_dict, price_dict=None, universe="VN30"):
        self.foreign_dict = foreign_dict
        self.prop_dict = prop_dict
        self.out_shares_dict = out_shares_dict
        self.price_dict = price_dict
        self.universe = universe
        
        self.MAX_DELAY_DAYS = 15
        self.thresh = self._set_dynamic_thresholds()

    def _set_dynamic_thresholds(self):
        """Thiết lập các mốc Tiền (Tỷ VNĐ) tùy thuộc vào độ lớn của Rổ Cổ Phiếu"""
        if self.universe == "VN30":
            return {
                "t1_spike": 40.0,         # Cú giật T-1 siêu mạnh (Gom/Xả đột biến)
                "intraday_danger": -20.0, # Phanh khẩn cấp T0
                "short_term_acc": 20.0,   # Gom 5D 
                "dump_3d": -80.0          # Tháo cống 3D liên tục
            }
        elif self.universe == "VNMidCap":
            return {
                "t1_spike": 12.0,       
                "intraday_danger": -10.0, 
                "short_term_acc": 10.0,   
                "dump_3d": -30.0          
            }
        else: # VNSmallCap / Penny
            return {
                "t1_spike": 3.0,        # Rất nhỏ nhưng bất thường với Penny
                "intraday_danger": -3.0, 
                "short_term_acc": 5.0,    
                "dump_3d": -10.0          
            }

    def analyze_ticker(self, ticker, board_info=None, target_date=None):
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
        # BƯỚC 0: TẠO THƯỚC ĐO THỜI GIAN CHUẨN (TRUE-TIME WINDOWS)
        # =====================================================================
        cutoff_6m = current_date - pd.DateOffset(months=6)
        cutoff_1m = current_date - pd.Timedelta(days=30)
        cutoff_1w = current_date - pd.Timedelta(days=7)
        cutoff_3d = current_date - pd.Timedelta(days=4)
        cutoff_10d = current_date - pd.Timedelta(days=14)
        
        total_liq_20d_bn = 0
        latest_trading_date = current_date # Mặc định

        if df_price is not None and not df_price.empty:
            df_price_valid = df_price[df_price['time'] <= current_date]
            if not df_price_valid.empty:
                # Lấy CHÍNH XÁC ngày giao dịch gần nhất của thị trường
                latest_trading_date = df_price_valid['time'].max()
                
                trading_dates = df_price_valid['time'].sort_values().unique()
                if len(trading_dates) >= 130: cutoff_6m = trading_dates[-130]
                if len(trading_dates) >= 20:  cutoff_1m = trading_dates[-20]
                if len(trading_dates) >= 10:  cutoff_10d = trading_dates[-10]
                if len(trading_dates) >= 5:   cutoff_1w = trading_dates[-5]
                if len(trading_dates) >= 3:   cutoff_3d = trading_dates[-3]
                
                df_price_20d = df_price_valid[df_price_valid['time'] >= cutoff_1m]
                total_liq_20d_bn = (df_price_20d['close'] * df_price_20d['volume']).sum() / 1_000_000_000

        df_f_valid = df_f[df_f['time'] <= current_date] if (df_f is not None and not df_f.empty) else pd.DataFrame()
        df_p_valid = df_p[df_p['time'] <= current_date] if (df_p is not None and not df_p.empty) else pd.DataFrame()

        # =====================================================================
        # LỚP 1: TẦM NHÌN DÀI HẠN (130 PHIÊN GẦN NHẤT)
        # =====================================================================
        total_net_f_6m = total_net_p_6m = 0
        
        if not df_f_valid.empty:
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
        # LỚP 2 & 3: TẦM NHÌN TRUNG HẠN (20D), NGẮN HẠN (5D) & PHIÊN CUỐI (T-1/T0)
        # =====================================================================
        foreign_base_positive, prop_base_positive = False, False
        foreign_base_negative, prop_base_negative = False, False

        # --- A. PHÂN TÍCH KHỐI NGOẠI ---
        if not df_f_valid.empty:
            last_date_f = pd.to_datetime(df_f_valid['time'].max())
            if (current_date - last_date_f).days <= self.MAX_DELAY_DAYS:
                result["valid_f"] = True
                try:
                    df_f_20d = df_f_valid[df_f_valid['time'] >= cutoff_1m]
                    net_val_f_20d_bn = df_f_20d['foreign_net_value'].sum() / 1_000_000_000 if 'foreign_net_value' in df_f_20d.columns else 0
                    f_footprint = (net_val_f_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if f_footprint >= 5.0: foreign_base_positive = True
                    elif f_footprint <= -5.0: foreign_base_negative = True

                    # =======================================================
                    # 🚀 KÍNH HIỂN VI T-1: NHẬN DIỆN MẪU HÌNH THEO RỔ
                    # =======================================================
                    df_f_latest = df_f_valid[df_f_valid['time'] == latest_trading_date]
                    latest_f_val = df_f_latest['foreign_net_value'].sum() / 1_000_000_000 if ('foreign_net_value' in df_f_latest.columns and not df_f_latest.empty) else 0
                    
                    # Tính tổng 3D và 130D để làm bối cảnh
                    df_f_3d = df_f_valid[df_f_valid['time'] >= cutoff_3d]
                    net_3d = (df_f_3d['foreign_net_value'].sum() / 1_000_000_000) if not df_f_3d.empty else 0
                    
                    # MẪU HÌNH 1: THE WASHOUT REVERSAL (RŨ BỎ KÉO CHỮ V)
                    # Dấu hiệu: 3D xả rát (dưới mức âm T1), nhưng T-1 quay xe múc đột biến lấp lại toàn bộ
                    if net_3d < -self.thresh['t1_spike'] and latest_f_val >= self.thresh['t1_spike']:
                        if absorption_rate > 2.0: # Bối cảnh 130D đang gom
                            result["total_sm_score"] += 25
                            result["sm_details"].append(f"🔥 SIÊU MẪU HÌNH WASHOUT: Rũ bỏ xong kéo giật ngược (+{latest_f_val:.1f} Tỷ T-1). Nền 6M gom cực chặt! (+25)")
                            result["is_danger"] = False # Gỡ mọi cờ đỏ nếu có
                        else:
                            result["total_sm_score"] += 5
                            result["sm_details"].append(f"⚠️ T-1 kéo mạnh (+{latest_f_val:.1f} Tỷ) nhưng Nền 6M đang phân phối. Nghi ngờ Bull-trap (+5)")

                    # MẪU HÌNH 2: STEALTH ACCUMULATION (GOM NGẦM - ĐẶC TRỊ MIDCAP)
                    elif f_footprint >= 5.0 and latest_f_val > 0 and latest_f_val < self.thresh['t1_spike']:
                        # Footprint 20D chiếm > 5% thanh khoản nhưng T-1 không có lệnh giật sốc -> Đang gom ngầm ém giá
                        result["total_sm_score"] += 10
                        result["sm_details"].append(f"🕵️ MẪU HÌNH STEALTH: Gom ngầm rỉ rả 20D (Footprint: {f_footprint:.1f}%). Chờ nổ SOS! (+10)")

                    # MẪU HÌNH 3: INSIDER ANOMALY (BẤT THƯỜNG Ở PENNY)
                    elif self.universe == "VNSmallCap" and latest_f_val >= self.thresh['t1_spike']:
                        result["total_sm_score"] += 15
                        result["sm_details"].append(f"🚨 INSIDER ANOMALY: Tây bất ngờ đổ {latest_f_val:.1f} Tỷ vào hàng Penny! Game M&A? (+15)")

                    # LOGIC CẢNH BÁO TIÊU CHUẨN CÒN LẠI
                    else:
                        if latest_f_val >= self.thresh['t1_spike']:
                            result["total_sm_score"] += 10
                            result["sm_details"].append(f"Tây gom mạnh phiên T-1 (+{latest_f_val:.1f} Tỷ) (+10)")
                        elif latest_f_val <= -self.thresh['t1_spike']:
                            result["is_danger"] = True
                            result["warnings"].append(f"Tây XẢ ĐỘT BIẾN T-1 ({latest_f_val:.1f} Tỷ)")
                            result["last_trade_date"] = latest_trading_date
                            
                        if net_3d < self.thresh['dump_3d']:
                            result["is_danger"] = True
                            result["warnings"].append(f"Tây tháo cống 3D ({net_3d:.1f} Tỷ)")
                            result["last_trade_date"] = latest_trading_date
                except Exception as e:
                    print(f"[!] Smart Money Lỗi phân tích Khối ngoại: {e}")
            else:
                result["sm_details"].append(f"(Tây bỏ rơi mã này {(current_date - last_date_f).days} ngày)")

        # --- B. PHÂN TÍCH TỰ DOANH ---
        if not df_p_valid.empty:
            last_date_p = pd.to_datetime(df_p_valid['time'].max())
            if (current_date - last_date_p).days <= self.MAX_DELAY_DAYS:
                result["valid_p"] = True
                try:
                    df_p_20d = df_p_valid[df_p_valid['time'] >= cutoff_1m]
                    net_val_p_20d_bn = df_p_20d['prop_net_value'].sum() / 1_000_000_000 if 'prop_net_value' in df_p_20d.columns else 0
                    p_footprint = (net_val_p_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if p_footprint >= 5.0: prop_base_positive = True
                    elif p_footprint <= -5.0: prop_base_negative = True

                    # =======================================================
                    # 🚀 KÍNH HIỂN VI T-1: NHẬN DIỆN MẪU HÌNH THEO RỔ
                    # =======================================================
                    df_p_latest = df_p_valid[df_p_valid['time'] == latest_trading_date]
                    latest_p_val = df_p_latest['prop_net_value'].sum() / 1_000_000_000 if ('prop_net_value' in df_p_latest.columns and not df_p_latest.empty) else 0
                    
                    # Tính tổng 3D và 130D để làm bối cảnh
                    df_p_3d = df_p_valid[df_p_valid['time'] >= cutoff_3d]
                    net_3d = (df_p_3d['prop_net_value'].sum() / 1_000_000_000) if not df_p_3d.empty else 0
                    
                    # MẪU HÌNH 1: THE WASHOUT REVERSAL (RŨ BỎ KÉO CHỮ V)
                    # Dấu hiệu: 3D xả rát (dưới mức âm T1), nhưng T-1 quay xe múc đột biến lấp lại toàn bộ
                    if net_3d < -self.thresh['t1_spike'] and latest_p_val >= self.thresh['t1_spike']:
                        if absorption_rate > 2.0: # Bối cảnh 130D đang gom
                            result["total_sm_score"] += 25
                            result["sm_details"].append(f"🔥 SIÊU MẪU HÌNH WASHOUT: Rũ bỏ xong kéo giật ngược (+{latest_p_val:.1f} Tỷ T-1). Nền 6M gom cực chặt! (+25)")
                            result["is_danger"] = False # Gỡ mọi cờ đỏ nếu có
                        else:
                            result["total_sm_score"] += 5
                            result["sm_details"].append(f"⚠️ T-1 kéo mạnh (+{latest_p_val:.1f} Tỷ) nhưng Nền 6M đang phân phối. Nghi ngờ Bull-trap (+5)")

                    # MẪU HÌNH 2: STEALTH ACCUMULATION (GOM NGẦM - ĐẶC TRỊ MIDCAP)
                    elif p_footprint >= 5.0 and latest_p_val > 0 and latest_p_val < self.thresh['t1_spike']:
                        # Footprint 20D chiếm > 5% thanh khoản nhưng T-1 không có lệnh giật sốc -> Đang gom ngầm ém giá
                        result["total_sm_score"] += 10
                        result["sm_details"].append(f"🕵️ MẪU HÌNH STEALTH: Gom ngầm rỉ rả 20D (Footprint: {p_footprint:.1f}%). Chờ nổ SOS! (+10)")

                    # MẪU HÌNH 3: INSIDER ANOMALY (BẤT THƯỜNG Ở PENNY)
                    elif self.universe == "VNSmallCap" and latest_p_val >= self.thresh['t1_spike']:
                        result["total_sm_score"] += 15
                        result["sm_details"].append(f"🚨 INSIDER ANOMALY: Tự doanh bất ngờ đổ {latest_p_val:.1f} Tỷ vào hàng Penny! Game M&A? (+15)")

                    # LOGIC CẢNH BÁO TIÊU CHUẨN CÒN LẠI
                    else:
                        if latest_p_val >= self.thresh['t1_spike']:
                            result["total_sm_score"] += 10
                            result["sm_details"].append(f"Tự doanh gom mạnh phiên T-1 (+{latest_p_val:.1f} Tỷ) (+10)")
                        elif latest_p_val <= -self.thresh['t1_spike']:
                            result["is_danger"] = True
                            result["warnings"].append(f"Tự doanh XẢ ĐỘT BIẾN T-1 ({latest_p_val:.1f} Tỷ)")
                            result["last_trade_date"] = latest_trading_date
                            
                        if net_3d < self.thresh['dump_3d']:
                            result["is_danger"] = True
                            result["warnings"].append(f"Tự doanh tháo cống 3D ({net_3d:.1f} Tỷ)")
                            result["last_trade_date"] = latest_trading_date
                except Exception as e:
                    print(f"[!] Smart Money Lỗi phân tích Tự doanh: {e}")
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
                result["last_trade_date"] = current_date

        # =====================================================================
        # LỚP 4: TẦM NHÌN TỨC THỜI (BẢNG ĐIỆN INTRADAY T0)
        # =====================================================================
        if board_info:
            net_f_intraday = board_info.get('net_foreign', 0)
            
            # CHỈ GIỮ LẠI CHỨC NĂNG PHANH KHẨN CẤP (EMERGENCY BRAKE)
            if net_f_intraday <= self.thresh['intraday_danger']:
                result["is_danger"] = True
                result["warnings"].append(f"Tây tháo cống INTRADAY ({net_f_intraday:.1f} Tỷ)")
                result["last_trade_date"] = current_date
            
            if net_f_intraday >= (self.thresh['t1_spike'] * 0.5): # T0 mua = nửa cú giật T-1 là đáng kể
                result["sm_details"].append(f"Tây Mua Tức Thời T0 (+{net_f_intraday:.1f} Tỷ)")
            elif net_f_intraday <= -(self.thresh['t1_spike'] * 0.5):
                result["sm_details"].append(f"Tây Bán Tức Thời T0 ({net_f_intraday:.1f} Tỷ)")

        return result