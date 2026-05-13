import pandas as pd
from datetime import datetime

class SmartMoneyEngine:
    """
    Động cơ Phân tích Dòng tiền Thông minh (Lịch sử & Vị thế) V3.0
    Chỉ chuyên trách phân tích xu hướng EOD từ T-130 đến T-1. Tuyệt đối KHÔNG đụng chạm vào dữ liệu T0
    CẬP NHẬT: 
       - Tích hợp Trọng số Chi phối (Impact Factor) để khử Nhiễu.
       - Tương thích 100% với master_price_l2 (Matched Flow).
    """
    def __init__(self, price_l2_dict, prop_dict, out_shares_dict, universe="VN30"):
        self.price_l2_dict = price_l2_dict
        self.prop_dict = prop_dict
        self.out_shares_dict = out_shares_dict
        self.universe = universe
        
        self.MAX_DELAY_DAYS = 15
        self.thresh = self._set_dynamic_thresholds()

    def _set_dynamic_thresholds(self):
        """Thiết lập các mốc Tiền (Tỷ VNĐ) tùy thuộc vào độ lớn của Rổ Cổ Phiếu"""
        if self.universe == "VN30":
            return {
                "t1_spike": 40.0,         # Cú giật T-1 siêu mạnh (Gom/Xả đột biến)
                "short_term_acc": 20.0,   # Gom 5D 
                "dump_3d": -80.0,         # Tháo cống 3D liên tục
                "min_impact_pct": 5.0     # Tỷ trọng xả tối thiểu phải > 5% mới giật cờ đỏ
            }
        elif self.universe == "VNMidCap":
            return {
                "t1_spike": 12.0,       
                "short_term_acc": 10.0,   
                "dump_3d": -30.0,
                "min_impact_pct": 4.0     # Midcap dễ bị tổn thương hơn, mốc là 4%
            }
        else: # VNSmallCap / Penny
            return {
                "t1_spike": 3.0,        # Rất nhỏ nhưng bất thường với Penny
                "short_term_acc": 5.0,    
                "dump_3d": -10.0,
                "min_impact_pct": 3.0     # Penny chỉ cần Lái nhả 3% thanh khoản là sập
            }

    def analyze_ticker(self, ticker, target_date=None):
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

        df_p_raw = self.prop_dict.get(ticker) if self.prop_dict else None
        df_l2_raw = self.price_l2_dict.get(ticker) if self.price_l2_dict else None
        shares_out = self.out_shares_dict.get(ticker, 0)

        df_l2_v = df_l2_raw[df_l2_raw['time'] <= current_date].copy() if (df_l2_raw is not None and not df_l2_raw.empty) else pd.DataFrame()
        if 'matched_volume' in df_l2_v.columns and 'volume' not in df_l2_v.columns:
            df_l2_v = df_l2_v.rename(columns={'matched_volume': 'volume'})

        df_p_v = df_p_raw[df_p_raw['time'] <= current_date].copy() if (df_p_raw is not None and not df_p_raw.empty) else pd.DataFrame()

        # =====================================================================
        # 🛡️ BƯỚC 0: QUY HOẠCH DÒNG TIỀN VÀ TÍNH TRỌNG SỐ (IMPACT FACTOR)
        # =====================================================================
        df_f_valid = pd.DataFrame()
        if not df_l2_v.empty and 'fr_net_value_matched' in df_l2_v.columns:
            df_f_valid = df_l2_v[['time', 'close', 'volume']].copy()
            df_f_valid['foreign_net_value'] = df_l2_v['fr_net_value_matched']
            df_f_valid['foreign_net_volume'] = df_l2_v['fr_net_volume_matched']
            result["valid_f"] = True

        df_p_valid = pd.DataFrame()
        if not df_p_v.empty and 'prop_net_value' in df_p_v.columns:
            # df_p_valid = df_p_v[['time', 'prop_net_value', 'prop_net_volume']].copy()
            df_p_valid = df_p_v[['time']].copy()
            df_p_valid['prop_net_value'] = df_p_v['prop_net_val_matched']
            df_p_valid['prop_net_volume'] = df_p_v['prop_net_vol_matched']
            result["valid_p"] = True

        # Xác định các mốc thời gian
        cutoff_6m = current_date - pd.DateOffset(months=6)
        cutoff_1m = current_date - pd.Timedelta(days=30)
        cutoff_3d = current_date - pd.Timedelta(days=4)
        
        total_liq_20d_bn = 0
        latest_trading_date = current_date # Mặc định

        if not df_l2_v.empty:
            latest_trading_date = df_l2_v['time'].max()
            trading_dates = df_l2_v['time'].sort_values().unique()
            if len(trading_dates) >= 130: cutoff_6m = trading_dates[-130]
            if len(trading_dates) >= 20:  cutoff_1m = trading_dates[-20]
            if len(trading_dates) >= 3:   cutoff_3d = trading_dates[-3]
            
            df_p_20d = df_l2_v[df_l2_v['time'] >= cutoff_1m]
            total_liq_20d_bn = (df_p_20d['close'] * df_p_20d['volume']).sum() / 1_000_000_000

        # =====================================================================
        # LỚP 1: TẦM NHÌN DÀI HẠN (130 PHIÊN GẦN NHẤT)
        # =====================================================================
        total_net_f_6m = total_net_p_6m = 0
        
        if not df_f_valid.empty:
            df_f_6m = df_f_valid[df_f_valid['time'] >= cutoff_6m]
            total_net_f_6m = df_f_6m['foreign_net_volume'].sum() if 'foreign_net_volume' in df_f_6m.columns else 0
            
        if not df_p_valid.empty:
            df_p_6m = df_p_valid[df_p_valid['time'] >= cutoff_6m]
            total_net_p_6m = df_p_6m['prop_net_volume'].sum() if 'prop_net_volume' in df_p_6m.columns else 0
            
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
                try:
                    df_f_20d = df_f_valid[df_f_valid['time'] >= cutoff_1m]
                    net_val_f_20d_bn = df_f_20d['foreign_net_value'].sum() / 1_000_000_000
                    f_footprint = (net_val_f_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if f_footprint >= 5.0: foreign_base_positive = True
                    elif f_footprint <= -5.0: foreign_base_negative = True

                    df_f_latest = df_f_valid[df_f_valid['time'] == latest_trading_date]
                    latest_f_val = df_f_latest['foreign_net_value'].sum() / 1_000_000_000
                    # latest_total_val_bn = df_f_latest['total_val_bn'].sum()
                    df_l2_latest = df_l2_v[df_l2_v['time'] == latest_trading_date]
                    latest_total_val_bn = df_l2_latest['matched_value'].sum() / 1_000_000_000 if not df_l2_latest.empty else 1.0
                    
                    # Tính tổng 3D và 130D để làm bối cảnh
                    df_f_3d = df_f_valid[df_f_valid['time'] >= cutoff_3d]
                    net_3d = df_f_3d['foreign_net_value'].sum() / 1_000_000_000
                    # total_val_3d_bn = df_f_3d['total_val_bn'].sum()
                    df_l2_3d = df_l2_v[df_l2_v['time'] >= cutoff_3d]
                    total_val_3d_bn = df_l2_3d['matched_value'].sum() / 1_000_000_000 if not df_l2_3d.empty else 1.0

                    # MẪU HÌNH WASHOUT
                    if net_3d < -self.thresh['t1_spike'] and latest_f_val >= self.thresh['t1_spike']:
                        if absorption_rate > 2.0: # Bối cảnh 130D đang gom
                            result["total_sm_score"] += 25
                            result["sm_details"].append(f"🔥 SIÊU MẪU HÌNH WASHOUT: Ngoại rũ xong kéo giật ngược (+{latest_f_val:.1f} Tỷ). (+25)")
                        else:
                            result["total_sm_score"] += 5
                            result["sm_details"].append(f"⚠️ Ngoại kéo T-1 (+{latest_f_val:.1f} Tỷ) nhưng Nền 6M phân phối. Đề phòng Bull-trap (+5)")

                    # MẪU HÌNH 2: STEALTH ACCUMULATION (GOM NGẦM - ĐẶC TRỊ MIDCAP)
                    elif f_footprint >= 5.0 and latest_f_val > 0 and latest_f_val < self.thresh['t1_spike']:
                        # Footprint 20D chiếm > 5% thanh khoản nhưng T-1 không có lệnh giật sốc -> Đang gom ngầm ém giá
                        result["total_sm_score"] += 10
                        result["sm_details"].append(f"🕵️ MẪU HÌNH STEALTH: Ngoại gom ngầm 20D (Chiếm {f_footprint:.1f}%). (+10)")

                    # MẪU HÌNH 3: INSIDER ANOMALY (BẤT THƯỜNG Ở PENNY)
                    elif self.universe == "VNSmallCap" and latest_f_val >= self.thresh['t1_spike']:
                        result["total_sm_score"] += 15
                        result["sm_details"].append(f"🚨 INSIDER ANOMALY: Tây đổ {latest_f_val:.1f} Tỷ vào hàng Penny! Game M&A? (+15)")

                    # LOGIC CẢNH BÁO TIÊU CHUẨN CÒN LẠI
                    else:
                        # KIỂM DUYỆT LỆNH BÁN QUA BỘ LỌC IMPACT FACTOR
                        if latest_f_val >= self.thresh['t1_spike']:
                            result["total_sm_score"] += 10
                            result["sm_details"].append(f"Tây gom mạnh T-1 (+{latest_f_val:.1f} Tỷ) (+10)")
                            
                        elif latest_f_val <= -self.thresh['t1_spike']:
                            impact_pct = (abs(latest_f_val) / latest_total_val_bn * 100) if latest_total_val_bn > 0 else 100
                            if impact_pct >= self.thresh['min_impact_pct']:
                                result["is_danger"] = True
                                result["warnings"].append(f"Tây XẢ RÁT T-1 ({latest_f_val:.1f} Tỷ | Chiếm {impact_pct:.1f}% Cung)")
                                result["last_trade_date"] = latest_trading_date
                            else:
                                result["sm_details"].append(f"Tây xả T-1 ({latest_f_val:.1f} Tỷ) nhưng thanh khoản dư sức hấp thụ (Chỉ {impact_pct:.1f}% Cung).")
                            
                        if net_3d < self.thresh['dump_3d']:
                            impact_3d = (abs(net_3d) / total_val_3d_bn * 100) if total_val_3d_bn > 0 else 100
                            if impact_3d >= self.thresh['min_impact_pct']:
                                result["is_danger"] = True
                                result["warnings"].append(f"Tây tháo cống 3D ({net_3d:.1f} Tỷ | Chiếm {impact_3d:.1f}% Cung)")
                                result["last_trade_date"] = latest_trading_date
                            else:
                                result["sm_details"].append(f"Tây xả 3D ({net_3d:.1f} Tỷ) nhưng lực mua đối ứng tốt (Chỉ chiếm {impact_3d:.1f}%).")

                except Exception as e:
                    print(f"[!] Smart Money Lỗi phân tích Khối ngoại: {e}")
            else:
                result["sm_details"].append(f"(Tây bỏ rơi mã này {(current_date - last_date_f).days} ngày)")

        # --- B. PHÂN TÍCH TỰ DOANH ---
        if not df_p_valid.empty:
            last_date_p = pd.to_datetime(df_p_valid['time'].max())
            if (current_date - last_date_p).days <= self.MAX_DELAY_DAYS:
                try:
                    df_p_20d = df_p_valid[df_p_valid['time'] >= cutoff_1m]
                    net_val_p_20d_bn = df_p_20d['prop_net_value'].sum() / 1_000_000_000
                    p_footprint = (net_val_p_20d_bn / total_liq_20d_bn * 100) if total_liq_20d_bn > 0 else 0
                    
                    if p_footprint >= 5.0: prop_base_positive = True
                    elif p_footprint <= -5.0: prop_base_negative = True

                    # =======================================================
                    # 🚀 KÍNH HIỂN VI T-1: NHẬN DIỆN MẪU HÌNH THEO RỔ
                    # =======================================================
                    df_p_latest = df_p_valid[df_p_valid['time'] == latest_trading_date]
                    latest_p_val = df_p_latest['prop_net_value'].sum() / 1_000_000_000
                    # latest_total_val_bn = df_p_latest['total_val_bn'].sum()
                    df_l2_latest = df_l2_v[df_l2_v['time'] == latest_trading_date]
                    latest_total_val_bn = df_l2_latest['matched_value'].sum() / 1_000_000_000 if not df_l2_latest.empty else 1.0
                    
                    # Tính tổng 3D và 130D để làm bối cảnh
                    df_p_3d = df_p_valid[df_p_valid['time'] >= cutoff_3d]
                    net_3d = df_p_3d['prop_net_value'].sum() / 1_000_000_000
                    # total_val_3d_bn = df_p_3d['total_val_bn'].sum()
                    df_l2_3d = df_l2_v[df_l2_v['time'] >= cutoff_3d]
                    total_val_3d_bn = df_l2_3d['matched_value'].sum() / 1_000_000_000 if not df_l2_3d.empty else 1.0

                    
                    # MẪU HÌNH 1: THE WASHOUT REVERSAL (RŨ BỎ KÉO CHỮ V)
                    # Dấu hiệu: 3D xả rát (dưới mức âm T1), nhưng T-1 quay xe múc đột biến lấp lại toàn bộ
                    if net_3d < -self.thresh['t1_spike'] and latest_p_val >= self.thresh['t1_spike']:
                        if absorption_rate > 2.0: # Bối cảnh 130D đang gom
                            result["total_sm_score"] += 25
                            result["sm_details"].append(f"🔥 SIÊU MẪU HÌNH WASHOUT: Nội rũ xong kéo giật ngược (+{latest_p_val:.1f} Tỷ). (+25)")
                        else:
                            result["total_sm_score"] += 5
                            result["sm_details"].append(f"⚠️ Nội kéo T-1 (+{latest_p_val:.1f} Tỷ) nhưng Nền 6M phân phối. Đề phòng Bull-trap (+5)")

                    # MẪU HÌNH 2: STEALTH ACCUMULATION (GOM NGẦM - ĐẶC TRỊ MIDCAP)
                    elif p_footprint >= 5.0 and latest_p_val > 0 and latest_p_val < self.thresh['t1_spike']:
                        # Footprint 20D chiếm > 5% thanh khoản nhưng T-1 không có lệnh giật sốc -> Đang gom ngầm ém giá
                        result["total_sm_score"] += 10
                        result["sm_details"].append(f"🕵️ MẪU HÌNH STEALTH: Nội gom ngầm 20D (Chiếm {p_footprint:.1f}%). (+10)")

                    # MẪU HÌNH 3: INSIDER ANOMALY (BẤT THƯỜNG Ở PENNY)
                    elif self.universe == "VNSmallCap" and latest_p_val >= self.thresh['t1_spike']:
                        result["total_sm_score"] += 15
                        result["sm_details"].append(f"🚨 INSIDER ANOMALY: Tự doanh đổ {latest_p_val:.1f} Tỷ vào hàng Penny! (+15)")

                    # LOGIC CẢNH BÁO TIÊU CHUẨN CÒN LẠI
                    else:
                        # KIỂM DUYỆT LỆNH BÁN QUA BỘ LỌC IMPACT FACTOR
                        if latest_p_val >= self.thresh['t1_spike']:
                            result["total_sm_score"] += 10
                            result["sm_details"].append(f"Tự doanh gom mạnh T-1 (+{latest_p_val:.1f} Tỷ) (+10)")
                            
                        elif latest_p_val <= -self.thresh['t1_spike']:
                            impact_pct = (abs(latest_p_val) / latest_total_val_bn * 100) if latest_total_val_bn > 0 else 100
                            if impact_pct >= self.thresh['min_impact_pct']:
                                result["is_danger"] = True
                                result["warnings"].append(f"Tự doanh XẢ RÁT T-1 ({latest_p_val:.1f} Tỷ | Chiếm {impact_pct:.1f}% Cung)")
                                result["last_trade_date"] = latest_trading_date
                            else:
                                result["sm_details"].append(f"Tự doanh xả T-1 ({latest_p_val:.1f} Tỷ) nhưng Impact thấp ({impact_pct:.1f}%). Bỏ qua nhiễu.")
                            
                        if net_3d < self.thresh['dump_3d']:
                            impact_3d = (abs(net_3d) / total_val_3d_bn * 100) if total_val_3d_bn > 0 else 100
                            if impact_3d >= self.thresh['min_impact_pct']:
                                result["is_danger"] = True
                                result["warnings"].append(f"Tự doanh tháo cống 3D ({net_3d:.1f} Tỷ | Chiếm {impact_3d:.1f}% Cung)")
                                result["last_trade_date"] = latest_trading_date
                            else:
                                result["sm_details"].append(f"Tự doanh xả 3D ({net_3d:.1f} Tỷ) nhưng Impact thấp ({impact_3d:.1f}%). Bỏ qua nhiễu.")

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

        return result