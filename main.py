# main.py
import os
import json
import pandas as pd
from pathlib import Path

# Sàn HOSE (TP.HCM)
# Khớp lệnh định kỳ mở cửa (ATO): 9h00 – 9h15
# Khớp lệnh liên tục: 9h15 – 11h30
# Nghỉ trưa: 11h30 – 13h00
# Khớp lệnh liên tục: 13h00 – 14h30
# Khớp lệnh định kỳ đóng cửa (ATC): 14h30 – 14h45
# Giao dịch thỏa thuận: 9h00 – 15h00

# from src.collector import VNStockDataPipeline
from src.run_bot import run_trading_system, run_vn30_trading_system, run_midcap_trading_system, run_smallcap_trading_system
# from src.run_bot import run_cashflow_report, run_cashflow_group_report
# from src.backtester import VectorizedBacktester
# from src.validator import ValidatePipeline
# from src.market_flow_by_unit import MarketFlowAnalyzer
# from src.optimizer import QuantOptimizer 
# from src.reporter import CashFlowReporter 
# from src.notifier import send_telegram_alert 
# from src.inspector import SignalInspector 
from src.flow_tracker import SmartMoneyTracker
# from src.shadow_profiler import ShadowProfiler
# from src.market_tracker import MarketTracker
# from src.post_mortem import PostMortemAnalyzer
from src.sniper import TargetSniper

def main():
    df_price = _load_parquet(Path('data/parquet/price/master_price.parquet'))
    df_intra = _load_parquet(Path('data/parquet/intraday/master_intraday.parquet'))
    df_foreign = _load_parquet(Path('data/parquet/macro/foreign_flow.parquet'))
    df_prop = _load_parquet(Path('data/parquet/macro/prop_flow.parquet'))
    df_comp = _load_parquet(Path('data/parquet/company/master_company.parquet'))
    df_industry = _load_parquet(Path('data/parquet/macro/groups_by_industries.parquet'))
    df_index = _load_parquet(Path('data/parquet/macro/index_components.parquet'))

    price_dict = _load_data_dict(Path('data/parquet/price/master_price.parquet'))
    foreign_dict = _load_data_dict(Path('data/parquet/macro/foreign_flow.parquet'))
    prop_dict = _load_data_dict(Path('data/parquet/macro/prop_flow.parquet'))
    out_shares_dict = {}
    if not df_comp.empty and 'ticker' in df_comp.columns and 'issue_share' in df_comp.columns:
        out_shares_dict = df_comp.set_index('ticker')['issue_share'].to_dict()

    # pipeline = VNStockDataPipeline(source='VCI')
    # # pipeline.get_group = 'HOSE'
    # # pipeline.get_group = 'VN30'
    # # pipeline.get_group = 'VN100'
    # # pipeline.get_group = 'VNMidCap'
    # # pipeline.get_group = 'VNSmallCap'
    # # pipeline.get_macro = True
    # # pipeline.get_index = True
    # # pipeline.get_com = True
    # # pipeline.get_price = True
    # # pipeline.get_intra = True
    # # pipeline.get_pt = True
    # # pipeline.get_board = True
    # # pipeline.get_fin = True
    # # pipeline.get_foreign = True
    # # pipeline.get_prop = True
    # # pipeline.get_fund = True
    # # pipeline.get_share_group = True
    # pipeline.run_pipeline()

    # Chạy vào 14h15 hàng ngày là tốt nhất
    # run_trading_system()
    # run_vn30_trading_system()
    # run_midcap_trading_system()
    # run_smallcap_trading_system()

    # # watch_list = ["SHB","GMD","FPT","VHM","VIC"]
    # watch_list = ["SHB","GMD"]
    # for ticker in watch_list:
    #     sniper = TargetSniper(ticker=ticker)
    #     sniper.analyze()

    # # Test: Kiểm tra mã tăng và signal trước đó
    # analyzer = PostMortemAnalyzer()
    # analyzer.analyze(ticker="GMD", target_date_str='2026-04-20', lookback_days=30)

    # # Test: Kiểm tra thị trường, thanh khoản và volume mà Cá Mập nắm giữ
    # analyzer = MarketFlowAnalyzer(df_price, df_foreign, df_prop)
    # res = analyzer.analyze_flow("VHM", target_date_str=None)
    # print(analyzer.format_report(res))

    # watch_list = _load_watchlist(Path("data/live/watchlist.json"))
    # for ticker in watch_list:
    #     res = analyzer.analyze_flow(ticker, target_date_str=None)
    #     print(analyzer.format_report(res))

    # manual_list = ["DHM","NO1","VSI"]
    # for ticker in manual_list:
    #     res = analyzer.analyze_flow(ticker, target_date_str=None)
    #     # res = analyzer.analyze_flow(ticker, target_date_str='2026-03-09')
    #     print(analyzer.format_report(res))

    # # Test: Chạy test để lấy chỉ số config tối ưu nhất
    # vn30_tickers = df_index[df_index['index_code'] == 'VN30']['ticker'].tolist()
    # mid_tickers = df_index[df_index['index_code'] == 'VNMidCap']['ticker'].tolist()
    # small_tickers = df_index[df_index['index_code'] == 'VNSmallCap']['ticker'].tolist()
    # opt = QuantOptimizer()
    # best_config = opt.run_optimization_pipeline(test_tickers=small_tickers)

    # # Test: Kiềm tra dòng tiền lớn ở các mã theo ngày
    # run_cashflow_report('day', df_foreign, df_prop, target_date='2026-04-17')
    # run_cashflow_report('week', df_foreign, df_prop, target_date='2026-04-17')
    # run_cashflow_report('month', df_foreign, df_prop, target_date='2026-04-17')

    # # Test: Kiềm tra dòng tiền lớn ở các mã theo ngày và theo ngành
    # run_cashflow_group_report('day', df_foreign, df_prop, df_industry, target_date='2026-03-20')
    # run_cashflow_group_report('week', df_foreign, df_prop, df_industry, target_date=None)
    # run_cashflow_group_report('month', df_foreign, df_prop, df_industry, target_date='2026-03-13')

    # # Test: Kiềm tra quá khứ của các mã để phân tích nếu cần
    # inspector = SignalInspector(
    #     universe='VNMidCap', df_price=df_price, 
    #     df_foreign=df_foreign, df_prop=df_prop, 
    #     df_comp=df_comp, df_idx=df_index,lookback_days=30
    # )
    # # Lấy Top 30 mã Win/Loss để chuẩn bị cho Giai đoạn 2
    # winners, losers = inspector.scan_momentum(top_n=10)
    # if winners is not None:
    #     audit_report = inspector.audit_winners(winners)
    # if losers is not None:
    #     audit_lose_report = inspector.audit_losers(losers)
    #     # Mang Tội đồ đi đối chất với Smart Money
    #     inspector.cross_audit_traps(audit_lose_report)
    # # Soi mã (đơn) muốn kiểm tra
    # manual_list = ["DIG", "GMD"]
    # for ticker in manual_list:
    #     inspector.inspect_single_ticker(ticker=ticker)

    # # Bóc tách dòng tiền của 1 mã bất kỳ
    # tracker = SmartMoneyTracker(df_price, df_foreign, df_prop, df_indx=df_index)
    # tracker.track_ticker(ticker='SHB', target_date=None, start_date='2026-03-01')

    # # Kiểm soát vùng xám
    # profiler = ShadowProfiler(df_price)
    # # Định nghĩa Tệp Tội phạm (Các mã Penny/Midcap siêu đầu cơ thường có lái)
    # all_tickers = df_price['ticker'].unique().tolist()
    # market_tickers = [t for t in all_tickers if len(str(t)) == 3]
    # training_tickers = profiler._filter_shadow_candidates(market_tickers)
    # # shadow_tickers = ['DHM', 'NO1', 'VSI']
    # shadow_tickers = ['HRC']
    # # Pha 1: Máy học tự động đo lường quy luật của nhóm này
    # rules = profiler.build_criminal_profile(training_tickers, lookback_days=300)
    # # Pha 2: Áp luật quét trực tiếp bảng điện hôm nay
    # if rules:
    #     profiler.live_shadow_radar(shadow_tickers, rules, target_date='2026-02-10')

    # tracker = MarketTracker(data_dir='data/parquet')
    # # # Tuần
    # # df_perf_1w = tracker.analyze_market_breadth(lookback_days=5, label="1 TUẦN QUA")
    # # leading_sectors = tracker.analyze_sector_rotation(df_perf_1w, top_n=5)
    # # tracker.analyze_flow_attribution(lookback_days=5)
    # # # Ngày
    # # df_perf = tracker.analyze_market_breadth(lookback_days=1, label="HÔM QUA")
    # # leading_sectors = tracker.analyze_sector_rotation(df_perf, top_n=5)
    # # tracker.analyze_flow_attribution(lookback_days=1)
    # tracker.analyze_full_intraday_macro(intraday_df=df_intra)

    # # Test thử giai đoạn Khó khăn nhất (Năm 2022 - Downtrend)
    # # bt = VectorizedBacktester(start_date='2022-01-01', end_date='2022-12-31')
    # bt = VectorizedBacktester(start_date='2025-01-01', end_date='2025-12-31')
    # # bt = VectorizedBacktester(start_date='2026-01-01')
    # # bt = VectorizedBacktester(start_date='2021-01-01')
    # df_market = bt.load_and_prep_data()
    # df_signals = bt.generate_signals(df_market)
    # eq_curve, trades = bt.run_simulation(df_signals)
    # bt.calculate_metrics(eq_curve, trades)

    # pipeline = ValidatePipeline()
    # pipeline.validate_master_price()

def _load_parquet(path):
    if path.exists():
        try: 
            return pd.read_parquet(path)
        except: 
            print(f"Could NOT read {path}")
            return pd.DataFrame()
    return pd.DataFrame()

def _load_watchlist(path):
    if not path.exists(): return {}
    try:
        with open(path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except: return {}

def _load_data_dict(path):
    df = _load_parquet(path)
    if df.empty or 'ticker' not in df.columns:
        return {}
        
    data_dict = {}
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('time')
        data_dict[ticker] = group
        
    return data_dict

if __name__ == "__main__":
    main()