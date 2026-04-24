import os
import json
import pandas as pd
import time
import copy
from datetime import datetime, timedelta
from pathlib import Path

# from vnstock_data import Listing, Company, Quote, Trading, Finance, Macro, CommodityPrice, Fund

# from vnstock_data.explorer.vci import Quote, Trading

# from vnstock_data import show_api, Reference, Market

# from src.omni_matrix import OmniFlowMatrix

class VN30DataPipeline:
    def __init__(self):
        self.source = 'vci'
        # vci
        # cafef
        # kbs
        # mbk

    def run_pipeline(self):
        df_price = self._load_parquet(Path('data/parquet/price/master_price.parquet'))
        df_board = self._load_parquet(Path('data/parquet/board/master_board.parquet'))
        df_intra = self._load_parquet(Path('data/parquet/intraday/master_intraday.parquet'))
        df_foreign = self._load_parquet(Path('data/parquet/macro/foreign_flow.parquet'))
        df_prop = self._load_parquet(Path('data/parquet/macro/prop_flow.parquet'))
        df_industry = self._load_parquet(Path('data/parquet/macro/groups_by_industries.parquet'))
        df_index = self._load_parquet(Path('data/parquet/macro/index_components.parquet'))
        df_fin = self._load_parquet(Path('data/parquet/financial/master_financial.parquet'))

        data_frames = {
            'price': df_price,
            'foreign': df_foreign,
            'prop': df_prop,
            'comp': df_industry,
            'idx': df_index,
            'board': df_board,
            'intra': df_intra
        }

        START_DATE = '2026-01-01'
        END_DATE = datetime.now().strftime('%Y-%m-%d')
        tickers = ['ACB', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VPL', 'VRE']
        ticker = ['VCB']

        # show_api(Reference())
        # show_api(Market())

        # ref = Reference()
        # df_profile = ref.company("ACB").info()
        # print(df_profile)

        # company = Company(symbol='ACB', source='KBS')
        # com_df = company.overview()
        # print(com_df)
        # print(f"Data: {type(com_df)}")

        # trading = Trading(symbol='ACB', source='vci')
        # data_df = trading.price_board(symbols_list=['ACB', 'BID', 'CTG'])
        # data_df = trading.price_history(start=START_DATE, end=END_DATE, resolution='1D')
        # data_df = trading.prop_trade(start=START_DATE, end=END_DATE, resolution='1D')
        # data_df = trading.insider_deal(lang='vi')
        # print(data_df) 

        # # Truy xuất dữ liệu giao dịch thỏa thuận (put-through)
        # trading = Trading(source='VCI')
        # df_pt = trading.put_through()
        # if not df_pt.empty:
        #     df_pt.to_json("put_through.json", orient='records', force_ascii=False, indent=4)

        # trading = Trading(symbol='ACB', source='cafef')
        # data_df = trading.price_history(start=START_DATE, end=END_DATE)
        # data_df = trading.order_stats(start=START_DATE, end=END_DATE)
        # data_df = trading.foreign_trade(start=START_DATE, end=END_DATE)
        # data_df = trading.prop_trade(start=START_DATE, end=END_DATE)
        # data_df = trading.insider_deal(start='2024-01-02', end=END_DATE)
        # print(data_df) 

        # trading = Trading(symbol='ACB', source='cafef')
        # data_df = trading.price_board(symbols_list=['HRC'])
        # if not data_df.empty:
        #     data_df.to_json("HRC_board.json", orient='records', force_ascii=False, indent=4)
        # data_df = trading.price_history()
        # print(data_df)
        # df_foreign = trading.foreign_trade(start=START_DATE, end=END_DATE)
        # if not df_foreign.empty:
        #     df_foreign.to_json("ACB_foreign.json", orient='records', force_ascii=False, indent=4)
        # print(df_foreign) 
        # df_prop = trading.prop_trade(start=START_DATE, end=END_DATE)
        # if not df_prop.empty:
        #     df_prop.to_json("ACB_prop.json", orient='records', force_ascii=False, indent=4)
        # print(df_prop) 
        # df_order = trading.order_stats(start=START_DATE, end=END_DATE)
        # if not df_order.empty:
        #     df_order.to_json("ACB_order.json", orient='records', force_ascii=False, indent=4)
        # print(df_order) 


        # quote = Quote(symbol='ACB', source='vci')
        # df_price = quote.history(start=START_DATE, end=END_DATE, interval='1D')
        # print(df_price.tail(10))
        # if not df_price.empty:
        #     df_price.to_json("ACB_price.json", orient='records', force_ascii=False, indent=4)
        # new_df = quote.intraday() # output: time, price, volume, match_type, id
        # if not new_df.empty:
        #     new_df.to_json("ACB_intra.json", orient='records', force_ascii=False, indent=4)
        # print(new_df)
        # depth_df = quote.price_depth() # output: price, volume, buy_volume, sell_volume, undefined_volume
        # print(depth_df)
        
        # fin = Finance(symbol='ACB', source='kbs', period='quarter')
        # fin_df = fin.ratio(lang='en', style='code', get_all=True) # chỉ lấy đc ở MAS
        # # print(fin_df)
        # if not fin_df.empty:
        #     fin_df.to_json("ACB.json", orient='records', force_ascii=False, indent=4)

        # macro = Macro(source='mbk')
        # macro = Macro()
        # df_fx = macro.currency().exchange_rate(start=START_DATE, end=END_DATE, period='day')
        # if not df_fx.empty:
        #     df_fx.to_json("usd_vnd.json", orient='records', force_ascii=False, indent=4)
        # df_gdp = macro.gdp(start='2025-01', end='2026-03')
        # df_gdp = macro.economy().gdp(start='2025-01', end='2026-03')
        # df_gdp = df_gdp.reset_index()
        # if not df_gdp.empty:
        #     df_gdp.to_json("vn_gdp.json", orient='records', force_ascii=False, indent=4)
        # df_cpi = macro.cpi(start="2025-01", end="2026-03", period="month")
        # df_cpi = df_cpi.reset_index()
        # if not df_cpi.empty:
        #     df_cpi.to_json("vn_cpi.json", orient='records', force_ascii=False, indent=4)
        # df_ip = macro.industry_prod(start="2025-01", end="2026-03", period="month")
        # df_ip = macro.economy().industry_prod(start="2025-01", end="2026-03", period="month")
        # df_ip = df_ip.reset_index()
        # if not df_ip.empty:
        #     df_ip.to_json("vn_ip.json", orient='records', force_ascii=False, indent=4)
        # df_retail = macro.retail(start="2025-01", end="2026-03", period="month")
        # df_retail = macro.economy().retail(start="2025-01", end="2026-03", period="month")
        # df_retail = df_retail.reset_index()
        # if not df_retail.empty:
        #     df_retail.to_json("vn_retail.json", orient='records', force_ascii=False, indent=4)
        # df_ie = macro.import_export(start="2025-01", end="2026-03", period="month")
        # df_ie = macro.economy().import_export(start="2025-01", end="2026-03", period="month")
        # df_ie = df_ie.reset_index()
        # if not df_ie.empty:
        #     df_ie.to_json("vn_ie.json", orient='records', force_ascii=False, indent=4)
        # df_fdi = macro.fdi(start="2025-01", end="2026-03", period="month")
        # df_fdi = macro.economy().fdi(start="2025-01", end="2026-03", period="month")
        # df_fdi = df_fdi.reset_index()
        # if not df_fdi.empty:
        #     df_fdi.to_json("vn_fdi.json", orient='records', force_ascii=False, indent=4)
        # df_ms = macro.money_supply(start="2025-01", end="2026-03", period="month")
        # df_ms = macro.money_supply(start="2025-01", end="2026-03", period="month")
        # df_ms = macro.economy().money_supply(start="2025-01", end="2026-03", period="month")
        # df_ms = df_ms.reset_index()
        # if not df_ms.empty:
        #     df_ms.to_json("vn_ms.json", orient='records', force_ascii=False, indent=4)
        # df_pl = macro.population_labor(start="2024", end="2026", period="year")
        # print(df_pl)

        # commodity = CommodityPrice(start=START_DATE, end=END_DATE)
        # df_fx = commodity.oil_crude()
        # df_fx = df_fx.reset_index()
        # if not df_fx.empty:
        #     df_fx.to_json("crude_oil.json", orient='records', force_ascii=False, indent=4)
        # print(df_fx)

        # f = Fund()
        # df_funds = f.listing(fund_type='BOND')
        # # if not df_funds.empty:
        # #     df_funds.to_json("funds.json", orient='records', force_ascii=False, indent=4)
        # print(df_funds)

        # l = Listing(source='vci')
        # df_ind = l.symbols_by_industries()
        # if not df_ind.empty:
        #     df_ind.to_json("groups_by_industries_new.json", orient='records', force_ascii=False, indent=4)
        # print(df_ind)

        # ['HOSE', 'VN30', 'VNMidCap', 'VNSmallCap', 'VNAllShare', 'VN100', 'ETF', 'HNX', 'HNX30', 'HNXCon', 'HNXFin', 'HNXLCap', 'HNXMSCap', 'HNXMan', 'UPCOM', 'FU_INDEX', 'FU_BOND', 'BOND', 'CW'
        # df_midcap = l.symbols_by_group(group='VNMidCap')
        # print(df_midcap)

        # df_oil = self._load_parquet(Path('data/parquet') / 'macro/crude_oil.parquet')
        # if not df_oil.empty:
        #     print(df_oil.tail(5))
        
        # df_f_flow = self._load_parquet(Path('data/parquet') / 'macro/foreign_flow.parquet')
        # if not df_f_flow.empty:
        #     print(df_f_flow.tail(5))

        # df_usd = self._load_parquet(Path('data/parquet') / 'macro/usd_vnd.parquet')
        # if not df_usd.empty:
        #     print(df_usd.tail(5))

        # df_vnindex = self._load_parquet(Path('data/parquet') / 'macro/vnindex.parquet')
        # if not df_vnindex.empty:
        #     print(df_vnindex.tail(5))

        # if not df_price.empty:
        #     df_group = {ticker: group for ticker, group in df_price.groupby('ticker')}
        #     # print(df_group['TCB'].tail(10))
        #     df_group['TCB'].to_json("TCB_price.json", orient='records', force_ascii=False, indent=4)

        # if not df_industry.empty:
        #     # print(df_industry.tail(10))
        #     df_industry.to_json("groups_by_industries_new.json", orient='records', force_ascii=False, indent=4)

        # if not df_intra.empty:
        #     df_group = {ticker: group for ticker, group in df_intra.groupby('ticker')}
        #     print(df_group['GMD'].tail(10))
        #     # df_intra.to_json("intraday.json", orient='records', force_ascii=False, indent=4)

        # if not df_foreign.empty:
        #     df_group = {ticker: group for ticker, group in df_foreign.groupby('ticker')}
        #     # print(df_group['TCB'].tail(10))
        #     df_group['TCB'].to_json("TCB_foreign.json", orient='records', force_ascii=False, indent=4)

        # if not df_index.empty:
        #     vn30_tickers = df_index[df_index['index_code'] == 'VN30']['ticker'].tolist()
        #     mid_tickers = df_index[df_index['index_code'] == 'VNMidCap']['ticker'].tolist()
        #     small_tickers = df_index[df_index['index_code'] == 'VNSmallCap']['ticker'].tolist()
        #     print(vn30_tickers)
            # df_index.to_json("df_index.json", orient='records', force_ascii=False, indent=4)

        # if not df_fin.empty:
        #     df_fin.to_json("df_fin.json", orient='records', force_ascii=False, indent=4)

        # if not df_board.empty:
        #     df_board.to_json("df_board.json", orient='records', force_ascii=False, indent=4)

        # omni = OmniFlowMatrix(data_frames, lookback_days=30)
        # test_tickers = ['VHM', 'VIC', 'HPG', 'DIG', 'GMD'] # Thử với VN30, MidCap và SmallCap
        # # test_tickers = ['GMD'] # Thử với VN30, MidCap và SmallCap
    
        
        # js_prod = self.load_json(Path('ps_sales_production.json'))
        # js_stag = self.load_json(Path('ps_sales_staging.json'))

        # l_prod, l_stag = set(), set()
        # for i in js_prod:
        #     l_prod.add(i['rowguid'])

        # for i in js_stag:
        #     l_stag.add(i['rowguid'])

        # l_prod = list(l_prod)
        # l_stag = list(l_stag)

        # # pd.DataFrame(l_prod).to_csv("sales_production.csv", orient='records', force_ascii=False, indent=4)
        # # pd.DataFrame(l_stag).to_csv("sales_staging.csv", orient='records', force_ascii=False, indent=4)

        # pd.DataFrame(l_prod).to_csv("sales_production.csv")
        # pd.DataFrame(l_stag).to_csv("sales_staging.csv")


    def _load_parquet(self, path):
        """Hàm đọc Parquet an toàn, tránh lỗi nếu file chưa tồn tại"""
        if path.exists():
            try: 
                return pd.read_parquet(path)
            except: 
                print(f"Could NOT read {path}")
                return pd.DataFrame()
        return pd.DataFrame()

    def load_json(self, path):
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                    # return data if isinstance(data, dict) else {}
                    return data
            except: 
                return {}
        return {}

if __name__ == "__main__":
    pipeline = VN30DataPipeline()
    pipeline.run_pipeline()