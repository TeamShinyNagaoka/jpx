# -*- coding: utf-8 -*-
import io
import os
import pickle
import math

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm

class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END_HIGH = '2019-12-01'
    TRAIN_END_LOW = '2018-12-31'
    TRAIN_END = '2018-12-31'
    # 評価期間開始日
    VAL_START = '2019-02-01'
    # 評価期間終了日
    VAL_END = '2019-12-01'
    # テスト期間開始日
    TEST_START = '2020-01-01'
    # 目的変数
    TARGET_LABELS = ['label_high_20', 'label_low_20']

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def getCodes(cls):
      return cls.codes

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            'stock_list': f'{dataset_dir}/stock_list.csv.gz',
            'stock_price': f'{dataset_dir}/stock_price.csv.gz',
            'stock_fin': f'{dataset_dir}/stock_fin.csv.gz',
            # 'stock_fin_price': f'{dataset_dir}/stock_fin_price.csv.gz',
            'stock_labels': f'{dataset_dir}/stock_labels.csv.gz',
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(cls.dfs[k].loc[:, "EndOfDayQuote Date"])
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(cls.dfs[k].loc[:, "base_date"])
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs['stock_list'].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list['prediction_target'] == True]['Local Code'].values

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        print(label,' Create Feature value')
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature['code'] == code]

            # 特定の銘柄コードのデータに絞る
            stock_labels = dfs['stock_labels'][dfs['stock_labels']['Local Code'] == code].copy()
            # 特定の目的変数に絞る
            labels = stock_labels[label]
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = {}
                _val_X = {}
                _test_X = {}

                _train_y = {}
                _val_y = {}
                _test_y = {}
                
                if label == 'label_high_20':
                  _train_X = feats[: cls.TRAIN_END_HIGH].copy()
                  _val_X = feats[cls.VAL_START : cls.VAL_END].copy()
                  _test_X = feats[cls.TEST_START :].copy()

                  _train_y = labels[: cls.TRAIN_END_HIGH].copy()
                  _val_y = labels[cls.VAL_START : cls.VAL_END].copy()
                  _test_y = labels[cls.TEST_START :].copy()
                elif label == 'label_low_20':
                  _train_X = feats[: cls.TRAIN_END_LOW].copy()
                  _val_X = feats[cls.VAL_START : cls.VAL_END].copy()
                  _test_X = feats[cls.TEST_START :].copy()

                  _train_y = labels[: cls.TRAIN_END_LOW].copy()
                  _val_y = labels[cls.VAL_START : cls.VAL_END].copy()
                  _test_y = labels[cls.TEST_START :].copy()
                else: 
                  _train_X = feats[: cls.TRAIN_END].copy()
                  _val_X = feats[cls.VAL_START : cls.VAL_END].copy()
                  _test_X = feats[cls.TEST_START :].copy()

                  _train_y = labels[: cls.TRAIN_END].copy()
                  _val_y = labels[cls.VAL_START : cls.VAL_END].copy()
                  _test_y = labels[cls.TEST_START :].copy()

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    #増加率の計算
    @classmethod
    def get_Rate_of_increase(cls, df):
      df_return_1 = df.shift(1)
      return (df - df_return_1) / df_return_1

    @classmethod
    def get_features_for_predict(cls, dfs, code, label, start_dt='2016-01-01'):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特定の銘柄コードのデータに絞る
        fin_data = dfs['stock_fin'][dfs['stock_fin']['Local Code'] == code]
        # 特徴量の生成対象期間を指定
        fin_data = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        #データを取得
        fin_feats = fin_data[['Result_FinancialStatement FiscalYear']].copy()
        fin_feats['Result_FinancialStatement NetSales'] = fin_data['Result_FinancialStatement NetSales']
        fin_feats['Result_FinancialStatement OperatingIncome'] = fin_data['Result_FinancialStatement OperatingIncome']
        fin_feats['Result_FinancialStatement OrdinaryIncome'] = fin_data['Result_FinancialStatement OrdinaryIncome']
        fin_feats['Result_FinancialStatement NetIncome'] = fin_data['Result_FinancialStatement NetIncome']
        fin_feats['Result_FinancialStatement TotalAssets'] = fin_data['Result_FinancialStatement TotalAssets']
        fin_feats['Result_FinancialStatement NetAssets'] = fin_data['Result_FinancialStatement NetAssets']
        fin_feats['Result_FinancialStatement CashFlowsFromOperatingActivities'] = fin_data['Result_FinancialStatement CashFlowsFromOperatingActivities']
        fin_feats['Result_FinancialStatement CashFlowsFromFinancingActivities'] = fin_data['Result_FinancialStatement CashFlowsFromFinancingActivities']
        fin_feats['Result_FinancialStatement CashFlowsFromInvestingActivities'] = fin_data['Result_FinancialStatement CashFlowsFromInvestingActivities']
        fin_feats['Forecast_FinancialStatement FiscalYear'] = fin_data['Forecast_FinancialStatement FiscalYear']
        fin_feats['Forecast_FinancialStatement NetSales'] = fin_data['Forecast_FinancialStatement NetSales']
        fin_feats['Forecast_FinancialStatement OperatingIncome'] = fin_data['Forecast_FinancialStatement OperatingIncome']
        fin_feats['Forecast_FinancialStatement OrdinaryIncome'] = fin_data['Forecast_FinancialStatement OrdinaryIncome']
        fin_feats['Forecast_FinancialStatement NetIncome'] = fin_data['Forecast_FinancialStatement NetIncome']
        fin_feats['Result_Dividend FiscalYear'] = fin_data['Result_Dividend FiscalYear']
        fin_feats['Result_Dividend QuarterlyDividendPerShare'] = fin_data['Result_Dividend QuarterlyDividendPerShare']
        fin_feats['Forecast_Dividend FiscalYear'] = fin_data['Forecast_Dividend FiscalYear']
        fin_feats['Forecast_Dividend QuarterlyDividendPerShare'] = fin_data['Forecast_Dividend QuarterlyDividendPerShare']
        fin_feats['Forecast_Dividend AnnualDividendPerShare'] = fin_data['Forecast_Dividend AnnualDividendPerShare']
        fin_feats['Result_FinancialStatement ReportType'] = fin_data['Result_FinancialStatement ReportType']
        fin_feats['Result_FinancialStatement ReportType'].replace(['Q1','Q2','Q3','Annual',],[0,1,2,3],inplace=True)
        # 欠損値処理
        fin_feats = fin_feats.fillna(0)

        # 特定の銘柄コードのデータに絞る
        price_data = dfs['stock_price'][dfs['stock_price']['Local Code'] == code]
        # 特徴量の生成対象期間を指定
        price_data = price_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        # 終値のみに絞る
        feats = price_data[['EndOfDayQuote ExchangeOfficialClose']].copy()
        #高値と安値の差額
        price_data['Stock price difference'] = price_data['EndOfDayQuote High'] - price_data['EndOfDayQuote Low']

        #騰落幅。前回終値と直近約定値の価格差
        feats['EndOfDayQuote ChangeFromPreviousClose'] = price_data['EndOfDayQuote ChangeFromPreviousClose']
        #騰落値
        feats['EndOfDayQuote RisingAndFallingPrices'] = price_data['EndOfDayQuote PreviousClose'] + price_data['EndOfDayQuote ChangeFromPreviousClose']
        #累積調整係数
        feats['EndOfDayQuote CumulativeAdjustmentFactor'] = price_data['EndOfDayQuote CumulativeAdjustmentFactor']
        #過去0,5,10,15,20日前の株価、出来高
        for nn in range(0, 21, 5):
          nn_str = str(nn)
          #高値
          feats['EndOfDayQuote High Return' + nn_str] = price_data['EndOfDayQuote High'].shift(nn)
          #安値
          feats['EndOfDayQuote Low Return' + nn_str] = price_data['EndOfDayQuote Low'].shift(nn)
          #始値	
          feats['EndOfDayQuote Open Return' + nn_str] = price_data['EndOfDayQuote Open'].shift(nn)
          #終値
          feats['EndOfDayQuote Close Return' + nn_str] = price_data['EndOfDayQuote Close'].shift(nn)
          #売買高
          feats['EndOfDayQuote Volume Return' + nn_str] = price_data['EndOfDayQuote Volume'].shift(nn)

        #銘柄情報
        list_data = dfs['stock_list'][dfs['stock_list']['Local Code'] == code].copy()
        #銘柄の33業種区分(コード)
        feats['33 Sector(Code)'] = list_data['33 Sector(Code)'].values[0]
        #銘柄の17業種区分(コード)
        feats['17 Sector(Code)'] = list_data['17 Sector(Code)'].values[0]
        #発行済株式数
        feats['IssuedShareEquityQuote IssuedShare'] = list_data['IssuedShareEquityQuote IssuedShare'].values[0]
        #Size Code (New Index Series)
        list_data['Size Code (New Index Series)'] = list_data['Size Code (New Index Series)'].replace('-', 0).astype(int)

        million = 1000000
        #来期の予測EPS（1株あたりの利益）
        forecast_EPS = (fin_feats['Forecast_FinancialStatement NetIncome'] * million) /  feats['IssuedShareEquityQuote IssuedShare']
        #feats['Forecast EPS'] = forecast_EPS
        #来期の予測PER（株価収益率）
        feats['Forecast PER ExchangeOfficialClose'] = price_data['EndOfDayQuote ExchangeOfficialClose'] / forecast_EPS
        #売買高加重平均価格(VWAP)
        feats['EndOfDayQuote VWAP'] = price_data['EndOfDayQuote VWAP']

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # データを結合
        feats = pd.concat([feats, fin_feats], axis=1).dropna()

        #決算種別gごとに分ける
        #Q1
        q1 = feats.loc[feats['Result_FinancialStatement ReportType'] == 0].copy()
        #Q2
        q2 = feats.loc[feats['Result_FinancialStatement ReportType'] == 1].copy()    
        #Q3
        q3 = feats.loc[feats['Result_FinancialStatement ReportType'] == 2].copy()   
        #Annual
        annual = feats.loc[feats['Result_FinancialStatement ReportType'] == 3].copy()

        #決算
        settlement = fin_data[['Forecast_FinancialStatement ReportType']].copy()
        settlement['Forecast_FinancialStatement ReportType'].replace(['Q1','Q2','Q3','Annual',],[0,1,2,3],inplace=True)
        settlement['Forecast_FinancialStatement FiscalYear'] = fin_data['Forecast_FinancialStatement FiscalYear']
        settlement['Forecast_FinancialStatement NetSales'] = fin_data['Forecast_FinancialStatement NetSales']
        settlement['Forecast_FinancialStatement OperatingIncome'] = fin_data['Forecast_FinancialStatement OperatingIncome']
        settlement['Result_FinancialStatement OperatingIncome'] = fin_data['Result_FinancialStatement OperatingIncome']
        #前の行と値が同じかどうか、同じならTrueを格納
        settlement['Forecast_FinancialStatement ReportType Flag'] = settlement['Forecast_FinancialStatement ReportType'].eq(settlement['Forecast_FinancialStatement ReportType'].shift(1))
        settlement['Forecast_FinancialStatement FiscalYear Flag'] = settlement['Forecast_FinancialStatement FiscalYear'].eq(settlement['Forecast_FinancialStatement FiscalYear'].shift(1))
        #0,1に変換
        settlement['Forecast_FinancialStatement ReportType Flag'] = settlement['Forecast_FinancialStatement ReportType Flag'] * 1
        settlement['Forecast_FinancialStatement FiscalYear Flag'] = settlement['Forecast_FinancialStatement FiscalYear Flag'] * 1
        #実行フラグを立てる
        settlement['Execution flag'] = ((settlement['Forecast_FinancialStatement ReportType Flag'] == 1) & (settlement['Forecast_FinancialStatement FiscalYear Flag'] == 1))
        #実行フラグがTrueなら値を格納
        settlement['Forecast_FinancialStatement NetSales Shift'] = 0
        settlement['Forecast_FinancialStatement NetSales Shift'].where(settlement['Execution flag'] != True, settlement['Forecast_FinancialStatement NetSales'].shift(1), inplace=True)
        settlement['Forecast_FinancialStatement OperatingIncome Shift'] = 0
        settlement['Forecast_FinancialStatement OperatingIncome Shift'].where(settlement['Execution flag'] != True, settlement['Forecast_FinancialStatement OperatingIncome'].shift(1), inplace=True)
        settlement['Result_FinancialStatement OperatingIncome Shift'] = 0
        settlement['Result_FinancialStatement OperatingIncome Shift'].where(settlement['Execution flag'] != True, settlement['Result_FinancialStatement OperatingIncome'].shift(1), inplace=True)

        #負債
        liabilities = feats['Result_FinancialStatement TotalAssets'] - feats['Result_FinancialStatement NetAssets']
        #AnnualのEPS（1株当たり利益）
        annual_EPS = (annual['Result_FinancialStatement NetIncome'] * million) / list_data['IssuedShareEquityQuote IssuedShare'].values[0]
        if label == 'label_high_20':
          #Size Code (New Index Series)
          feats['Size Code (New Index Series)'] = list_data['Size Code (New Index Series)'].values[0]
          #Annual純利益増加率
          annual['Annual Net income increase rate'] = cls.get_Rate_of_increase(annual['Result_FinancialStatement NetIncome'])
          #欠損値処理を行います。
          annual = annual.replace([np.nan], 0)
          feats['Annual Net income increase rate'] = annual['Annual Net income increase rate']
          #Q1,Q2,Q3,Annualの営業利益増加率
          q1['Q1 Operating income increase rate'] = cls.get_Rate_of_increase(q1['Result_FinancialStatement OperatingIncome'])
          q2['Q2 Operating income increase rate'] = cls.get_Rate_of_increase(q2['Result_FinancialStatement OperatingIncome'])
          q3['Q3 Operating income increase rate'] = cls.get_Rate_of_increase(q3['Result_FinancialStatement OperatingIncome'])
          annual['Annual Operating income increase rate'] = cls.get_Rate_of_increase(annual['Result_FinancialStatement OperatingIncome'])
          #欠損値処理を行います。
          q1 = q1.replace([np.nan], 0)
          q2 = q2.replace([np.nan], 0)
          q3 = q3.replace([np.nan], 0)
          annual = annual.replace([np.nan], 0)
          feats['Q1 Operating income increase rate'] = q1['Q1 Operating income increase rate']
          feats['Q2 Operating income increase rate'] = q2['Q2 Operating income increase rate']
          feats['Q3 Operating income increase rate'] = q3['Q3 Operating income increase rate']
          feats['Annual Operating income increase rate'] = annual['Annual Operating income increase rate']
          #Q1,Q2,Q3,Annualの当期純利益増加率
          q1['Q1 Net income increase rate'] = cls.get_Rate_of_increase(q1['Result_FinancialStatement NetIncome'])
          q2['Q2 Net income increase rate'] = cls.get_Rate_of_increase(q2['Result_FinancialStatement NetIncome'])
          q3['Q3 Net income increase rate'] = cls.get_Rate_of_increase(q3['Result_FinancialStatement NetIncome'])
          annual['Annual Net income increase rate'] = cls.get_Rate_of_increase(annual['Result_FinancialStatement NetIncome'])
          #欠損値処理を行います。
          q1 = q1.replace([np.nan], 0)
          q2 = q2.replace([np.nan], 0)
          q3 = q3.replace([np.nan], 0)
          annual = annual.replace([np.nan], 0)
          feats['Q1 Net income increase rate'] = q1['Q1 Net income increase rate']
          feats['Q2 Net income increase rate'] = q2['Q2 Net income increase rate']
          feats['Q3 Net income increase rate'] = q3['Q3 Net income increase rate']
          feats['Annual Net income increase rate'] = annual['Annual Net income increase rate']
          #PER（株価収益率）
          feats['Annual PER'] = price_data['EndOfDayQuote ExchangeOfficialClose'] / annual_EPS        
          #決算営業利益増加率
          feats['Settlement operating income increase rate'] = (settlement['Result_FinancialStatement OperatingIncome'] - settlement['Result_FinancialStatement OperatingIncome Shift']) / settlement['Result_FinancialStatement OperatingIncome Shift']  

          #欠損値処理を行います。
          feats = feats.replace([np.nan], -99999)

          #来期決算種別
          feats['Forecast_FinancialStatement ReportType'] = settlement['Forecast_FinancialStatement ReportType']
          #来期の予想決算売上高増加率
          feats['Expected settlement of accounts for the next fiscal year Sales increase rate'] = (settlement['Forecast_FinancialStatement NetSales'] - settlement['Forecast_FinancialStatement NetSales Shift']) / settlement['Forecast_FinancialStatement NetSales Shift']
          #売上高増加率
          feats['Sales growth rate'] = cls.get_Rate_of_increase(feats['Result_FinancialStatement NetSales'])
          #営業利益増加率
          feats['Operating income increase rate'] = cls.get_Rate_of_increase(feats['Result_FinancialStatement OperatingIncome'])
          #経常利益増加率
          feats['Ordinary income increase rate'] = cls.get_Rate_of_increase(feats['Result_FinancialStatement OrdinaryIncome'])
          #BPS（1株あたりの純資産）
          BPS = (feats['Result_FinancialStatement NetAssets'] * million) /  feats['IssuedShareEquityQuote IssuedShare']
          #PBR（株価純資産倍率）
          feats['PBR'] = feats['EndOfDayQuote ExchangeOfficialClose'] / BPS
          #CFPS（1株あたりのキャッシュフロー）
          CFPS = (feats['Result_FinancialStatement CashFlowsFromOperatingActivities'] * million) / feats['IssuedShareEquityQuote IssuedShare']
          #PCFR(株価キャッシュフロー倍率)
          feats['PCFR'] = feats['EndOfDayQuote ExchangeOfficialClose'] / CFPS
          #来期の予測配当利回り
          feats['Forecast Dividend yield'] = feats['Forecast_Dividend AnnualDividendPerShare'] / feats['EndOfDayQuote ExchangeOfficialClose']
          #時価総額
          feats['Market capitalization'] = (feats['EndOfDayQuote ExchangeOfficialClose'] * million) * feats['IssuedShareEquityQuote IssuedShare']
          #キャッシュフローマージン
          feats['Forecast Cash flow margin'] = feats['Result_FinancialStatement CashFlowsFromOperatingActivities'] / feats['Forecast_FinancialStatement NetSales']
          #高値と安値の5日間の差額の平均
          feats['Stock price difference Mean 5'] = price_data['Stock price difference'].rolling(5).mean()
          #5日間平均から当日株価を引く
          EndOfDayQuote_ExchangeOfficialClose_Mean_5 = price_data['EndOfDayQuote ExchangeOfficialClose'].rolling(5).mean()
          feats['Subtract the current days stock price from the 5-day average'] = EndOfDayQuote_ExchangeOfficialClose_Mean_5 - feats['EndOfDayQuote ExchangeOfficialClose']
          #売上高に対しての負債割合
          feats['Ratio of sales to liabilities'] = liabilities / feats['Result_FinancialStatement NetSales']
          #負債増加率
          feats['Debt growth rate'] = cls.get_Rate_of_increase(liabilities)
          #終値の20営業日ボラティリティ
          feats['20 business days volatility'] = (np.log(price_data['EndOfDayQuote ExchangeOfficialClose']).diff().rolling(20).std())
          #終値の40営業日ボラティリティ
          feats['40 business days volatility'] = (np.log(price_data['EndOfDayQuote ExchangeOfficialClose']).diff().rolling(40).std())
          #終値の60営業日ボラティリティ
          feats['60 business days volatility'] = (np.log(price_data['EndOfDayQuote ExchangeOfficialClose']).diff().rolling(60).std())
          #終値の20営業日リターン
          feats['20 business day return'] = price_data['EndOfDayQuote ExchangeOfficialClose'].pct_change(20)

          #ドロップ
          for nn in range(0, 21, 5):
            nn_str = str(nn)
            feats = feats.drop(['EndOfDayQuote High Return' + nn_str], axis=1)
            feats = feats.drop(['EndOfDayQuote Low Return' + nn_str], axis=1)
            feats = feats.drop(['EndOfDayQuote Open Return' + nn_str], axis=1)
            feats = feats.drop(['EndOfDayQuote Close Return' + nn_str], axis=1)
        elif label == 'label_low_20':
          #Q1,Q2,Q3,Annualの売上高増加率
          q1['Q1 Sales growth rate'] = cls.get_Rate_of_increase(q1['Result_FinancialStatement NetSales'])
          q2['Q2 Sales growth rate'] = cls.get_Rate_of_increase(q2['Result_FinancialStatement NetSales'])
          q3['Q3 Sales growth rate'] = cls.get_Rate_of_increase(q3['Result_FinancialStatement NetSales'])
          annual['Annual Sales growth rate'] = cls.get_Rate_of_increase(annual['Result_FinancialStatement NetSales'])
          #欠損値処理を行います。
          q1 = q1.replace([np.nan], 0)
          q2 = q2.replace([np.nan], 0)
          q3 = q3.replace([np.nan], 0)
          annual = annual.replace([np.nan], 0)
          feats['Q1 Sales growth rate'] = q1['Q1 Sales growth rate']
          feats['Q2 Sales growth rate'] = q2['Q2 Sales growth rate']
          feats['Q3 Sales growth rate'] = q3['Q3 Sales growth rate']
          feats['Annual Sales growth rate'] = annual['Annual Sales growth rate']
          #Annual財務キャッシュフロー増加率
          annual['Annual Rate of increase in financial cash flow'] = cls.get_Rate_of_increase(annual['Result_FinancialStatement CashFlowsFromFinancingActivities'])
          #欠損値処理を行います。
          annual = annual.replace([np.nan], 0)
          feats['Annual Rate of increase in financial cash flow'] = annual['Annual Rate of increase in financial cash flow']
          #Annual EPS（1株当たり利益）
          feats['Annual EPS'] = annual_EPS

          #欠損値処理を行います。
          feats = feats.replace([np.nan], -99999)

          #来期の予想決算営業利益増加率
          feats['Expected settlement of accounts for the next fiscal year Operating income increase rate'] = (settlement['Forecast_FinancialStatement OperatingIncome'] - settlement['Forecast_FinancialStatement OperatingIncome Shift']) / settlement['Forecast_FinancialStatement OperatingIncome Shift'] 
          #負債比率
          feats['Debt ratio'] = liabilities / feats['Result_FinancialStatement NetAssets']
          #利益率
          Profit_rate = feats['Result_FinancialStatement NetIncome'] / feats['Result_FinancialStatement NetSales']
          #利益率増加率
          feats['Profit margin increase rate'] = cls.get_Rate_of_increase(Profit_rate)
          #自己資本比率
          feats['equity_ratio'] = feats['Result_FinancialStatement NetAssets'] / feats['Result_FinancialStatement TotalAssets']
          #純利益増加率
          feats['Net income increase rate'] = cls.get_Rate_of_increase(feats['Result_FinancialStatement NetIncome'])
          #EPS（1株当たり利益）
          EPS = feats['Result_FinancialStatement NetIncome'] / feats['IssuedShareEquityQuote IssuedShare']
          #PER（株価収益率）
          PER = price_data['EndOfDayQuote ExchangeOfficialClose'] / EPS
          #目標株価
          feats['Target stock price'] = EPS * PER
          #ドロップ
          feats = feats.drop(['EndOfDayQuote RisingAndFallingPrices','Result_FinancialStatement TotalAssets',
                              'Result_FinancialStatement CashFlowsFromOperatingActivities',
                              'Forecast_Dividend QuarterlyDividendPerShare','Result_FinancialStatement CashFlowsFromFinancingActivities',
                              'Forecast_FinancialStatement FiscalYear','Result_Dividend FiscalYear',  
                              'Forecast_FinancialStatement NetIncome', 'Forecast_FinancialStatement OperatingIncome',
                              'Forecast_FinancialStatement NetSales','Result_FinancialStatement OrdinaryIncome',], axis=1)
        
        feats = feats.drop(['EndOfDayQuote ExchangeOfficialClose',], axis=1)
        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf, np.nan], 0)

        # 銘柄コードを設定
        feats['code'] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt) :]

        return feats

    @classmethod
    def create_model(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            lgb.LGBMRegressor
        """
        # 特徴量を取得
        buff = []
        print(label,' Get Feature value')
        for code in tqdm(codes):
            buff.append(cls.get_features_for_predict(cls.dfs,code,label))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, _, _, _, _ = cls.get_features_and_label(dfs, codes, feature, label)
        # モデル作成
        if label == 'label_high_20':
          model = lgb.LGBMRegressor(
            learning_rate=0.1,
            lambda_l1=0.72021,
            lambda_l2=0.1001,
            num_leaves=31,
            feature_fraction=1,
            n_estimators=148,
            max_bin= 255,
            min_child_samples=230,
            random_state=0)
        elif label == 'label_low_20':
          model = lgb.LGBMRegressor(
            learning_rate=0.1,
            lambda_l1=0.0015,
            lambda_l2=9.71016,
            num_leaves=51,
            feature_fraction=0.63,
            n_estimators=60,
            max_bin= 125,
            min_child_samples=52,
            random_state=0)
        else:
          model = lgb.LGBMRegressor(random_seed=0)
        model.fit(train_X, train_y)
        return model

    @classmethod
    def save_model(cls, model, label, model_path='../model'):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f'my_model_{label}.pkl'), 'wb') as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_model(cls, model_path='../model', labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        try:
            for label in labels:
                m = os.path.join(model_path, f'my_model_{label}.pkl')
                with open(m, 'rb') as f:
                    # pickle形式で保存されているモデルを読み込み
                    cls.models[label] = pickle.load(f)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path='../model'
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            model = cls.create_model(cls.dfs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff_high = []
        buff_low = []
        feats = {}
        print('Create Feature value')
        for code in tqdm(codes):
            buff_high.append(cls.get_features_for_predict(cls.dfs, code,cls.TARGET_LABELS[0], start_dt))
            buff_low.append(cls.get_features_for_predict(cls.dfs, code,cls.TARGET_LABELS[1], start_dt))
        feats[cls.TARGET_LABELS[0]] = pd.concat(buff_high)
        feats[cls.TARGET_LABELS[1]] = pd.concat(buff_low)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats[cls.TARGET_LABELS[0]].loc[:, ['code']].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, 'code'] = df.index.strftime('%Y-%m-%d-') + df.loc[:, 'code'].astype(str)

        # 出力対象列を定義
        output_columns = ['code']

        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            df[label] = cls.models[label].predict(feats[label])
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()