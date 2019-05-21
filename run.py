# ------------------------- IMPORT LIBRARIES --------------------
import datetime
import pandas as pd
import numpy as np
# This line below  resolve mismatched pandas and pandas-datareader. Uncomment if necessary.
# pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from calendar import isleap
import warnings
import time
import seaborn as sns
from time import strftime, localtime
warnings.filterwarnings("ignore")

# ------------------------- GLOBAL PARAMETERS -------------------------

# Set date format for graph plotting
YRMTH_FMT = mdates.DateFormatter('%b %Y')

# Set the initial fund allocation
ACCOUNT_FUND = 100000.00
# Set start and ent time for data download
START = datetime.datetime(1993, 7, 1)
END = datetime.datetime(2018, 8, 30)

# The KPIs
KPI = ['Win %', 'Win to Loss Ratio', 'Mean Return/Trade %', 'Max Consecutive Losers', 'Max dd', 'CAGR',
               'Lake ratio', 'Gain to Pain']

# Set the train/test set split fraction
TRAIN_FRACTION = 0.6

# Stocks to download
DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DWDP', 'XOM', 'GE', 'GS',
          'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UTX',
          'UNH', 'VZ', 'V', 'WMT']
DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney',
         'Dupont','ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
         'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble','Travelers',
         'United Technologies','UnitedHealth Group','Verizon Communications','Visa','Wal Mart']

# Set period for EMA and ATR
N = np.array([21, 45])
# Set the type of pyramiding strategies
PYRAMIDING = np.array(["Upright", "Inverted", "Reflecting"])
# Set position sizing methods
POSITION_SIZING = np.array(["Pct Volatility", "Markets Money","MultiTier"])
# Percentage of equity to risk
RISK_PCT =0.1
#Set the fund amount for the next level risk level calculation in Multi-tier position sizing
EQUITY_LEVEL_FUND = 50000
# Set price impact for slippage
PRICE_IMPACT = 0.1

# ------------------------------ CLASSES ---------------------------------

class DataRetrieval:

    def __init__(self):
        # Initiate component data downloads
        self._dji_components_data()

    def _get_daily_data(self, symbol):
        """
        This class prepares data by downloading historical data from Yahoo Finance,

        """
        flag = False
        # Set counter for download trial
        counter = 0

        # Safety loop to handle unstable Yahoo finance download
        while not flag and counter < 6:
            try:
                # Define data range
                yf.pdr_override()
                daily_price = pdr.get_data_yahoo(symbol.strip('\n'), START, END)
                flag = True
            except:
                flag = False
                counter += 1
                time.sleep(10)
                if counter < 6:
                    continue
                else:
                    raise Exception("Yahoo finance is down, please try again later. ")

        return daily_price

    def _dji_components_data(self):
        """
        This function download all components data and assembles the required OHLCV data into respective data
        """

        for i in DJI:
            print "Downloading {}'s historical data".format(DJI_N[DJI.index(i)])
            if i == DJI[0]:
                df = self._get_daily_data(i)
                self.components_df_c = pd.DataFrame(index=df.index, columns=DJI)
                self.components_df_h = pd.DataFrame(index=df.index, columns=DJI)
                self.components_df_l = pd.DataFrame(index=df.index, columns=DJI)
                self.components_df_v = pd.DataFrame(index=df.index, columns=DJI)
                # Since this span 25 years of data, many corporate actions could have happened,
                # adjusted closing price is used instead
                self.components_df_c[i] = df["Adj Close"]
                self.components_df_h[i] = df["High"]
                self.components_df_l[i] = df["Low"]
                self.components_df_v[i] = df["Volume"]
            else:
                stock_df = self._get_daily_data(i)
                self.components_df_c[i] = stock_df["Adj Close"]
                self.components_df_h[i] = stock_df["High"]
                self.components_df_l[i] = stock_df["Low"]
                self.components_df_v[i] = stock_df["Volume"]

    def _train_test_split(self):

        """
        This function split the downloaded data into train test sets determined the set fraction
        """

        train_set_length = int(round(len(self.components_df_c) * TRAIN_FRACTION))
        test_set_length = len(self.components_df_c) - train_set_length
        self.train_c = self.components_df_c[0:train_set_length]
        self.test_c = self.components_df_c[-test_set_length:]
        self.train_h = self.components_df_h[0:train_set_length]
        self.test_h = self.components_df_h[-test_set_length:]
        self.train_l = self.components_df_l[0:train_set_length]
        self.test_l = self.components_df_l[-test_set_length:]
        self.train_v = self.components_df_v[0:train_set_length]
        self.test_v = self.components_df_v[-test_set_length:]

    def get_full_dataset(self):
        """
        This function returns the full length data
        """
        return self.components_df_c, self.components_df_h, self.components_df_l, self.components_df_v

    def get_split_dataset(self):
        """
        This function returns the split dataset
        """
        self._train_test_split()
        return self.train_c, self.test_c, self.train_h, self.test_h, self.train_l, self.test_l, self.train_v, self.test_v

    def get_DJI(self):
        """
        This function returns the full length DJIA index data
        :return:
        """
        return self._get_daily_data("^DJI")

class MathCalc:
    """
    This class performs all the mathematical calculations
    """

    @staticmethod
    def colrow(i):
        """
        This function calculate the row and columns index number based on the total number of subplots in the plot.

        Return:
             row: axis's row index number
             col: axis's column index number
        """

        # Do odd/even check to get col index number
        if i % 2 == 0:
            col = 0
        else:
            col = 1
        # Do floor division to get row index number
        row = i // 2

        return col, row

    @staticmethod
    def diff_year(start_date, end_date):
        """
        This function computes the fractional year for CAGR calculation
        """
        diffyears = end_date.year - start_date.year
        difference = end_date - start_date.replace(end_date.year)
        days_in_year = isleap(end_date.year) and 366 or 365
        difference_in_years = diffyears + (difference.days + difference.seconds / 86400.0) / days_in_year
        return difference_in_years

    @staticmethod
    def cagr(portfolio_value):
        """
        This function computes CAGR
        """
        st = portfolio_value.index[0]
        en = portfolio_value.index[-1]
        num_year = MathCalc.diff_year(st, en)
        return (portfolio_value[-1] / portfolio_value[0]) ** (1.0 / float(num_year)) - 1

    @staticmethod
    def calc_return(period):
        """
        This function compute the return of a series
        """
        period_return = period / period.shift(1) - 1
        return period_return[1:len(period_return)]

    @staticmethod
    def max_drawdown(r):
        """
        This function calculates maximum drawdown occurs in a series of cummulative returns
        """
        dd = r.div(r.cummax()).sub(1)
        maxdd = dd.min()
        return round(maxdd, 2)

    @staticmethod
    def calc_gain_to_pain(returns):
        """
        This function computes the gain to pain ratio given a series of profits and losses
        """
        sum_returns = returns.sum()
        sum_neg_months = abs(returns[returns < 0].sum())
        gain_to_pain = sum_returns / sum_neg_months

        return gain_to_pain

    @staticmethod
    def calc_lake_ratio(series):

        """
        This function computes lake ratio
        """
        water = 0
        earth = 0
        series = series.dropna()
        water_level = []
        for i, s in enumerate(series):
            if i == 0:
                peak = s
            else:
                peak = np.max(series[0:i])
            water_level.append(peak)
            if s < peak:
                water = water + peak - s
            earth = earth + s
        return water / earth

    @staticmethod
    def construct_book(stocks_values):
        """
        This function construct the trading book for stock trading
        """
        portfolio = pd.DataFrame(index=stocks_values.index, columns=["Values", "Returns", "CumReturns"])
        portfolio["Values"] = stocks_values
        portfolio["Returns"] = portfolio["Values"] / portfolio["Values"].shift(1) - 1
        portfolio["CumReturns"] = portfolio["Returns"].add(1).cumprod().fillna(1)

        return portfolio

    @staticmethod
    def winpct(realized_pnl):
        return float(len(realized_pnl[realized_pnl > 0])) / float(len(realized_pnl[realized_pnl != 0])) * 100

    @staticmethod
    def winloss(realized_pnl):
        """
        This function calculates win to loss ratio
        """
        return float(realized_pnl[realized_pnl > 0].mean()) / abs(float(realized_pnl[realized_pnl < 0].mean()))

    @staticmethod
    def meanreturn_trade(trade_returns):
        """
        This function calculates the mean of all trade returns
        """
        trade_returns = trade_returns.dropna()
        return trade_returns.mean() * 100

    @staticmethod
    def longestconsecutive_loss(arr):
        """
        This function computes the longest losing streak
        """

        # remove all non trading activities
        arr = filter(lambda a: a != 0, arr)

        n = len(arr)
        # Initialize result
        res = 0

        # Traverse array
        for i in range(n):

            # Count of current
            # non-negative integers
            curr_count = 0
            while (i < n and arr[i] < 0):
                curr_count += 1
                i += 1

            # Update result if required.
            res = max(res, curr_count)

        return res

    @staticmethod
    def calc_kpi(portfolio, stocks_values, symbol):
        """
        This function calculates individual portfolio KPI related its risk-return profile
        """
        kpi = pd.DataFrame(index=[symbol], columns=KPI)

        kpi['Win %'] = MathCalc.winpct(stocks_values["Profit & Loss"])
        kpi['Win to Loss Ratio'] = MathCalc.winloss(stocks_values["Profit & Loss"])
        kpi['Mean Return/Trade %'] = MathCalc.meanreturn_trade(stocks_values["Trade Returns"])
        kpi['Max Consecutive Losers'] = MathCalc.longestconsecutive_loss(stocks_values["Profit & Loss"])
        kpi['CAGR'].iloc[0] = MathCalc.cagr(portfolio["Values"])
        kpi['Max dd'].iloc[0] = MathCalc.max_drawdown(portfolio["CumReturns"])
        kpi['Lake ratio'].iloc[0] = MathCalc.calc_lake_ratio(portfolio['CumReturns'])
        kpi['Gain to Pain'].iloc[0] = MathCalc.calc_gain_to_pain(portfolio['Returns'])

        return kpi

class Trading:

    def __init__(self, daily_c, daily_h, daily_l, daily_v, symbol):
        self.daily_c = daily_c
        self.daily_h = daily_h
        self.daily_l = daily_l
        self.daily_v = daily_v
        self.symbol = symbol

    def calc_ema(self, period):
        """
        This function calculates EMA

        """
        return pd.ewma(self.daily_c, span=period)

    def calc_avg_truerange(self, period):
        """
        This function calculates average true range (ATR)
        """

        truerange = self.daily_h - self.daily_l
        return truerange.rolling(window=period).mean()

    def calc_ema_truerange(self, period):
        """
        This function calculates average true range (ATR)
        """
        truerange = self.daily_h - self.daily_l
        return pd.ewma(truerange, span=period)

    def generate_trading_signal_ema(self, n):
        """
        This function generate trading signal based on price data breaking through 0.5 * ATR +/- EMA
        """

        # Call up the indicators calculations and get their values
        ema = self.calc_ema(period=n)
        self.avg_tr = self.calc_avg_truerange(period=n)

        # Creating the signal series defined by ratios values
        self.signal_ema = self.daily_c.copy()
        self.signal_ema[:] = 0
        self.ema_high = ema + (0.5 * self.avg_tr)
        self.ema_low = ema - (0.5 * self.avg_tr)
        self.signal_ema[self.daily_c > self.ema_high] = 1
        self.signal_ema[self.daily_c < self.ema_low] = -1

    def get_ema_signals(self, n):
        """
        This function returns signal generated by EMA-ATR
        """
        self.generate_trading_signal_ema(n)
        return self.signal_ema

    def commission(self, num_share, share_value):
        """
        This function computes commission fee of every trade
        https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks1
        """

        comm_fee = 0.005 * num_share
        max_comm_fee = 0.005 * share_value

        if num_share < 1.0:
            comm_fee = 1.0
        elif comm_fee > max_comm_fee:
            comm_fee = max_comm_fee

        return comm_fee

    def slippage_price(self, order, price, stock_quantity, day_volume):
        """
        This function performs slippage price calculation using Zipline's volume share model
        https://www.zipline.io/_modules/zipline/finance/slippage.html
        """

        volumeShare = stock_quantity / float(day_volume)
        impactPct = volumeShare ** 2 * PRICE_IMPACT

        if order == 1:
            slipped_price = price * (1 + impactPct)
        else:
            slipped_price = price * (1 - impactPct)

        # print order, " price: ", price, "slipped price: ", slipped_price
        return slipped_price

    def position_sizing(self, sizing, account_equity, account_profit):
        """
        This function calculates position allocation based on the following methods:
        Percent volatility: using a percent of equity based on the margin on the volatility of the underlying asset
        rather than the risk.
        Market's money: risking percentage of one 's starting equity and a different percentage of one's profit.
        Multi-tier position sizing: Risking 1% until one's equity reaches a certain level and then risking another
        percentage at the second level.
        """

        if sizing == "Pct Volatility":
            stock_volatility = self.calc_ema_truerange(10)
            position_size = account_equity * RISK_PCT / stock_volatility[-1]

        elif sizing == "Markets Money":
            position_size = (ACCOUNT_FUND * RISK_PCT) + (account_profit * RISK_PCT)

        elif sizing == "MultiTier":
            # This position with add 0.5% to original 1% fund to risk for every 50,000 (EQUITY_LEVEL_FUND)
            # increase of equity value (fund size)
            if account_equity > ACCOUNT_FUND:
                position_size = ACCOUNT_FUND * RISK_PCT
                equity_level = account_equity // EQUITY_LEVEL_FUND
                remainder_fund = account_equity % EQUITY_LEVEL_FUND
                levelled_risk = RISK_PCT + 0.005
                # print "equity_level-1", equity_level-1
                for e in range(int(equity_level - 1)):
                    position_size = position_size + (EQUITY_LEVEL_FUND * levelled_risk)
                    levelled_risk = levelled_risk + 0.005
                position_size = position_size + (remainder_fund * levelled_risk)
            else:
                position_size = account_equity * RISK_PCT
        # print "Postion size calculated:", position_size
        return position_size

    def pyramiding_strategies(self, sizing, p_strategy, account_equity, account_profit):
        """
        This function allocate funds based different pyramiding strategies from sizing determined by position sizing
        """

        trading_fund = self.position_sizing(sizing, account_equity, account_profit)
        trading_fund = int(round(trading_fund))

        if p_strategy == "Upright":
            trading_fund_fr1 = int(round(trading_fund * 0.5))
            trading_fund_fr2 = int(round(trading_fund * 0.25))
            trading_fund_fr3 = int(round(trading_fund * 0.15))
            trading_fund_fr4 = trading_fund - trading_fund_fr1 - trading_fund_fr2 - trading_fund_fr3
            return [trading_fund_fr1, trading_fund_fr2, trading_fund_fr3, trading_fund_fr4]

        elif p_strategy == "Inverted":
            trading_fund_fr = int(round(trading_fund * 0.25))
            trading_fund_fr1 = trading_fund - (trading_fund_fr * 3)
            return [trading_fund_fr1, trading_fund_fr, trading_fund_fr, trading_fund_fr]

        elif p_strategy == "Reflecting":
            trading_fund_fr1 = int(round(trading_fund * 0.5))
            trading_fund_fr2 = int(round(trading_fund * 0.25))
            trading_fund_fr3 = int(round(trading_fund * 0.15))
            trading_fund_fr4 = trading_fund - trading_fund_fr1 - trading_fund_fr2 - trading_fund_fr3
            upright_pyramid = [trading_fund_fr1, trading_fund_fr2, trading_fund_fr3, trading_fund_fr4]

            return upright_pyramid + [-1 * i for i in upright_pyramid[::-1]]

    @staticmethod
    def compile_strategy_list():
        """
        This function assembles strategies into a list
        """
        strategy_list = []
        for n in N:
            for s in POSITION_SIZING:
                for p in PYRAMIDING:
                    string_value = "{}-{}-{}".format(p, s, n)
                    strategy_list.append(string_value)
        return strategy_list

    @staticmethod
    def find_robust_strategy(strategy_list, kpi_mean):
        """
        This funtion finds the most robust strategy among the 18 strategies by aggregating all rated KPIs.
        All KPIs are normalized to have values range from 0 to 1. The most robust strategy is the one with the highest
        summed values. For Maximum consecutive losers and lake ratio, where the lesser the value the better it is,
        reversed rating (1 to 0 ) is used instead.
        """
        kpi_up = ['Win %', 'Win to Loss Ratio', 'Mean Return/Trade %', 'Max dd', 'CAGR', 'Gain to Pain']
        kpi_dn = ['Max Consecutive Losers', 'Lake ratio']
        kpi_mean_rated = pd.DataFrame(index=strategy_list, columns=KPI)

        for u in kpi_up:
            min_k = kpi_mean[u].min()
            max_k = kpi_mean[u].max()
            diff_k = max_k - min_k
            for i, k in enumerate(kpi_mean[u]):
                kpi_mean_rated[u].iloc[i] = float((kpi_mean[u][i] - min_k)) / float(diff_k)

        for d in kpi_dn:
            min_k = kpi_mean[d].min()
            max_k = kpi_mean[d].max()
            diff_k = max_k - min_k
            for i, k in enumerate(kpi_mean[d]):
                kpi_mean_rated[d].iloc[i] = float((max_k - kpi_mean[d][i])) / float(diff_k)

        robust_strategy = kpi_mean_rated.sum(axis=1).sort_values(ascending=False).index[0]
        # Extracting pyramiding strategy, sizing method and EMA from the found robus strategy.
        pyramid_strategy = robust_strategy.split('-')[0]
        sizing_method = robust_strategy.split('-')[1]
        ema_period = robust_strategy.split('-')[2]

        return kpi_mean_rated, robust_strategy, pyramid_strategy, sizing_method, ema_period

    @staticmethod
    def stocks_corr(test_returns, test_c):
        """
        This function calculate the correlation coefficient between a portfolio returns and a stock returns
        """

        components_corr = pd.Series(index=DJI)
        for stock in DJI:
            stock_return = MathCalc.calc_return(test_c[stock])
            components_corr[stock] = test_returns[1:].corr(stock_return)
        return components_corr.sort_values(ascending=True)

    def execute_trading(self, sizing, pyramid, signal):
        """
        This function performs long only trades.
        """
        account_value = ACCOUNT_FUND
        pyramiding = self.pyramiding_strategies(sizing, pyramid, account_value, 0)

        stocks_values = pd.DataFrame(index=self.daily_c.index,
                                     columns=["Stock Price", "Stock Quantity", "Profit & Loss", "Trade Returns",
                                              "Portfolio Value",
                                              "Account Value"])
        stock_quantity = 0
        pyramiding = [n for n in pyramiding if n > 0]
        buy_length = len([n for n in pyramiding if n > 0])
        exit_pyramiding = []
        pyramiding_sell = False
        account_profit_holder = 0
        account_equity_holder = 0

        # Slide through the timeline
        for d in self.daily_c.index:
            # if this is the first day and signal is buy
            if d == self.daily_c.index[0] and signal.loc[d] == 1:
                # Do floor division to get stock quantity since fractional stock is not available in practice
                stock_quantity = pyramiding[0] // self.daily_c.loc[d]
                exit_pyramiding.append(stock_quantity)
                # Slippage occured and the buy price is no longer the desired price
                slipped_price = self.slippage_price(signal.loc[d], self.daily_c.loc[d], stock_quantity,
                                                    self.daily_v.loc[d])
                # Portfolio recorded the value calculated with market price,
                # not slipped price, as that is the actual market value
                portfolio_value = pyramiding[0]
                realized_pnl = 0.0
                realized_ret = float('nan')
                avg_buy_price = slipped_price
                # buy_position is used to signal the position within the pyramid during buying
                buy_position = 1
                commission_cost = self.commission(stock_quantity, portfolio_value)
                account_value = account_value - portfolio_value - commission_cost

            # if this the first day and no buy signal
            elif d == self.daily_c.index[0] and signal.loc[d] != 1:
                stock_quantity = 0
                portfolio_value = 1
                realized_pnl = 0.0
                realized_ret = float('nan')
                buy_position = 0

            # if there's existing buy_position and trading signal is sell
            elif stock_quantity > 0 and signal.loc[d] == -1:
                if pyramid == "Reflecting" and len(exit_pyramiding) != 0:
                    # Slippage occured and the sell price is no longer the desired price
                    slipped_price = self.slippage_price(signal.loc[d], self.daily_c.loc[d], exit_pyramiding[-1],
                                                        self.daily_v.loc[d])
                    # Realized profit/loss and returns based on average buy price
                    realized_pnl = exit_pyramiding[-1] * (slipped_price - avg_buy_price)
                    realized_ret = realized_pnl / (exit_pyramiding[-1] * avg_buy_price)
                    commission_cost = self.commission(exit_pyramiding[-1], (exit_pyramiding[-1] * slipped_price))
                    # Return the profit/loss to account after closing position, with commission fee deducted
                    account_value = account_value + (exit_pyramiding[-1] * slipped_price) - commission_cost
                    stock_quantity = stock_quantity - exit_pyramiding[-1]
                    portfolio_value = stock_quantity * self.daily_c.loc[d]
                    exit_pyramiding.pop()
                    if len(exit_pyramiding) > 0:
                        pyramiding_sell = True
                    elif len(exit_pyramiding) == 0:
                        pyramiding_sell = False
                else:
                    # Slippage occurred and the sell price is no longer the desired price
                    slipped_price = self.slippage_price(signal.loc[d], self.daily_c.loc[d], stock_quantity,
                                                        self.daily_v.loc[d])
                    realized_pnl = stock_quantity * (slipped_price - avg_buy_price)
                    realized_ret = realized_pnl / (stock_quantity * avg_buy_price)
                    commission_cost = self.commission(stock_quantity, (stock_quantity * slipped_price))
                    # Return the profit/loss to account after closing position, with commission fee deducted
                    account_value = account_value + (stock_quantity * slipped_price) - commission_cost
                    stock_quantity = 0
                    portfolio_value = 0.0
                buy_position = 0

            # With position and trading signal is buy
            elif stock_quantity > 0 and signal.loc[d] == 1 and pyramiding_sell == False:
                # if pyramiding is not done yet
                if buy_position < buy_length:
                    previous_stock_quantity = stock_quantity
                    current_stock_quantity = pyramiding[buy_position] // self.daily_c.loc[d]
                    exit_pyramiding.append(current_stock_quantity)
                    stock_quantity = stock_quantity + current_stock_quantity

                    # Slippage occurred and the sell price is no longer the desired price
                    slipped_price = self.slippage_price(signal.loc[d], self.daily_c.loc[d], current_stock_quantity,
                                                        self.daily_v.loc[d])
                    portfolio_value = stock_quantity * self.daily_c.loc[d]
                    realized_pnl = 0.0
                    realized_ret = float('nan')
                    avg_buy_price = ((avg_buy_price * previous_stock_quantity) + (
                            slipped_price * current_stock_quantity)) / stock_quantity
                    commission_cost = self.commission(current_stock_quantity, (current_stock_quantity * slipped_price))
                    account_value = account_value - (current_stock_quantity * slipped_price) - commission_cost
                    buy_position = buy_position + 1

                # if pyramiding is done
                else:
                    portfolio_value = stock_quantity * self.daily_c.loc[d]
                    realized_pnl = 0.0
                    realized_ret = float('nan')

            # With buy_position, hold and trading signal is do nothing
            elif stock_quantity > 0 and signal.loc[d] == 0:
                portfolio_value = stock_quantity * self.daily_c.loc[d]
                realized_pnl = 0.0
                realized_ret = float('nan')

            # With no buy_position, trading signal is buy
            elif stock_quantity == 0 and signal.loc[d] == 1 and d != self.daily_c.index[0]:
                stock_quantity = pyramiding[0] // self.daily_c.loc[d]
                exit_pyramiding.append(stock_quantity)
                # Slippage occured and the sell price is no longer the desired price
                slipped_price = self.slippage_price(signal.loc[d], self.daily_c.loc[d], stock_quantity,
                                                    self.daily_v.loc[d])
                portfolio_value = pyramiding[0]
                avg_buy_price = self.daily_c.loc[d]
                realized_pnl = 0.0
                realized_ret = float('nan')
                buy_position = 1
                commission_cost = self.commission(stock_quantity, slipped_price * stock_quantity)
                account_value = account_value - (slipped_price * stock_quantity) - commission_cost

            # With no position, trading signal is not buy, do nothing
            elif stock_quantity == 0 and signal.loc[d] != 1 and d != self.daily_c.index[0]:
                realized_pnl = 0.0
                realized_ret = float('nan')

            # Record it in the stock position value book
            stocks_values["Profit & Loss"].loc[d] = realized_pnl
            stocks_values["Trade Returns"].loc[d] = realized_ret
            stocks_values["Stock Quantity"].loc[d] = stock_quantity
            stocks_values["Portfolio Value"].loc[d] = portfolio_value
            stocks_values["Stock Price"].loc[d] = self.daily_c.loc[d]
            stocks_values["Account Value"].loc[d] = account_value
            account_equity = stocks_values["Portfolio Value"].loc[d] + stocks_values["Account Value"].loc[d]
            account_profit = stocks_values["Profit & Loss"].sum()
            # Revise pyramiding and trading fund with updated account equity and account profit
            # Only do this if there's any changes
            if account_profit_holder != account_profit and sizing == "Markets Money":
                # Only do this if it is not in the middle of pyramiding
                if buy_position == buy_length or buy_position == 0:
                    pyramiding = self.pyramiding_strategies(sizing, pyramid, account_equity, account_profit)
                    account_profit_holder = account_profit

            # Revise pyramiding and trading fund with updated account equity and account profit
            # Only do this if there's any changes
            if account_equity_holder != account_equity and sizing == "MultiTier":
                # Only do this if it is not in the middle of pyramiding
                if buy_position == buy_length or buy_position == 0:
                    pyramiding = self.pyramiding_strategies(sizing, pyramid, account_equity, account_profit)
                    account_equity_holder = account_equity
        total_value = stocks_values["Portfolio Value"] + stocks_values["Account Value"]
        # Calculate trading book
        portfolio_returns = MathCalc.construct_book(total_value)
        # Calculate trade KPI
        kpi = MathCalc.calc_kpi(portfolio_returns, stocks_values, self.symbol)
        return portfolio_returns, kpi, stocks_values

    def diversified_trade(self, stock_list):
        """
        This function create trading book for the diversifed portfolios
        """
        portfolio_values = pd.DataFrame(index=self.daily_c.index,
                                        columns=["Stock Price", "Stock Quantity", "Profit & Loss", "Trade Returns",
                                                 "Portfolio Value",
                                                 "Account Value"])

        # Calculate equally weighted fund allocation for each stock
        single_component_fund = (ACCOUNT_FUND * RISK_PCT) / 10
        share_distribution = single_component_fund / self.daily_c[stock_list].iloc[0]

        stocks_values = self.daily_c[stock_list].mul(share_distribution, axis=1)
        all_stocks_value = stocks_values.sum(axis=1)
        portfolio_values["Profit & Loss"] = all_stocks_value - all_stocks_value.shift(1)
        portfolio_values["Trade Returns"] = all_stocks_value / all_stocks_value.shift(1) - 1
        portfolio_values["Portfolio Value"] = all_stocks_value
        portfolio_values["Stock Price"] = self.daily_c
        # Account value left is the value deduct the initial amount used for buying stock on day 1
        portfolio_values["Account Value"] = ACCOUNT_FUND - stocks_values.iloc[0].sum()
        total_value = portfolio_values["Portfolio Value"] + portfolio_values["Account Value"]
        portfolio_returns = MathCalc.construct_book(total_value)
        kpi = MathCalc.calc_kpi(portfolio_returns, portfolio_values, "Non-correlate")

        return portfolio_returns, kpi, portfolio_values

    def index_buyhold_trade(self, symbol):
        """
        This function performs simple buy and hold DJIA index
        """
        portfolio_values = pd.DataFrame(index=self.daily_c.index,
                                        columns=["Stock Price", "Stock Quantity", "Profit & Loss", "Trade Returns",
                                                 "Portfolio Value",
                                                 "Account Value"])

        # Calculate equally weighted fund allocation for each stock
        trading_fund = (ACCOUNT_FUND * RISK_PCT)
        share_distribution = trading_fund / self.daily_c.iloc[0]
        index_value = self.daily_c * share_distribution
        portfolio_values["Profit & Loss"] = index_value - index_value.shift(1)
        portfolio_values["Trade Returns"] = index_value / index_value.shift(1) - 1
        portfolio_values["Portfolio Value"] = index_value
        portfolio_values["Stock Price"] = self.daily_c
        portfolio_values["Account Value"] = ACCOUNT_FUND * (1 - RISK_PCT)
        total_value = portfolio_values["Portfolio Value"] + portfolio_values["Account Value"]
        portfolio_returns = MathCalc.construct_book(total_value)
        kpi = MathCalc.calc_kpi(portfolio_returns, portfolio_values, symbol)

        return portfolio_returns, kpi, portfolio_values

    @staticmethod
    def strategies_trading(data_index, strategy_list, trading_instance, ema_21, ema_45):
        """
        This function performs trading for all 18 variations of EMA period, position sizing and pyramiding strategies.
        It performs on the training data with the purpose of finding the most robust strategy from ALL strategies' KPI
        across all DJIA component stocks.
        :param data_index: the date index for the price data
        :param strategy_list: the list of all 18 strategies
        :param trading_instance: the trading instance specific to a stock's training data
        :param ema_21: ema21 trades signal specific to a stock's training data
        :param ema_45: ema45 trades signal specific to a stock's training data
        :return: the mean (across all component stocks) KPI, returns and cumulative returns
        """
        kpi_mean = pd.DataFrame(index=strategy_list, columns=KPI)
        returns_mean = pd.DataFrame(index=data_index, columns=strategy_list)
        cumreturns_mean = pd.DataFrame(index=data_index, columns=strategy_list)
        start_time_p = time.time()
        print "Trading loops start time:", datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        for n in N:
            for s in POSITION_SIZING:
                for p in PYRAMIDING:
                    start_time = time.time()
                    print "Strategy Start time:", strftime("%a, %d %b %Y %H:%M:%S", localtime()), n, s, p
                    kpi_df = pd.DataFrame(index=DJI, columns=KPI)
                    returns_df = pd.DataFrame(index=data_index, columns=DJI)
                    cumreturns_df = pd.DataFrame(index=data_index, columns=DJI)
                    # Do this for every DJIA component stock
                    for stock in DJI:
                        returns, kpi, stocks_values = trading_instance.loc[stock].execute_trading(s, p, locals()[
                            "ema_" + str(n)][stock])
                        kpi_df.loc[stock] = kpi.loc[stock]
                        returns_df[stock] = returns['Returns']
                        cumreturns_df[stock] = returns['CumReturns']
                    kpi_mean.loc["{}-{}-{}".format(p, s, n)] = kpi_df.mean(axis=0)
                    returns_mean["{}-{}-{}".format(p, s, n)] = returns_df.mean(axis=1)
                    cumreturns_mean["{}-{}-{}".format(p, s, n)] = cumreturns_df.mean(axis=1)
                    elapsed_time = time.time() - start_time
                    print "Strategy Elapsed time: ", str(datetime.timedelta(seconds=elapsed_time))
        elapsed_time_p = time.time() - start_time_p
        print "Trading loops elapsed time: ", str(datetime.timedelta(seconds=elapsed_time_p))

        return kpi_mean, returns_mean, cumreturns_mean

    @staticmethod
    def robust_strategy_trading(data_index, robust_strategy, trading_instance, sizing_method, pyramid_strategy,
                                ema_signal, stock_list):
        """
        This function performs trading for the most robus strategies on the 10 chosen stocks on the test data set.
        :param data_index: The date time index on the test data
        :param robust_strategy: The identified most robust strategies
        :param trading_instance: The trading instance specific to the stocks on test data
        :param sizing_method: The chosen position sizing method
        :param pyramid_strategy: The chosen pyramiding strategy
        :param ema_signal: The chosen EMA signal
        :param stock_list: The 10 chosen most non-correlate stocks
        :return: the mean (across 10 chosen component stocks) KPI, returns and cumulative returns
        """
        kpi_mean = pd.DataFrame(index=[robust_strategy], columns=KPI)
        start_time_p = time.time()
        print "Program Start time:", datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        kpi_df = pd.DataFrame(index=[robust_strategy], columns=KPI)
        returns_df = pd.DataFrame(index=data_index, columns=stock_list)
        cumreturns_df = pd.DataFrame(index=data_index, columns=stock_list)
        for stock in stock_list:
            returns, kpi, stocks_values = trading_instance.loc[stock].execute_trading(sizing_method, pyramid_strategy,
                                                                                      ema_signal[stock])
            kpi_df.loc[stock] = kpi.loc[stock]
            returns_df[stock] = returns['Returns']
            cumreturns_df[stock] = returns['CumReturns']
        kpi_mean.loc[robust_strategy] = kpi_df.mean(axis=0)

        elapsed_time_p = time.time() - start_time_p
        print "Program Elapsed time: ", str(datetime.timedelta(seconds=elapsed_time_p))

        return kpi_mean, returns_df.mean(axis=1), cumreturns_df.mean(axis=1)

    @staticmethod
    def compile_ema_signals(data_index, trading_instance):
        """
        This function compiles EMA trading signals for all 30 components for all datasets (train, test and full)
        :param data_index:
        :param trading_instance: The instance that contains the daily price data for the specific stock on all datasets
        :return: the 2 (21 & 45) compiled EMA trade signals
        """
        for n in N:
            locals()["ema_" + str(n)] = pd.DataFrame(index=data_index, columns=DJI)
            for t in DJI:
                locals()["ema_" + str(n)][t] = trading_instance.loc[t].get_ema_signals(n)
        return locals()["ema_" + str(N[0])], locals()["ema_" + str(N[1])]


class UserInterfaceDisplay:
    """
    The class to display plot(s) to users
    """

    def plot_portfolio_return(self, cum_returns, strategy):
        """
        Function to plot all portfolio cumulative returns
        """
        # Set a palette so that all 14 lines can be better differentiated
        color_palette = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                         '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                         '#808000', '#ffd8b1', '#000075', '#808080']
        fig, ax = plt.subplots(figsize=(14, 6))

        # Iterate the compared list to get correlation coefficient array for every compared index
        # Plot the correlation line on the plot canvas
        for i, d in enumerate(cum_returns):
            ax.plot(cum_returns.index, cum_returns[d], '-', label=cum_returns.columns[i], linewidth=2,
                    color=color_palette[i])

        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Cumulative returns')
        if strategy == "TrainingSet" or strategy =='Test&FullSet':
            plt.title('Cumulative returns for portfolios with different trading strategies on {}'.format(strategy))
        else:
            plt.title('Cumulative returns for portfolios with {} trading strategy'.format(strategy))
        plt.subplots_adjust(hspace=0.5)

        # Display and save the graph
        plt.savefig('portfolios_returns_{}.png'.format(strategy))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved as portfolios_returns_{}.png. When done viewing, please close this plot for next plot. Thank You!".format(
                strategy))

        plt.show()

    def plot_portfolio_risk(self, returns, strategy):
        """
        This function plot the histograms of returns for all portfolios.
        """

        plt.close('all')
        # Define axes, number of rows and columns
        if strategy == "TrainingSet":
            f, ax = plt.subplots(9, 2, figsize=(16, 48))
        elif strategy == "TestSet":
            f, ax = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(hspace=0.5)

        for i, d in enumerate(returns):
            # Do odd/even check to col number for plot axes
            col, row = MathCalc.colrow(i)

            # plot line graph
            ax[row, col].hist(returns[d].dropna(), bins=30, color='darkgreen')
            ax[row, col].axvline(returns[d].mean(), color='red',
                                 linestyle='-.', linewidth=2.5, label='Mean')
            ax[row, col].set_title("Returns histogram for portfolio {}".format(returns.columns[i]), fontsize=12)
            ax[row, col].legend()
        plt.savefig('portfolios_risk_{}.png'.format(strategy))

        print(
            "Plot saved as portfolios_risk_{}.png. When done viewing, please close this plot to end program. Thank You!".format(
                strategy))

        plt.show()

    def plt_heatmap(self, kpi_mean_rated):
        """
        This function plots the heatmap of the training data's mean KPIs, to visualize how each of the 18 strategies
        performs on all KPIs.
        :param kpi_mean_rated: the mean (all 30 component stocks) KPId
        """
        kpi_mean_rated = kpi_mean_rated[kpi_mean_rated.columns].astype(float)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(kpi_mean_rated, annot=True, annot_kws={"size": 10}, cmap="RdYlGn")
        ax.set_title('Normalized KPI heatmap for all 18 trading strategies ', fontsize=14)
        plt.savefig('strategies_heatmap.png')
        print(
            "Plot saved as strategies_heatmap.png. When done viewing, please close this plot for next plot. Thank You!")
        plt.show()

# ----------------------------- MAIN PROGRAM ---------------------------------

def main():
    """
    The main program

    """
    print ("\n")
    print ("############################## Seeking Alpha with EMA-ATR trade signals #################################")
    print ("\n")
    print "**********************************  Data download and preprocessing ***************************************"
    # Set the print canvas right
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1600)

    print "Starting to download data"
    dataset = DataRetrieval()

    print "Starting to split data into train and test sets ..."

    # Deciding the portion to split the data into train and test set
    train_c, test_c, train_h, test_h, train_l, test_l, train_v, test_v = dataset.get_split_dataset()
    full_c, full_h, full_l, full_v = dataset.get_full_dataset()
    dji_daily = dataset.get_DJI()

    # Preparing trading instances by assembling all instances to a dataframe arranged by train, test, full dataset
    # for each of the 30 component stock
    # The reason to pre-compiled this is to avoid performing the same action for 18 times (18 strategies)
    dataset_series = ["train", "test", "full"]
    trading_instance = pd.DataFrame(index=train_c.columns, columns=dataset_series)
    for d in dataset_series:
        for stock in locals()[str(d) + "_c"]:
            trading_instance[d].loc[stock] = Trading(locals()[str(d) + "_c"][stock], locals()[str(d) + "_h"][stock],
            locals()[str(d) + "_l"][stock], locals()[str(d) + "_v"][stock], stock)
    print ("\n")
    print "Data download done, calculating EMA Signals ..."
    print ("\n")
    # Get the training set ema signals
    # Again, calculating the EMA signals before hand to avoid performing same calculation (30 stocks) for 18 times
    train_ema_21, train_ema_45 = Trading.compile_ema_signals(train_c.index, trading_instance["train"])
    test_ema_21, test_ema_45 = Trading.compile_ema_signals(test_c.index, trading_instance["test"])
    full_ema_21, full_ema_45 = Trading.compile_ema_signals(full_c.index, trading_instance["full"])

    print "Compiling strategy list ..."
    print ("\n")
    strategy_list = Trading.compile_strategy_list()
    print "Strategy list: "
    for s in strategy_list:
        print s

    print ("\n")
    print "*****************************  EMA-ATR Strategy Training Set Trading  *********************************"

    print "Training set for 18 strategies trading starts now. This may take a while, grab a coffee and some snacks"

    # To avoid the long training data trading session, commented out line 983 - 989, uncomment line 994 - 998 to
    # the pre-saved KPI file directed.

    train_kpi_mean, train_returns_mean, train_cumreturns_mean = Trading.strategies_trading(train_c.index, strategy_list,
            trading_instance["train"], train_ema_21,train_ema_45)

    # Save the trained KPI, returns & cumulative returns to csv file
    train_kpi_mean.to_csv("trading_kpi.csv", encoding='utf-8')
    train_returns_mean.to_csv("trading_returns_mean.csv", encoding='utf-8')
    train_cumreturns_mean.to_csv("trading_cumreturns_mean.csv", encoding='utf-8')

    """

    train_kpi_mean = pd.read_csv('trading_kpi.csv')
    train_kpi_mean.index = train_kpi_mean['Unnamed: 0']
    train_kpi_mean.drop('Unnamed: 0', axis=1, inplace=True)
    train_returns_mean = pd.read_csv('trading_returns_mean.csv', index_col='Date', parse_dates=True)
    train_cumreturns_mean = pd.read_csv('trading_cumreturns_mean.csv', index_col='Date', parse_dates=True)
    """
    print ("\n")
    print "Done, Training set's mean KPIs across all for 18 strategies  and 30 component stocks: "
    print train_kpi_mean

    kpi_mean_rated, robust_strategy, pyramid_strategy, sizing_method, ema_period = Trading.find_robust_strategy(
        strategy_list, train_kpi_mean)
    print ("\n")
    print "Normalized mean KPIs: "
    print kpi_mean_rated
    print ("\n")
    UserInterfaceDisplay().plt_heatmap(kpi_mean_rated)
    print ("\n")
    print "The most robust strategy is found to be: "
    print robust_strategy

    print ("\n")
    print "********************************  EMA-ATR Strategy Test Set Trading  ***********************************"
    print ("\n")
    print "Trading on the test set on {} strategy across all 30 DJIA component stocks ... ".format(robust_strategy)
    test_kpi_mean, test_returns_mean, test_cumreturns_mean = Trading.robust_strategy_trading(test_c.index,
        robust_strategy, trading_instance["test"], sizing_method, pyramid_strategy, locals()["test_ema_" + str(
        ema_period)], DJI)

    print ("\n")
    print "The mean KPIs for test set on {} strategy across all 30 DJIA component stocks:".format(robust_strategy)
    print test_kpi_mean
    print ("\n")
    components_corr = Trading.stocks_corr(test_returns_mean, test_c)
    non_correlate_stocks = components_corr[0:10].index
    print "The 10 most non-correlated stocks with DJIA are: ",
    print ("\n")
    for c in non_correlate_stocks:
        print "{}({}): {}".format(DJI_N[DJI.index(c)], c, components_corr[c])

    print ("\n")
    print "Trading on the full set on {} strategy across 10 most non-correlated stocks ... ".format(robust_strategy)
    full_kpi_mean, full_returns_mean, full_cumreturns_mean = Trading.robust_strategy_trading(full_c.index,
        robust_strategy, trading_instance["full"], sizing_method, pyramid_strategy, locals()["full_ema_" + str(
        ema_period)],non_correlate_stocks)

    print ("\n")
    print "The mean KPIs for full set on {} strategy across 10 most non-correlated stocks: ".format(robust_strategy)
    print full_kpi_mean
    print ("\n")
    print "Trading on the full set on simple buy and hold strategy across 10 most non-correlated stocks ... "
    stock_trading_non_correlate = Trading(full_c, full_h, full_l, full_v, "Non-correlate")
    longonly_returns, longonly_kpi, stocks_values_longonly = stock_trading_non_correlate.diversified_trade(
        non_correlate_stocks)

    print ("\n")
    print "The mean KPIs for the full set on simple buy and hold strategy across 10 most non-correlated stocks:"
    print longonly_kpi
    print ("\n")
    print "Trading on DJIA index on simple buy and hold strategy ... "
    dji_longonly = Trading(dji_daily['Adj Close'], dji_daily['High'], dji_daily['Low'], dji_daily['Volume'],
                           "Simple buy & hold")
    dji_longonly_returns, dji_longonly_kpi, dji_values_longonly = dji_longonly.index_buyhold_trade(['^DJI'])

    print ("\n")
    print "The mean KPIs for DJIA index on simple buy and hold strategy: "
    print dji_longonly_kpi
    print ("\n")

    # Assemble dataframes for test result returns and cumulative returns
    trade_strategies = ["Robust strategy on testset", "Robust strategy on fullset NC", "Simple BuyHold on fullset NC",
                        "Simple BuyHold on DJIA"]
    cum_returns_df = pd.DataFrame(index=full_c.index, columns=trade_strategies)
    cum_returns_df["Robust strategy on testset"] = test_cumreturns_mean
    cum_returns_df["Robust strategy on fullset NC"] = full_cumreturns_mean
    cum_returns_df["Simple BuyHold on fullset NC"] = longonly_returns['CumReturns']
    cum_returns_df["Simple BuyHold on DJIA"] = dji_longonly_returns['CumReturns']

    returns_df = pd.DataFrame(index=full_c.index, columns=trade_strategies)
    returns_df["Robust strategy on testset"] = test_returns_mean
    returns_df["Robust strategy on fullset NC"] = full_returns_mean
    returns_df["Simple BuyHold on fullset NC"] = longonly_returns['Returns']
    returns_df["Simple BuyHold on DJIA"] = dji_longonly_returns['Returns']
    returns_df = returns_df[returns_df.columns].astype(float)
    train_returns_mean = train_returns_mean[train_returns_mean.columns].astype(float)

    # Visualized results
    UserInterfaceDisplay().plot_portfolio_return(train_cumreturns_mean, "TrainingSet")
    UserInterfaceDisplay().plot_portfolio_return(cum_returns_df, "Test&FullSet")
    UserInterfaceDisplay().plot_portfolio_return(
        cum_returns_df[["Robust strategy on testset", "Robust strategy on fullset NC"]], "RobustStrategy")
    UserInterfaceDisplay().plot_portfolio_return(
        cum_returns_df[["Simple BuyHold on fullset NC", "Simple BuyHold on DJIA"]], "SimpleBuyHold")
    UserInterfaceDisplay().plot_portfolio_risk(train_returns_mean, "TrainingSet")
    UserInterfaceDisplay().plot_portfolio_risk(returns_df, "TestSet")

    print ("#########################################   END OF PROGRAM   ##############################################")

if __name__ == '__main__':
    main()

    # -------------------------------- END  ---------------------------------------


