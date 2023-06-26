from pandas import Series, DataFrame
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import DataReader
from datetime import datetime

sns.set_style("whitegrid")

# Global Variables
# tech_list = ['AAPL', 'GOOG', "MSFT", "AMZN"]
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)


def fetchStockData(stock_tick):
    df = DataReader(stock_tick, 'yahoo', start, end)
    return df


class Stock:
    # class Attributes
    def __init__(self, stock_tick):
        self.stock_tick = stock_tick
        self.stock_df = fetchStockData(stock_tick)
        self.calculateTotalTraded()
        self.calculateDailyReturn()
        self.calculateMovingAverage()
        self.VaR = self.stock_df["Daily Return"].quantile(0.05)

    def stockVolumeAnalysis(self):
        self.stock_df["Volume"].plot(label=self.stock_tick, title="Volume")

    def compareStockVolume(self, stock_list):
        self.stockVolumeAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockVolumeAnalysis()
        plt.legend()

    def stockAdjCloseAnalysis(self):
        self.stock_df["Adj Close"].plot(label=self.stock_tick, title="Adjusted Closing Price")

    def compareStockAdjClose(self, stock_list):
        self.stockAdjCloseAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockAdjCloseAnalysis()
        plt.legend()

    def calculateTotalTraded(self):
        self.stock_df["Total Traded"] = self.stock_df["Open"] * self.stock_df["Volume"]

    def stockTotalTradedAnalysis(self):
        self.stock_df["Total Traded"].plot(label=self.stock_tick, title="Total Traded Stocks")
        plt.legend()

    def compareTotalTraded(self, stock_list):
        self.stockTotalTradedAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockTotalTradedAnalysis()
        plt.legend()

    def calculateMovingAverage(self):
        ma_days = [10, 20, 50]
        for ma in ma_days:
            column_name = "MA for %s days" % (str(ma))
            self.stock_df[column_name] = self.stock_df["Adj Close"].rolling(ma).mean()

    def stockMovingAverageAnalysis(self):
        titlestr = "Moving Averages for {} Stocks".format(self.stock_tick)
        self.stock_df[["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]].plot(title=titlestr,
                                                                                                subplots=False)
        plt.legend()

    def compareMovingAverages(self, stock_list):
        self.stockMovingAverageAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockMovingAverageAnalysis()
        plt.legend()

    def calculateDailyReturn(self):
        self.stock_df["Daily Return"] = self.stock_df["Adj Close"].pct_change()

    def stockDailyReturnAnalysis(self):
        self.stock_df["Daily Return"].plot(label=self.stock_tick, title="Daily Return")
        plt.legend()

    def compareDailyReturns(self, stock_list):
        self.stockDailyReturnAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockDailyReturnAnalysis()
        plt.legend()

    def stockRiskAnalysis(self):
        returns = self.stock_df["Daily Return"].dropna()
        plt.scatter(returns.mean(), returns.std(), s=62.8, label=self.stock_tick)
        plt.title("Risk Analysis")
        plt.xlabel('Expected returns')
        plt.ylabel('Risk')

    def compareStockRisk(self, stock_list):
        self.stockRiskAnalysis()
        for stock_obj in stock_list:
            stock_obj.stockRiskAnalysis()
        plt.legend()

    def stockCorrelationAnalysis(self, stock):
        ax = sns.jointplot(self.stock_df["Daily Return"], stock.stock_df["Daily Return"], kind='reg').annotate(stats.pearsonr)
        ax.set_axis_labels(xlabel=self.stock_tick, ylabel=stock.stock_tick)
        plt.title("Stock Daily Return Correlation")

    def compareStockCorrelation(self, stock_list):
        DailyReturns = DataFrame()
        DailyReturns[self.stock_tick] = self.stock_df["Daily Return"]
        for stock_obj in stock_list:
            DailyReturns[stock_obj.stock_tick] = stock_obj.stock_df["Daily Return"]

        ax = sns.PairGrid(DailyReturns.dropna())
        ax.map_diag(sns.distplot, bins=30)
        ax.map_offdiag(sns.regplot)
        plt.title("Stock Daily Return Correlation")

    def monteCarloResultAnalysis(self):
        for x in range(1, 100):
            plt.plot(self.monteCarloStimulation(self.stock_df["Adj Close"].max(), 365))
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.title("Monte Carlo Simulation Analysis for {} Stocks".format(self.stock_tick))

    def monteCarloStimulation(self, start_price, days):
        # delta
        dt =1/days
        mean = self.stock_df["Daily Return"].mean()
        sigma = self.stock_df["Daily Return"].std()

        price = np.zeros(days)
        price[0] = start_price

        shock = np.zeros(days)
        drift = np.zeros(days)

        for x in range(1, days):
            shock[x] = np.random.normal(loc=mean*dt, scale=sigma*np.sqrt(dt))
            drift[x] = mean * dt
            price[x] = price[x-1] + (price[x-1]*(drift[x]+ shock[x]))
        return price


def main():
    s1 = Stock("MSFT")
    s2 = Stock("GOOG")
    s3 = Stock("AAPL")
    s4 = Stock("AMZN")
    # compareIndividualMovingAverages(df1)
    #s1.compareMovingAverages([s2, s3, s4])
    # s1.compareStockCorrelation([s2, s3, s4])
    s2.monteCarloResultAnalysis()
    plt.show()


if __name__ == "__main__":
    main()
