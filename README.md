# StockMarket_Data-Analysis
Conducting stock market analysis


o The Stock class fetches data regarding stock from now to 1 year ago using pandas Datareader

o The class has 3 basic hierarchical functions: calculate, analyse and compare

o Daily Change is analysed using a normal distribution plot and a histogram

o Risk is analysed using a scatter plot(it plots only one exact point on the graph of Estimated Return(x, the mean of Daily Return) and Risk(y, the standard deviation))

o All of these functions are recalled by certain objects when one decides to compare itself with other Stock objects

o There are also 2 correlation analysis functions, one between 2 stocks(seaborn jointplot), and one between a list of stocks(seaborn PairGrid)

o It uses the Monte Carlo Stimulation in order to visualize VaR and Risk for a year timespan
