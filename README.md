# Stock Market Exploratory Data Analysis
An educational project for me to learn some basics on Data Visualization and Analysis.

o The Stock class fetches data regarding stock from now to 1 year ago using pandas Datareader

o The class has 3 basic hierarchical functions: calculate, analyse and compare

o Daily Change is analysed using a normal distribution plot and a histogram

o Risk is analysed using a scatter plot(it plots only one exact point on the graph of Estimated Return(x, the mean of Daily Return) and Risk(y, the standard deviation))

o There are also 2 correlation analysis functions, one between 2 stocks(seaborn jointplot), and one between a list of stocks(seaborn PairGrid)

o It uses the Monte Carlo Stimulation in order to visualize VaR and Risk for a year timespan
