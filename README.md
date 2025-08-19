# investment_returns_simulator
simulates ROI of equities


What are optimal ways of investing in equities  compared to stock picking?
how much sensitivity is there in dollar cost averaging (DCA) compared to active trading?
Is it that hard to beat an index like VTSAX or VOO? Can you beat the market? let's find out!

This project ventures to simulate any proposed equity portfolio composition, across any range of time (data allowing), accounting for capital investment dates and comparing it to other strategies. 

all possible returns
X = Start date to date, y = start date to date, z = total portfolio return from start date (x) to end date (y). 

Histogram of total portfolio returns. split by negative returns and positive returns. histogram of start dates and neg return count. barchart of start dates and returns. bar chart of end dates and returns. is there a pattern related to when to start and stop? brief investment horizon? is it more about particular days? list all common time patterns. 

Statistics:
Probability of losing money, breaking even, making money.  
Relationship between duration of investment and ROI.  

optimal Timeline:
a given start-end date range for a particular portfolio return vs index.

general Timeline:
show all equities over time Including indicies. 

timeline statistics:
show % of companies more successful than market, and those less successful than market.

progress notes:

test commit ✅
do whole repo update cycle (connect, clone, edit, save, commit & push, disconnect)

6/5/2025: added equities universe.csv ✅

6/6/2025: access and connect equities universe.csv and save a var. ✅ -- need to research if can store db in google console.

6/7/2025: create google big query studio account, uploaded symbols_universe ✅

6/8/2025: update/ get from yfinance save to aapl csv ✅ connect gbq to vs code. ❌ abandoning this until I get help. 

6/9/2025: clean yfinance output and save to csv [Date, Close] ✅ see how google sheets did it.

6/10/2025: save all historical data of one symbol to csv: check for min date, max date, ✅ and compare to min/max if query to yfinance. 

6/11/2025: if symbol file, get start end date.  if no symbol file, create and add start end date to symbols CSV. get yfinance data, and save to symbol csv. log start end date to all_symbols file for next time. ✅

6/12/2025: fill out data based on new dates from user, without repulling existing data. and test on yfinance. ✅

6/13/2025: save yfinance_data to csv based on if it's case 0 - 4, or case 5.  ✅

6/14/2025: update readme with greater clarity on what the analysis on the time vars will actually be. ✅

6/15/2025: research backend dbs: gcp/cloud storage. see Straico. Connected google big query to vs code. ✅

6/16/2025: test all cases again. ✅

6/17/2025: update readme. ✅

6/18/2025: resolve bug: end date is off by 1 sometimes. roll up the 2 functions into 1 function, because get_existing_dates is only being called once.  testing providing multiple symbols. start getting data by symbol. ✅   

6/19/2025: apply SQLite. get python files talking to each other. ✅

6/20/2025: return stock price change via SQL WHILE appling medallion data architecture to format code. ✅

6/21/2025: update readme. ✅ 

6/22/2025: return portfolio (symbols, start, end) ROI, summing up portfolio returns. ✅

7/06/2025: make symbol and porfolio ROI table (start x end) and plot surface. ✅

7/07/2025: update/ can compare portfolios, outputs binary predictory stats (accu… ✅

7/15/2025: feature/ symbol_data_to_returns how subtracts two dfs if provided. st… ✅

7/16/2025: feature/ added timeline() to plot close. stats() outputs quantity and… ✅

7/17/2025: feature/ added various value options for symbol_data_to_returns_df(),… ✅

7/19/2025: update/ deleted old symbols_data files. ✅

7/21/2025: update/ created symbols_data folder to clean up the space. ✅

7/26/2025: working on implementing .pipe() ✅

7/29/2025: symbol_data_to_returns_df/vecterized df/test results. ✅  define main.py workflow. symbols_stats: faulty values if run after downloading new data, but if you rerun, it's fine.  ⚠

7/30/2035: update symbols_stats to handle results_df and returns stats that make sense. Can't without histogram first. ✅ target stats: max days between gains, max days within gains.  set results_df to Close for testing.

8/2/2025: target stats: max days between gains, max days within gains.✅

8/17/2025: add save feature to symbols_and_results_stats(). ✅ bug: symbols_data isn't updating under old code, but it does under new code.  and so end_date is stuck at 7/10/2025, which throws off the training window ⚠️ 

8/18/2025: Switched to class version of get_data() by chatgpt. ✅ bug: stats(): start_date and end_date are the same per symbol, but should be dependent on symbol data. Create get() for reading saved stats_df.  create histogram function(symbols_df, returns_df, stats_df), create timeline of relative gains
