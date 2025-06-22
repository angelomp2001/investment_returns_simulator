import sqlite3
import pandas as pd
from functions import symbols, get_yfinance_data, start_date, end_date
from pathlib import Path

# This query selects all columns from the first symbol in the symbols list.
# query = f"""
# SELECT *
# FROM {symbols[0]};
# """

# This query calculates the percentage change between the first and last non-null prices in a table.
query = f"""
SELECT
  t1.start_date,
  t1.start_price,
  t2.end_date,
  t2.end_price,
  (t2.end_price - t1.start_price) / t1.start_price
    AS percent_change
FROM
  (
    SELECT
      date  AS start_date,
      close AS start_price
    FROM {symbols[0]}
    WHERE close IS NOT NULL
    ORDER BY date ASC
    LIMIT 1
  ) AS t1
CROSS JOIN
  (
    SELECT
      date  AS end_date,
      close AS end_price
    FROM {symbols[0]}
    WHERE close IS NOT NULL
    ORDER BY date DESC
    LIMIT 1
  ) AS t2;
"""

# WHERE type = 'table', WHERE sqlite_master """

def query_bronze(symbol):
    # Step 1: Create an SQLite database file
    db_file_path = 'equity_returns_database.db' #'C:\Users\Angelo\Documents\vscode\investment_returns_simulator\investment_returns_simulator\equity_returns_database.db'  
    db_path = Path(db_file_path)

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)

    # Step 2: Load the CSV file into the SQLite database
    csv_file_path = Path(symbol + '.csv') 

    # Step 3: Query the data, aka cursor
    cursor = conn.cursor()

    # SQL query to select all data from the specified symbol
    sql = f"""
    SELECT
    date,
    close
    FROM {symbol}
    WHERE close IS NOT NULL
    ORDER BY date ASC
    """

    try:
        # Calculate return on symbol
        print(f"1. Querying database for {symbol} data...")
        cursor.execute(sql) #SELECT * FROM my_table')
        return cursor.fetchall()    # -> List of (date, close)

    except sqlite3.Error as e:
        print(f"Query failed: {e}")
        
        # check for CSV file using pandas
        try:
            print(f"2. Querying CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            # Write df to SQLite database
            df.to_sql(symbol, conn, if_exists='replace', index=False)
            cursor.execute(sql)
            return cursor.fetchall()    # -> List of (date, close)
        except FileNotFoundError:
            print(f"Query failed: The file {csv_file_path} does not exist.")
            
            try:
                print(f"3. Fetching data for {symbol} from yfinance...")
                get_yfinance_data(symbol, start_date, end_date)
                # Write df to SQLite database
                df = pd.read_csv(csv_file_path)
                df.to_sql(symbol, conn, if_exists='replace', index=False)
                cursor.execute(sql)
                return cursor.fetchall()    # -> List of (date, close)
            except Exception as e:
                print(f"An error occurred while fetching data: {e}")

    # Close the connection
    conn.close()


def transform_silver(raw_rows):
    # this function transforms the raw data into 4 key values
    # 2) extract first & last
    start_date, start_price = raw_rows[0]
    end_date,   end_price   = raw_rows[-1]
    return start_date, start_price, end_date, end_price

def aggregate_gold(start_date, start_price, end_date, end_price):
    # this function computes the aggregate values from inputs
    # format vars
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # 3) compute aggregate values
    gain_loss = (end_price - start_price) # gain or loss in investment
    avg_pct_change = (end_price/start_price) ** (1/(end_date-start_date).days) - 1  # Geometric mean of daily returns
    return gain_loss, avg_pct_change

def gain_loss(symbols, start_date, end_date):
    #initialize an empty DataFrame and new row data to store the results
    df = pd.DataFrame(columns=['symbol', 'start_date_close', 'end_date_close', 'aggregate'])
    new_rows = []
    # Loop through each symbol
    for symbol in symbols:
        # bronze layer queries the database
        print(f"Processing {symbol}...")
        raw_rows = query_bronze(symbol)

        # silver layer turns many rows into 4 key values
        print(f"Transforming {symbol} data...")
        start_date, start_price, end_date, end_price = transform_silver(raw_rows)

        # gold layer computes the final metric
        print(f"Aggregating {symbol} data...")
        aggregate, _ = aggregate_gold(start_date, start_price,
                                        end_date, end_price)
        
        # Append aggregate to df
        print(f"Appending {symbol} data to DataFrame...")
        new_rows.append({
            'symbol': symbol,
            'start_date_close': start_price,
            'end_date_close': end_price,
            'aggregate': aggregate
        })
    
    # Append new rows to the DataFrame
    print(f"Appending new rows to DataFrame...")
    df_new_rows = pd.DataFrame(new_rows)
    df = pd.concat([df, df_new_rows], axis=0)
    # calculate gain/loss
    print(f"Calculating gain/loss for portfolio...")
    gain_loss = df['aggregate'].sum()
    return df, gain_loss

df, gain_loss = gain_loss(symbols, start_date, end_date)

print("Portfolio: \n", df.head())
print(f"gain/loss: {gain_loss} " # {aggregate:.2%} " # for percentage
    f"(from {start_date} → {end_date})")


# # Clean up: Remove the database file after use
# db_path.unlink()

#check db is deleted
# if not db_path.exists():
#     print(f"Database file {db_file_path} has been deleted successfully.")
# else:
#     print(f"Failed to delete database file {db_file_path}.")