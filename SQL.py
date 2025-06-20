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
    # 2) extract first & last
    start_date, start_price = raw_rows[0]
    end_date,   end_price   = raw_rows[-1]
    return start_date, start_price, end_date, end_price

def aggregate_gold(start_date, start_price, end_date, end_price):
    # 3) compute percent change
    pct = (end_price - start_price) / start_price
    return pct


# bronze layer queries the database
raw_rows = query_bronze(symbols[0])

# silver layer turns many rows into 4 key values
start_date, start_price, end_date, end_price = transform_silver(raw_rows)

# gold layer computes the final metric
percent_change = aggregate_gold(start_date, start_price,
                                end_date, end_price)

print(f"{symbols[0]}: {percent_change:.2%} "
      f"(from {start_date}→{end_date})")


# # Clean up: Remove the database file after use
# db_path.unlink()

#check db is deleted
# if not db_path.exists():
#     print(f"Database file {db_file_path} has been deleted successfully.")
# else:
#     print(f"Failed to delete database file {db_file_path}.")