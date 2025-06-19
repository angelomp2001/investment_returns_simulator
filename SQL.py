import sqlite3
import pandas as pd
from functions import all_symbols_path, symbols
from pathlib import Path

# Step 1: Create an SQLite database file
db_file_path = 'equity_returns_database.db' #'C:\Users\Angelo\Documents\vscode\investment_returns_simulator\investment_returns_simulator\equity_returns_database.db'  

# Create a connection to the SQLite database
conn = sqlite3.connect(db_file_path)

# Step 2: Load the CSV file into the SQLite database
csv_file_path = Path(symbols[0] + '.csv')  # Replace with your CSV file path

# Read the CSV file using pandas
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} does not exist.")
    exit(1)

# Write the DataFrame to the SQLite database
df.to_sql('my_table', conn, if_exists='replace', index=False)

# Step 3: Query the data
cursor = conn.cursor()

# Example query: Select all records from the table
cursor.execute('SELECT * FROM my_table')
rows = cursor.fetchall()

for row in rows:
    print(row)

# Close the connection
conn.close()