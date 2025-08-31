from get_data_chatgpt import get_symbol_data
df = get_symbol_data(symbols='VOOG', start_date='8/1/2025', end_date='8/20/2025')
print(df.head())
print(type(df))