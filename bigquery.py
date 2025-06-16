from google.cloud import bigquery
from google.oauth2 import service_account
import os

# Set the environment variable for the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "H:\\My Drive\\Projects\\The Money Tree github investment returns simulator\\investment-returns-simulator-8b0934705438.json"

# Load the service account credentials
credentials = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# Initialize the BigQuery client
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Query the BigQuery dataset
query = """
SELECT *
FROM `investment-returns-simulator.Equities_Universe_Symbols.Symbols`

LIMIT 100;
"""

# Execute the query
query_job = client.query(query)

# Convert the result to a pandas DataFrame
df = query_job.to_dataframe()

# Print the first 10 rows
print(df.head(10))