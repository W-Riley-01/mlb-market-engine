import json
import boto3
import psycopg2

SECRET_ARN = "arn:aws:secretsmanager:us-east-1:687050094462:secret:rds!db-e3e1711b-71a6-4339-9ebb-79ace00465a4-hT3Uzj"

client = boto3.client("secretsmanager", region_name="us-east-1")
secret = client.get_secret_value(SecretId=SECRET_ARN)
creds = json.loads(secret["SecretString"])

print(f"Username from secret: {creds['username']!r}")
print(f"Password length from secret: {len(creds['password'])}")
print(f"Password repr from secret: {creds['password']!r}")

conn = psycopg2.connect(
    host="localhost",
    port=5434,
    dbname="mlb_engine",
    user=creds["username"],
    password=creds["password"],
)
print("CONNECTED OK")
conn.close()
