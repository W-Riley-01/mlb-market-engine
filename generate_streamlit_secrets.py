"""
generate_streamlit_secrets.py
------------------------------
Writes .streamlit/secrets.toml at container startup so that
prediction_logger._get_conn()'s st.connection("predictions_db", type="sql")
call has something to read — Streamlit's own connection API only reads
credentials from that file (or st.secrets), never from plain env vars.

Only invoked on the streamlit_app entrypoint path — never runs for the
batch jobs, which use their own (separate) connection logic.

Credentials are fetched live from Secrets Manager on every container
start and combined with the RDS_ENDPOINT / RDS_PORT / RDS_DB_NAME env vars
the task definition already injects (ecs.tf's local.common_environment) —
matching the documented pattern that the RDS-managed secret only ever
contains username/password, never host/port/dbname. Nothing here is
baked into the image or written anywhere but this container's own
ephemeral filesystem, which disappears when the task stops.
"""
from __future__ import annotations

import json
import os

import boto3

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
RDS_SECRET_ARN = os.environ["RDS_SECRET_ARN"]
RDS_ENDPOINT = os.environ["RDS_ENDPOINT"]
RDS_PORT = os.environ.get("RDS_PORT", "5432")
RDS_DB_NAME = os.environ["RDS_DB_NAME"]

SECRETS_PATH = ".streamlit/secrets.toml"


def main() -> None:
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = client.get_secret_value(SecretId=RDS_SECRET_ARN)
    creds = json.loads(secret["SecretString"])  # only username + password

    url = (
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{RDS_ENDPOINT}:{RDS_PORT}/{RDS_DB_NAME}"
    )

    os.makedirs(os.path.dirname(SECRETS_PATH), exist_ok=True)
    with open(SECRETS_PATH, "w") as f:
        f.write("[connections.predictions_db]\n")
        f.write(f'url = "{url}"\n')


if __name__ == "__main__":
    main()