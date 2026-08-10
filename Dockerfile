# MLB Market Engine — shared image for the three scheduled jobs
# (auto_log_predictions.py, record_outcomes.py, prediction_logger.py)
#
# One image, dispatched to the right script via entrypoint.sh + the
# ECS task definition's `command` override. Data files (parquet) are
# NOT baked in — bootstrap_data.py pulls them from S3 at container
# start, same as it does today in GitHub Actions. This keeps the image
# small and means updated parquet files propagate without a rebuild.

FROM python:3.11-slim

# ca-certificates: needed for HTTPS calls to S3, Secrets Manager, and
# the MLB Stats API. Nothing else — psycopg2-binary avoids the need
# for libpq-dev / build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3
    # ^ boto3 installed explicitly here as a stopgap since it's not yet
    # in requirements.txt (see note in chat). Once boto3 is added to
    # requirements.txt directly, this second pip install line can be
    # removed — but leaving both doesn't hurt in the meantime.

# Copy the full repo context — these scripts use flat relative imports
# (from bootstrap_data import ..., from resolver import ..., from
# engine_runner import ...) rather than a packaged module layout, so
# everything needs to land in the same working directory.
COPY . .

# bootstrap_data.py writes here at runtime; create it so the app user
# doesn't need write permission on /app itself.
RUN mkdir -p /app/data

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as a non-root user — good practice generally, and specifically
# relevant here since this container only ever needs to read its own
# code/data and talk outbound to AWS APIs, never needs root.
RUN useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
