# MLB Market Engine — shared image for the three scheduled jobs
# (auto_log_predictions.py, record_outcomes.py, prediction_logger.py)
# plus, as of Weekend 6, the standing Streamlit web service (app.py).
#
# One image, dispatched to the right entrypoint via entrypoint.sh + the
# ECS task/service definition's `command` override. Data files (parquet)
# are NOT baked in — bootstrap_data.py pulls them from S3 at container
# start for the batch jobs. app.py doesn't need them at all (read-only
# DB viewer), so the app service's cold start is fast.

FROM python:3.11-slim

# ca-certificates: needed for HTTPS calls to S3, Secrets Manager, and
# the MLB Stats API. Nothing else — psycopg2-binary avoids the need
# for libpq-dev / build-essential.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Only meaningful for the Weekend 6 app service (streamlit_app command) —
# the batch-job commands never bind a port, so this is a no-op for them.
EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
