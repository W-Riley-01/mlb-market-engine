#!/usr/bin/env sh
# Dispatches to the correct script based on the ECS task definition's
# `command` field. First arg selects the script; everything after is
# forwarded as-is (e.g. record_outcomes needs --retry-pending).
#
# Example ECS container `command` overrides:
#   ["auto_log_predictions"]
#   ["record_outcomes", "--retry-pending"]
#   ["prediction_logger"]   (manual/one-off use)
#   ["streamlit_app"]       (Weekend 6 — standing web service, not a batch job)

set -e

SCRIPT="$1"
shift || true

case "$SCRIPT" in
  auto_log_predictions)
    exec python auto_log_predictions.py "$@"
    ;;
  record_outcomes)
    exec python record_outcomes.py "$@"
    ;;
  prediction_logger)
    exec python prediction_logger.py "$@"
    ;;
  streamlit_app)
    python generate_streamlit_secrets.py
    exec streamlit run app.py \
      --server.port=8501 \
      --server.address=0.0.0.0 \
      --server.headless=true \
      "$@"
    ;;
  *)
    echo "Unknown script '$SCRIPT'. Expected one of: auto_log_predictions, record_outcomes, prediction_logger, streamlit_app" >&2
    exit 64
    ;;
esac