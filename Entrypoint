#!/usr/bin/env sh
# Dispatches to the correct script based on the ECS task definition's
# `command` field. First arg selects the script; everything after is
# forwarded as-is (e.g. record_outcomes needs --retry-pending).
#
# Example ECS container `command` overrides:
#   ["auto_log_predictions"]
#   ["record_outcomes", "--retry-pending"]
#   ["prediction_logger"]   (manual/one-off use — it's primarily a library)

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
  *)
    echo "Unknown script '$SCRIPT'. Expected one of: auto_log_predictions, record_outcomes, prediction_logger" >&2
    exit 64
    ;;
esac
