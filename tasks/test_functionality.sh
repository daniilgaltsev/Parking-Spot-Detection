#!/bin/bash

set -uo pipefail
set +e

FAILURE=false
pytest -s parking_spot_detection || FAILURE=true

if [ "$FAILURE" = true ]; then
    echo "Testing failed"
    exit 1
fi

echo "Testing passed"
exit 0
