#!/bin/bash
set -euo pipefail
: "${1:?need command}" 
export KCORO_TRACE=/tmp/kcoro_trace.log
exec "$@"
