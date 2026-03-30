#!/bin/bash
# Thematic Engine — pre-market fetch + scan
# Runs once at 6 AM PT: fetches fresh data from Polygon, then scans

cd "$(dirname "$0")/.."

# Load env vars
set -a
source .env
set +a

python3 -c "
import sys
sys.path.insert(0, 'src')
from engine import cmd_fetch, cmd_scan
print('=== PRE-MARKET FETCH + SCAN ===')
cmd_fetch()
print()
cmd_scan()
" >> data/scan.log 2>&1

echo "---" >> data/scan.log
