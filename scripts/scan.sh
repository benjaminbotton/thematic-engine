#!/bin/bash
# Thematic Engine — auto-execute scan
# Loads .env, runs scan with live prices, auto-executes trades

cd "$(dirname "$0")/.."

# Load env vars
set -a
source .env
set +a

python3 -c "
import sys
sys.path.insert(0, 'src')
from engine import cmd_scan
cmd_scan()
" >> data/scan.log 2>&1

echo "---" >> data/scan.log
