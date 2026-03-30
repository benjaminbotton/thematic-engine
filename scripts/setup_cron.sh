#!/bin/bash
# Installs the crontab with correct paths for this machine.
# Run once after cloning: bash scripts/setup_cron.sh

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCAN="$REPO_DIR/scripts/scan.sh"
FETCH="$REPO_DIR/scripts/fetch_and_scan.sh"

cat <<CRON | crontab -
# Thematic Catalyst L/S Engine — automated trading scans
# Installed: $(date)
# Remove: crontab -r
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin

# Pre-market: fetch fresh data + scan (6:03 AM PT, weekdays)
3 6 * * 1-5 $FETCH

# Intraday scans every 30 min (6:30 AM - 1:00 PM PT, weekdays)
30 6 * * 1-5 $SCAN
0,30 7-12 * * 1-5 $SCAN
0 13 * * 1-5 $SCAN
CRON

echo "Crontab installed for: $REPO_DIR"
echo ""
crontab -l
echo ""
echo "Make sure your Mac stays awake during market hours (6-1 PM PT)."
echo "System Preferences > Energy Saver > Prevent automatic sleeping"
