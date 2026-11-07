#!/usr/bin/env bash
# Run the OFFICIAL judge (over HTTP) against every baseline bot.
#   bash dev/judge_all.sh [deals] [seed]
cd "$(dirname "$0")/.."
DEALS=${1:-1000}
SEED=${2:-7}
wait_up() { for i in $(seq 1 60); do curl -s "http://127.0.0.1:$1/" >/dev/null && return 0; sleep 0.5; done; return 1; }

python agent.py >/dev/null 2>&1 &
ME=$!
wait_up 5008 || { echo "agent failed to start"; kill $ME; exit 1; }
for b in baseline-bots/*.py; do
  python "$b" >/dev/null 2>&1 &
  OPP=$!
  wait_up 5009
  echo "=== vs $(basename "$b" .py)"
  python judge_engine.py --deals "$DEALS" --seed "$SEED" --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 \
    | grep -E "agent1: |interval|agent1 half|RESULT"
  kill $OPP; wait $OPP 2>/dev/null
done
kill $ME
