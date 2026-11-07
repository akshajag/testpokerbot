#!/usr/bin/env bash
# Compare our internal time accounting with the judge's charged time.
cd "$(dirname "$0")/.."
BRAIN_LOG=1 python agent.py 2> dev/agent_stderr.log >/dev/null &
ME=$!
python baseline-bots/baseline_always_pot.py >/dev/null 2>&1 &
OPP=$!
for i in $(seq 1 60); do curl -s 127.0.0.1:5008/ >/dev/null && curl -s 127.0.0.1:5009/ >/dev/null && break; sleep 0.5; done
for seed in 41 42; do
  python judge_engine.py --seed $seed --agent1 http://127.0.0.1:5008 --agent2 http://127.0.0.1:5009 | grep -E "agent1 half"
done
kill $ME $OPP
grep brain dev/agent_stderr.log
