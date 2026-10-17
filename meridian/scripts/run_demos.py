"""Run the three competition demos headlessly (temporary state; production untouched)."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import scenarios
a = scenarios.demo_safe(); r = a['result']
print(f"DEMO 1  SAFE        {a['vendor']}: P_ai {a['state']['p_ai']:.0%}, remaining {r['remaining_after']:.0%} -> {r['decision']}"); assert r['decision'] == 'AUTO'
b = scenarios.demo_escalation(); x, y = b['before'], b['after']
print(f"DEMO 2  ESCALATION  {b['vendor']}: P_ai {x['p_ai']:.0%}->{y['p_ai']:.0%}, budget {x['allowance']:.0f}->{y['allowance']:.0f}, "
      f"{b['result_before']['decision']} -> {b['result_after']['decision']}"); assert (b['result_before']['decision'], b['result_after']['decision']) == ('AUTO', 'REVIEW')
c = scenarios.demo_hard_rule(); r = c['result']
print(f"DEMO 3  HARD RULE   {c['vendor']}: remaining {r['remaining_after']:.0%} -> {r['decision']} ({r['why']['reasons'][0]})"); assert r['decision'] == 'BLOCK'
print('ALL DEMOS PASSED')
