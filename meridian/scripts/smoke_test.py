import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.demo import load,vendor_assessment
from src.budget import doc_cost,decision
from src.ledger import Ledger
vendors,evidence=load(); assert len(vendors)==6
for v in vendors:
 a=vendor_assessment(v,evidence[v['vendor_id']],1000,1000); assert 0<=a['p_ai']<=1
assert doc_cost([{'type':'EMAIL','confidence':.5}],.2)>0
assert decision(800,1000,10)=='AUTO' and decision(300,1000,10)=='STRICT' and decision(100,1000,10)=='REVIEW' and decision(40,1000,10)=='BLOCK' and decision(900,1000,1,True)=='BLOCK'
l=Ledger('ledger/test.db'); l.append('Demo','D1',10,100,90); l.append('Demo','D2',5,90,85); assert l.verify(); print('SMOKE TEST PASSED')
