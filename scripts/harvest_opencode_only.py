import sys, time
from src.config import load
from src.scheduler import Scheduler
from src.harvest import plan_harvest

cfg = load()
sched = Scheduler(cfg)
plan = plan_harvest(cfg)
print(f"[harvest-driver] plan_id={plan.plan_id} batch={plan.batch_labels} total_new={plan.total_new}")
if not plan.should_harvest:
    print("[harvest-driver] nothing to harvest"); sys.exit(0)
t0 = time.time()
ok, stats = sched.harvest(plan)
print(f"[harvest-driver] ok={ok} elapsed={time.time()-t0:.0f}s")
print(stats)
sys.exit(0 if ok else 1)
