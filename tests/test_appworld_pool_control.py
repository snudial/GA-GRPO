import json
from appworld_ga.pool_control import AdmissionTuner,allowed_ports

def window(**kw):
    return dict(dict(active_samples=6,gpu_util=65,cpu_busy=40,leased_fraction=1,request_rate=20,preemptions=0),**kw)

def test_sleep_and_drained_tail_do_not_expand_pool():
    t=AdmissionTuner()
    assert t.decide(window(active_samples=0))[0]==64
    assert t.decide(window(leased_fraction=.1))[0]==64
    assert t.decide(window(gpu_util=98))[0]==64

def test_bound_increases_and_rollback_with_sustained_regression():
    t=AdmissionTuner()
    assert t.decide(window())[0]==80
    assert t.decide(window(request_rate=10,cpu_busy=60))[0]==80
    assert t.decide(window(request_rate=10,cpu_busy=60))[0]==64
    for _ in range(20):t.decide(window(request_rate=25))
    assert t.slots==128

def test_preemption_rolls_back_trial():
    t=AdmissionTuner();t.decide(window())
    assert t.decide(window(preemptions=1))[0]==64

def test_stale_invalid_admission_falls_back_without_mutating_ports(tmp_path):
    ports=list(range(19300,19428));p=tmp_path/'control.json'
    assert len(allowed_ports(ports,str(p),now=1000))==64
    p.write_text(json.dumps(dict(active_slots=96,updated=999)))
    assert len(allowed_ports(ports,str(p),now=1000))==96
    assert len(allowed_ports(ports,str(p),now=1200))==64
    p.write_text(json.dumps(dict(active_slots=200,updated=999)))
    assert len(allowed_ports(ports,str(p),now=1000))==64
    assert len(ports)==128
    assert allowed_ports(ports,None)==ports

def test_live_admission_expansion_and_shrink_preserve_existing_leases(tmp_path,monkeypatch):
    import asyncio,time
    from appworld_ga import environment as env
    control=tmp_path/'admission.json';monkeypatch.setenv('APPWORLD_POOL_CONTROL',str(control))
    monkeypatch.setattr(env,'LEASE_DIR',str(tmp_path/'leases'))
    def set_limit(n):control.write_text(json.dumps(dict(active_slots=n,updated=time.time())))
    ports=list(range(25000,25096));held=[env._try_lease(p) for p in ports[:64]];set_limit(64)
    async def run():
        task=asyncio.create_task(env.acquire(ports,poll=.01,timeout=3))
        await asyncio.sleep(.15)
        assert not task.done(),'Admission must wait while all allowed slots are held'
        set_limit(80);lease=await asyncio.wait_for(task,2)
        assert lease.port in ports[64:80]
        set_limit(64)
        assert env._try_lease(lease.port) is None,'Shrinking admission must not revoke an existing lease'
        lease.release()
    try:asyncio.run(run())
    finally:
        for lease in held:lease.release()
