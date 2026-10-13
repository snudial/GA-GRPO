"""Bounded admission control; never revoke an episode's existing server lease."""
import json
import time
from pathlib import Path


def allowed_ports(ports, control_path, now=None):
    if not control_path:
        return list(ports)
    fallback=min(64,len(ports));now=time.time() if now is None else now
    try:
        data=json.loads(Path(control_path).read_text())
        limit=data['active_slots']
        if type(limit) is not int or not fallback<=limit<=len(ports):raise ValueError('invalid limit')
        if not 0<=now-data['updated']<=180:raise ValueError('stale control')
    except (OSError,ValueError,KeyError,TypeError):limit=fallback
    return list(ports[:limit])


class AdmissionTuner:
    """Use sustained inference windows only, with bounded steps and rollback."""
    def __init__(self, minimum=64, maximum=128, step=16):
        self.minimum=minimum;self.maximum=maximum;self.step=step
        self.slots=minimum;self.last=None;self.trial=None;self.bad_windows=0

    def decide(self, window):
        if window['active_samples']<4:
            return self.slots,'hold: insufficient awake-inference samples'
        old=self.slots
        pressure=window['cpu_busy']>90 or window['preemptions']>0
        if self.trial:
            baseline=self.trial
            worse=window['request_rate']<baseline['request_rate']*.75 and (window['cpu_busy']>baseline['cpu_busy']+10 or pressure)
            self.bad_windows=self.bad_windows+1 if worse else 0
            if self.bad_windows>=2 or pressure:
                self.slots=baseline['slots'];self.trial=None;self.bad_windows=0
                self.last=window
                return self.slots,'rollback: sustained throughput regression with pressure'
            if window['request_rate']>=baseline['request_rate']*.9:
                self.trial=None
        if pressure:
            self.slots=max(self.minimum,self.slots-self.step)
            reason='reduce: CPU pressure or KV preemption'
        elif self.trial is None and window['gpu_util']<90 and window['leased_fraction']>=.85 and window['cpu_busy']<80:
            self.slots=min(self.maximum,self.slots+self.step)
            reason='increase: environment slots occupied and inference GPUs underfed'
            if self.slots>old:self.trial=dict(slots=old,request_rate=window['request_rate'],cpu_busy=window['cpu_busy'])
        else:reason='hold: sufficient GPU activity, tail, or no CPU headroom'
        self.last=window
        return self.slots,reason
