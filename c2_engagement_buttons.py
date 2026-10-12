"""Offline geometry button baseline; independent of learned BC scores."""
class EngagementButtons:
    def __init__(self, confirm_ms=60, aim_release_ms=150):
        self.confirm_ns=int(confirm_ms*1e6)
        self.release_ns=int(aim_release_ms*1e6)
        self.reset()

    def reset(self):
        self.target_id=None
        self.aligned_since=None
        self.last_valid_ns=None
        self.last_ns=None

    def update(self, now_ns, target_id, valid, aligned, enabled=True):
        if self.last_ns is not None and now_ns<self.last_ns:
            raise ValueError('Button time moved backward')
        self.last_ns=now_ns
        if not enabled:
            self.reset()
            return False,False
        valid=bool(valid and target_id is not None)
        if valid:
            self.last_valid_ns=now_ns
        aim=valid or (self.last_valid_ns is not None and now_ns-self.last_valid_ns<self.release_ns)
        if not valid or not aligned:
            self.aligned_since=None
            self.target_id=None
            return False,aim
        if target_id!=self.target_id:
            self.aligned_since=now_ns
            self.target_id=target_id
        fire=now_ns-self.aligned_since>=self.confirm_ns
        return fire,aim
