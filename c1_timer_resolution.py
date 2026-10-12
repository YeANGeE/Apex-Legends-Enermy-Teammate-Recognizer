"""Scoped Windows wait-timer request, restored on ordinary exit/errors."""
import ctypes


class TimerResolution:
    def __init__(self,api=None):
        self.api=api
        self.active=False

    def start(self):
        if self.active:return
        if self.api is None:
            self.api=ctypes.WinDLL('winmm')
            self.api.timeBeginPeriod.argtypes=[ctypes.c_uint]
            self.api.timeBeginPeriod.restype=ctypes.c_uint
            self.api.timeEndPeriod.argtypes=[ctypes.c_uint]
            self.api.timeEndPeriod.restype=ctypes.c_uint
        if self.api.timeBeginPeriod(1)!=0:
            raise RuntimeError('1ms timer resolution request failed')
        self.active=True

    def stop(self):
        if self.active:
            result=self.api.timeEndPeriod(1)
            self.active=False
            if result!=0:raise RuntimeError('Timer resolution cleanup failed')
