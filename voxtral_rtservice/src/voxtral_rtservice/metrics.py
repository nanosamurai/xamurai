"""Bounded-cardinality capacity and latency metrics; no tenant/session labels."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else CollectorRegistry()
        self.active = Gauge("xamurai_voxtral_active_sessions", "Admitted streams", registry=self.registry)
        self.capacity = Gauge("xamurai_voxtral_session_capacity", "Configured admission capacity", registry=self.registry)
        self.admissions = Counter("xamurai_voxtral_admissions", "Admission outcomes", ("outcome",), registry=self.registry)
        self.closed = Counter("xamurai_voxtral_closed_sessions", "Stream completion outcomes", ("outcome",), registry=self.registry)
        self.first_partial = Histogram("xamurai_voxtral_first_partial_seconds", "Epoch start to first nonempty text", registry=self.registry)
        self.flush = Histogram("xamurai_voxtral_flush_seconds", "Input epoch EOF to completed inference", registry=self.registry)
