"""Causal host-stage timing, not photon/display or device latency."""


def split_capture_latency(packet, inference_start_ns):
    stamps=(packet.timestamp_ns,packet.capture_copy_start_ns,packet.capture_copy_end_ns,
            packet.capture_ready_ns,inference_start_ns)
    if any(value is None for value in stamps):
        return {}
    if any(b<a for a,b in zip(stamps,stamps[1:])):
        raise ValueError('Capture diagnostic timestamps are not causal in QPC coordinates')
    source,start,end,ready,infer=stamps
    return dict(source_to_capture_copy=(start-source)/1e6,capture_copy=(end-start)/1e6,
                capture_dispatch=(ready-end)/1e6,capture_to_inference=(infer-ready)/1e6)
