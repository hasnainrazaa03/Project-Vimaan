"""Mac usage sampling (vimaan_nlu.usage), on recorded tool output."""

import pytest
from vimaan_nlu import usage

PS_1 = "  101   1:00.00 2048000\n  202   0:02.50 983040\n"
PS_2 = "  101   1:30.00 2050000\n  202   0:03.00 990000\n"
IOREG = (
    '| "PerformanceStatistics" = {"Alloc system memory"=2682175488,"Tiler Utilization %"=3,'
    '"Renderer Utilization %"=59,"Device Utilization %"=75,"In use system memory"=653197312}'
)


@pytest.mark.parametrize(
    "text,seconds",
    [("0:01.25", 1.25), ("12:03.50", 723.5), ("1:02:03", 3723.0), ("2-01:02:03", 176523.0)],
)
def test_ps_cpu_time(text, seconds):
    assert usage.parse_cputime(text) == pytest.approx(seconds)


def test_ps_lines():
    assert usage.parse_ps(PS_1 + "garbage\n") == {101: (60.0, 2000.0), 202: (2.5, 960.0)}


def test_gpu_counters():
    assert usage.parse_ioreg(IOREG) == {
        "gpu_pct": 75.0,
        "gpu_renderer_pct": 59.0,
        "gpu_tiler_pct": 3.0,
        "gpu_mem_mb": 623,
    }
    assert usage.parse_ioreg("no accelerator") == {}


def test_swap():
    assert usage.parse_swap("total = 6144.00M  used = 4763.00M  free = 1381.00M") == 4763
    assert usage.parse_swap("total = 8.00G  used = 1.50G  free = 6.50G") == 1536
    assert usage.parse_swap("") is None


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_cpu_is_the_share_of_one_core_since_the_last_sample():
    outputs = iter([PS_1, PS_2])
    clock = Clock()

    def run(argv):
        if argv[0] == "ps":
            return next(outputs)
        if argv[0] == "ioreg":
            return IOREG
        if argv[-1] == "kern.memorystatus_level":
            return "53\n"
        return "total = 6144.00M  used = 4763.00M  free = 1381.00M"

    s = usage.UsageSampler({"xplane": 101, "worker": 202}, run=run, clock=clock)
    first = s.sample()
    assert "xplane_cpu_pct" not in first  # no baseline yet
    assert first["xplane_rss_mb"] == 2000 and first["worker_rss_mb"] == 960
    clock.now = 15.0
    second = s.sample()
    assert second["xplane_cpu_pct"] == 200.0  # 30 s of CPU in 15 s: two cores
    assert second["worker_cpu_pct"] == pytest.approx(3.3, abs=0.05)
    assert second["gpu_pct"] == 75.0 and second["mem_free_pct"] == 53
    assert second["swap_used_mb"] == 4763 and "load_1m" in second


def test_a_missing_tool_leaves_its_fields_out():
    def run(argv):
        raise FileNotFoundError(argv[0])

    out = usage.UsageSampler({"xplane": 1}, run=run).sample()
    assert not any(k.startswith(("xplane_", "gpu_", "mem_", "swap_")) for k in out)
