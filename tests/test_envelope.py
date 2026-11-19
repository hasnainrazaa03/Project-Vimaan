"""Speed margins (vimaan_nlu.envelope), with the .acf limits of the 737-800
and the Baron 58 (X-Plane 12.4.3)."""

import pytest
from vimaan_nlu.envelope import Limits, flap_notch, margins

B738 = Limits(
    vs_kt=139.0,
    vso_kt=114.0,
    vfe_first_kt=250.0,
    vfe_full_kt=162.0,
    flap_detents=8,
    vle_kt=270.0,
    vno_kt=280.0,
    vne_kt=340.0,
    mmo=0.82,
    gear_retractable=True,
)
BE58 = Limits(
    vs_kt=84.0,
    vso_kt=73.0,
    vfe_first_kt=152.0,
    vfe_full_kt=122.0,
    flap_detents=2,
    vle_kt=152.0,
    vno_kt=195.0,
    vne_kt=223.0,
    mmo=0.0,  # a piston has none
    gear_retractable=True,
)


def test_clean_in_cruise():
    m = margins(B738, ias_kt=280.0, mach=0.78, flaps=0.0, gear=0.0, on_ground=False)
    assert m == {
        "stall": 141.0,  # above the clean stall
        "flaps": None,  # flaps up
        "gear": None,  # gear up
        "vno": 0.0,
        "vne": 60.0,
        "mmo": 0.04,
    }


def test_full_flaps_too_fast_is_beyond_the_full_flap_limit():
    m = margins(B738, ias_kt=170.0, mach=0.3, flaps=1.0, gear=1.0, on_ground=False)
    assert m["flaps"] == -8.0  # flaps 40: 162 kt
    assert m["stall"] == 56.0  # against Vso with flaps out
    assert m["gear"] == 100.0


@pytest.mark.parametrize(
    ("ratio", "notch", "limit"), [(0.125, 1, 250.0), (0.5, 4, 212.3), (0.75, 6, 187.1)]
)
def test_each_notch_has_its_own_flap_limit(ratio, notch, limit):
    assert flap_notch(ratio, 8) == notch
    m = margins(B738, ias_kt=0.0, flaps=ratio, on_ground=False)
    assert m["flaps"] == pytest.approx(limit, abs=0.1)


def test_gear_out_above_vle():
    assert margins(BE58, ias_kt=160.0, gear=1.0, on_ground=False)["gear"] == -8.0
    assert margins(BE58, ias_kt=160.0, gear=0.5, on_ground=False)["gear"] == -8.0  # in transit
    assert margins(BE58, ias_kt=160.0, gear=0.0, on_ground=False)["gear"] is None


def test_fixed_gear_has_no_gear_margin():
    fixed = Limits(vle_kt=0.0, gear_retractable=False, vne_kt=163.0)
    assert margins(fixed, ias_kt=100.0, gear=1.0, on_ground=False)["gear"] is None


def test_no_stall_margin_on_the_ground_and_none_without_a_reading():
    assert margins(BE58, ias_kt=20.0, gear=1.0, on_ground=True)["stall"] is None
    m = margins(BE58, ias_kt=None, mach=None, flaps=0.5, gear=1.0, on_ground=False)
    assert all(v is None for v in m.values())


def test_a_piston_has_no_mach_margin():
    assert margins(BE58, ias_kt=180.0, mach=0.3, on_ground=False)["mmo"] is None


def test_unknown_limits_give_no_margins():
    assert all(
        v is None
        for v in margins(Limits(), ias_kt=150.0, mach=0.4, flaps=0.5, on_ground=False).values()
    )


def test_flap_notch_bounds():
    assert flap_notch(0.0, 8) == 0 and flap_notch(1.0, 8) == 8 and flap_notch(1.2, 8) == 8
    assert flap_notch(0.5, 0) == 0
