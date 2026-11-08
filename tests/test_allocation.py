import numpy as np
import pytest

from leg_sim.allocation import (
    METHODS,
    allocate,
    allocate_dynamic,
    allocate_fair,
    allocate_static,
    static_shares,
)


@pytest.mark.parametrize("method", METHODS)
def test_balance_closes_exactly_every_interval(method, rand_case):
    loads, prod, shares = rand_case
    a = allocate(method, loads, prod, shares)
    export = prod - a.sum(axis=1)
    assert a.dtype == np.int64
    assert (a.sum(axis=1) + export == prod).all()
    assert (export >= 0).all()


@pytest.mark.parametrize("method", METHODS)
def test_member_never_gets_more_than_consumed_or_negative(method, rand_case):
    loads, prod, shares = rand_case
    a = allocate(method, loads, prod, shares)
    assert (a >= 0).all() and (a <= loads).all()


@pytest.mark.parametrize("method", ["dynamic", "fair"])
def test_efficient_methods_use_all_usable_production(method, rand_case):
    loads, prod, shares = rand_case
    a = allocate(method, loads, prod, shares)
    assert (a.sum(axis=1) == np.minimum(prod, loads.sum(axis=1))).all()


def test_static_wastes_surplus_that_dynamic_uses(rand_case):
    loads, prod, shares = rand_case
    assert allocate_static(loads, prod, shares).sum() < allocate_dynamic(loads, prod).sum()


def test_static_hand_example_caps_at_share():
    loads = np.array([[100, 300]])
    out = allocate_static(loads, np.array([200]), np.array([0.25, 0.75]))
    assert out.tolist() == [[50, 150]]


def test_static_unused_share_is_exported_not_redistributed():
    loads = np.array([[10, 300]])
    out = allocate_static(loads, np.array([200]), np.array([0.5, 0.5]))
    assert out.tolist() == [[10, 100]]


def test_dynamic_hand_example_proportional():
    out = allocate_dynamic(np.array([[100, 300]]), np.array([200]))
    assert out.tolist() == [[50, 150]]


def test_dynamic_surplus_meets_all_demand():
    out = allocate_dynamic(np.array([[100, 300]]), np.array([900]))
    assert out.tolist() == [[100, 300]]


def test_fair_redistributes_unused_entitlement():
    out = allocate_fair(np.array([[50, 300]]), np.array([200]), np.array([1.0, 1.0]))
    assert out.tolist() == [[50, 150]]


def test_fair_respects_weights():
    out = allocate_fair(np.array([[300, 300]]), np.array([400]), np.array([1.0, 3.0]))
    assert out.tolist() == [[100, 300]]


def test_fair_guarantees_entitlement_floor(rand_case):
    loads, prod, shares = rand_case
    a = allocate_fair(loads, prod, shares)
    floor = np.minimum(loads, np.floor(prod[:, None] * shares[None, :]))
    assert (a >= floor - 1).all()  # -1: whole-Wh rounding


def test_fair_equals_dynamic_when_weights_follow_load():
    loads = np.array([[100, 300, 600]])
    w = np.array([1.0, 3.0, 6.0])
    assert allocate_fair(loads, np.array([500]), w).tolist() == allocate_dynamic(loads, np.array([500])).tolist()


def test_zero_production_gives_zero_allocation(rand_case):
    loads, _, shares = rand_case
    for m in METHODS:
        assert allocate(m, loads, np.zeros(len(loads), dtype=np.int64), shares).sum() == 0


def test_zero_load_gives_zero_allocation():
    loads = np.zeros((5, 3), dtype=np.int64)
    for m in METHODS:
        assert allocate(m, loads, np.full(5, 100), np.array([0.2, 0.3, 0.5])).sum() == 0


def test_rounding_keeps_interval_total_whole_wh():
    out = allocate_dynamic(np.array([[1, 1, 1]]), np.array([1]))
    assert out.sum() == 1 and out.max() == 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        allocate_dynamic(np.ones((3, 2), dtype=np.int64), np.ones(4, dtype=np.int64))


def test_negative_values_raise():
    with pytest.raises(ValueError, match="negative"):
        allocate_dynamic(np.array([[-1, 2]]), np.array([5]))


def test_invalid_static_shares_raise():
    with pytest.raises(ValueError):
        allocate_static(np.ones((2, 2), dtype=np.int64), np.ones(2, dtype=np.int64), np.array([0.6, 0.6]))


def test_invalid_fair_weights_raise():
    with pytest.raises(ValueError):
        allocate_fair(np.ones((2, 2), dtype=np.int64), np.ones(2, dtype=np.int64), np.array([1.0, 0.0]))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown allocation method"):
        allocate("magic", np.ones((1, 1), dtype=np.int64), np.ones(1, dtype=np.int64), np.array([1.0]))


def test_static_shares_normalise_and_validate():
    assert static_shares(np.array([1, 3])).tolist() == [0.25, 0.75]
    with pytest.raises(ValueError):
        static_shares(np.array([0, 0]))
