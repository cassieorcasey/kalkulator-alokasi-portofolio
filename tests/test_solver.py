"""Test solver.py. numpy dipakai sebagai pembanding independen.

Jalankan dari root project: pytest -v
"Rencana test N" = butir N di spesifikasi v2 bagian 7.
"""

import math

import numpy as np
import pytest

from solver import (
    EPS_SINGULAR,
    Allocation,
    InfeasibleTargetError,
    InvalidInputError,
    SingularSystemError,
    allocate,
    build_system,
    clean_near_zero,
    det3,
    feasible_range,
    gauss_solve,
    round_allocation,
    validate_inputs,
)

# Contoh spesifikasi bagian 3F.
M, E, P = 100_000_000, 20_000_000, 9_000_000
RX, RY, RZ = 0.05, 0.065, 0.12

rng = np.random.default_rng(42)


# Plan test 1–2

@pytest.mark.parametrize("seed", range(20))
def test_det3_sama_dengan_numpy(seed):
    A = np.random.default_rng(seed).uniform(-10, 10, (3, 3))
    assert det3(A.tolist()) == pytest.approx(np.linalg.det(A), rel=1e-9, abs=1e-9)


def test_det3_matriks_model_sama_dengan_rz_minus_ry():
    A, _ = build_system(M, P, E, RX, RY, RZ)
    assert det3(A) == pytest.approx(RZ - RY)


def test_det3_bukan_3x3():
    with pytest.raises(ValueError):
        det3([[1, 2], [3, 4]])


# Plan 1

@pytest.mark.parametrize("n", [2, 3, 4, 6])
@pytest.mark.parametrize("seed", range(10))
def test_gauss_sama_dengan_numpy(n, seed):
    g = np.random.default_rng(seed)
    A = g.uniform(-10, 10, (n, n)) + n * np.eye(n)  # + n*I menjauhkan A dari singular
    b = g.uniform(-100, 100, n)
    assert np.allclose(gauss_solve(A.tolist(), b.tolist()), np.linalg.solve(A, b))


def test_gauss_butuh_pivoting():
    # Pivot awal 0: tanpa partial pivoting terjadi pembagian dengan nol.
    A = [[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]]
    b = [3.0, 6.0, 4.0]
    assert np.allclose(gauss_solve(A, b), np.linalg.solve(A, b))


def test_gauss_tidak_mengubah_input():
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [3.0, 5.0]
    A_salin, b_salin = [r[:] for r in A], b[:]
    gauss_solve(A, b)
    assert A == A_salin and b == b_salin


def test_gauss_singular():
    with pytest.raises(SingularSystemError):
        gauss_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_gauss_ukuran_tidak_cocok():
    with pytest.raises(ValueError):
        gauss_solve([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0])


# Plan test 3

def test_build_system_bentuk():
    A, b = build_system(M, P, E, RX, RY, RZ)
    assert A == [[1, 1, 1], [RX, RY, RZ], [1, 0, 0]]
    assert b == [M, P, E]


def test_gauss_sama_dengan_solusi_tertutup():
    A, b = build_system(M, P, E, RX, RY, RZ)
    x, y, z = gauss_solve(A, b)
    z_tutup = (P - RX * E - RY * (M - E)) / (RZ - RY)
    assert x == pytest.approx(E)
    assert z == pytest.approx(z_tutup)
    assert y == pytest.approx((M - E) - z_tutup)


def test_feasible_range_contoh_spesifikasi():
    assert feasible_range(M, E, RX, RY, RZ) == pytest.approx((6_200_000, 10_600_000))


def test_feasible_range_urutan_terbalik_simetris():
    # Menukar ry dan rz tidak mengubah rentang.
    assert feasible_range(M, E, RX, RZ, RY) == pytest.approx(
        feasible_range(M, E, RX, RY, RZ)
    )


@pytest.mark.parametrize(
    "m, e",
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 150)],
)
def test_input_tidak_valid(m, e):
    with pytest.raises(InvalidInputError):
        validate_inputs(m, e, RX, RY, RZ)


@pytest.mark.parametrize("gap", [0.0, EPS_SINGULAR / 2, -EPS_SINGULAR / 2])
def test_return_terlalu_mirip(gap):
    with pytest.raises(SingularSystemError):
        validate_inputs(M, E, RX, RY, RY + gap)


def test_urutan_lazim_tanpa_peringatan():
    assert validate_inputs(M, E, RX, RY, RZ) == []


def test_ry_lebih_besar_dari_rz_ada_peringatan():
    assert len(validate_inputs(M, E, RX, 0.07, 0.06)) >= 1


def test_rx_lebih_besar_dari_ry_ada_peringatan():
    assert len(validate_inputs(M, E, 0.08, RY, RZ)) >= 1


def test_clean_near_zero():
    assert clean_near_zero([-3.7e-10, 5.0, 0.0]) == [0.0, 5.0, 0.0]


def test_clean_near_zero_gagal_jika_benar_benar_negatif():
    with pytest.raises(AssertionError):
        clean_near_zero([-1.0, 5.0, 3.0])


def test_round_allocation_contoh_spesifikasi():
    assert round_allocation(29_090_909.0909, 50_909_090.9090, M) == (
        20_000_001,
        29_090_909,
        50_909_090,
    )


def test_allocate_contoh_spesifikasi():
    hasil = allocate(M, P, E, RX, RY, RZ)
    assert isinstance(hasil, Allocation)
    assert (hasil.x, hasil.y, hasil.z) == (20_000_001, 29_090_909, 50_909_090)
    assert abs(hasil.selisih) < 1.0
    assert hasil.peringatan == []


# Plan test 4: di batas rentang, noise float tidak boleh menjadi nilai negatif.
@pytest.mark.parametrize("batas", [0, 1])
def test_allocate_tepat_di_batas(batas):
    p = feasible_range(M, E, RX, RY, RZ)[batas]
    hasil = allocate(M, p, E, RX, RY, RZ)
    assert min(hasil.x, hasil.y, hasil.z) >= 0


# Plan test 5
@pytest.mark.parametrize("geser", [-1, +1])
def test_allocate_di_luar_rentang(geser):
    p_min, p_max = feasible_range(M, E, RX, RY, RZ)
    p = p_min - 1 if geser < 0 else p_max + 1
    with pytest.raises(InfeasibleTargetError) as info:
        allocate(M, p, E, RX, RY, RZ)
    assert info.value.p_min == pytest.approx(p_min)
    assert info.value.p_max == pytest.approx(p_max)


# Plan test 6
def test_allocate_singular():
    with pytest.raises(SingularSystemError):
        allocate(M, P, E, RX, 0.08, 0.08)


# Plan test 7
def test_allocate_urutan_terbalik_valid_dengan_peringatan():
    ry, rz = 0.07, 0.06
    p_min, p_max = feasible_range(M, E, RX, ry, rz)
    hasil = allocate(M, (p_min + p_max) / 2, E, RX, ry, rz)
    assert min(hasil.x, hasil.y, hasil.z) >= 0
    assert len(hasil.peringatan) >= 1


# Plan test 8
@pytest.mark.parametrize("seed", range(200))
def test_invarian_acak(seed):
    g = np.random.default_rng(seed)
    m = int(g.integers(1_000_000, 10_000_000_000))
    e = int(g.integers(0, m))
    rx, ry, rz = g.uniform(0.0, 0.25, 3)
    if abs(rz - ry) < EPS_SINGULAR:
        pytest.skip("kebetulan singular")
    p_min, p_max = feasible_range(m, e, rx, ry, rz)
    p = float(g.uniform(p_min, p_max))

    hasil = allocate(m, p, e, rx, ry, rz)

    assert all(isinstance(v, int) for v in (hasil.x, hasil.y, hasil.z))
    assert hasil.x + hasil.y + hasil.z == m          # tepat, tanpa toleransi
    assert min(hasil.x, hasil.y, hasil.z) >= 0
    assert hasil.x >= e                               # sisa pembulatan hanya masuk ke x
    assert abs(hasil.selisih) < 1.0


# Plan test 9

def test_ill_conditioning_naik_saat_gap_mengecil():
    conds = []
    for gap in [0.05, 0.01, 0.001, 0.0001]:
        A, _ = build_system(M, P, E, RX, RY, RY + gap)
        conds.append(np.linalg.cond(np.array(A)))
    assert conds == sorted(conds)
    assert conds[-1] > 100 * conds[0]