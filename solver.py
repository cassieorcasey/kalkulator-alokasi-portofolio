"""Solver SPLTV untuk Kalkulator Alokasi Portofolio.

Aturan modul:
    - Tanpa print/input: hanya return nilai atau raise exception.
    - Tanpa numpy. numpy hanya dipakai di tests/ sebagai pembanding.

Model (spesifikasi v2, bagian 3):
    x + y + z          = M
    rx*x + ry*y + rz*z = P
    x                  = E
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Matrix = list[list[float]]
Vector = list[float]

# |rz - ry| di bawah 0,01% dianggap singular / ill-conditioned.
EPS_SINGULAR = 1e-4
# Pivot di bawah nilai ini dianggap nol.
EPS_PIVOT = 1e-12
# Nilai di [-EPS_ROUNDING, 0) adalah noise floating point, dijadikan 0.
EPS_ROUNDING = 1e-6


class PortfolioError(Exception):
    """Induk semua error solver."""


class InvalidInputError(PortfolioError):
    """Input di luar domain: M <= 0, E < 0, atau E >= M."""


class SingularSystemError(PortfolioError):
    """|rz - ry| < EPS_SINGULAR, atau pivot < EPS_PIVOT saat eliminasi."""


class InfeasibleTargetError(PortfolioError):
    """Target P di luar [p_min, p_max]. Batasnya disimpan untuk pesan ke user."""

    def __init__(self, p: float, p_min: float, p_max: float):
        self.p = p
        self.p_min = p_min
        self.p_max = p_max
        super().__init__(
            f"Target {p:,.0f} di luar rentang feasible "
            f"{p_min:,.0f} – {p_max:,.0f}"
        )


@dataclass(frozen=True)
class Allocation:
    x: int                      # RDPU (Rp)
    y: int                      # SBN (Rp)
    z: int                      # Saham (Rp)
    p_target: float
    p_aktual: float             # dari x, y, z setelah pembulatan
    peringatan: list[str] = field(default_factory=list)

    @property
    def selisih(self) -> float:
        return self.p_aktual - self.p_target


# Aljabar linear umum. Tidak bergantung pada model portofolio.

def det3(A: Matrix) -> float:
    """Determinan matriks 3x3.

    Rumus (ekspansi kofaktor baris pertama, atau pakai Sarrus):
        a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

    Raise:
        ValueError  A bukan 3x3.
    """
    if len(A) != 3 or any(len(baris) != 3 for baris in A):
        raise ValueError("det3 membutuhkan matriks 3x3")

    (a11, a12, a13), (a21, a22, a23), (a31, a32, a33) = A

    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


def gauss_solve(A: Matrix, b: Vector) -> Vector:
    """Selesaikan A·X = b (n x n) dengan eliminasi Gauss + partial pivoting.

    Langkah:
        1. Salin ke augmented matrix [A | b]. Input asli tidak boleh berubah.
        2. Untuk tiap kolom k = 0..n-1:
           a. Pivot = baris p >= k dengan |M[p][k]| terbesar.
           b. |M[p][k]| < EPS_PIVOT -> raise SingularSystemError.
           c. Tukar baris k dan p.
           d. Untuk tiap baris i > k: baris_i -= (M[i][k] / M[k][k]) * baris_k.
        3. Substitusi balik, i = n-1..0:
           x[i] = (M[i][n] - sum(M[i][j] * x[j] for j > i)) / M[i][i]

    Raise:
        ValueError           A tidak persegi, atau len(b) != n.
        SingularSystemError  pivot < EPS_PIVOT.
    """
    n = len(A)
    if any(len(baris) != n for baris in A):
        raise ValueError("A harus matriks persegi")
    if len(b) != n:
        raise ValueError(f"len(b) = {len(b)}, seharusnya {n}")

    aug = [[float(v) for v in A[i]] + [float(b[i])] for i in range(n)]

    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[p][k]) < EPS_PIVOT:
            raise SingularSystemError(f"pivot kolom {k} bernilai nol")
        aug[k], aug[p] = aug[p], aug[k]

        for i in range(k + 1, n):
            faktor = aug[i][k] / aug[k][k]
            for j in range(k, n + 1):
                aug[i][j] -= faktor * aug[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        jumlah = sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug[i][n] - jumlah) / aug[i][i]
    return x


# Model portofolio. Memakai det3 dan gauss_solve.

def build_system(
    M: float, P: float, E: float, rx: float, ry: float, rz: float
) -> tuple[Matrix, Vector]:
    """Return (A, b) sesuai spesifikasi 3D."""
    A = [
        [1, 1, 1],
        [rx, ry, rz],
        [1, 0, 0],
    ]
    b = [M, P, E]
    return A, b


def feasible_range(
    M: float, E: float, rx: float, ry: float, rz: float
) -> tuple[float, float]:
    """Rentang target P yang menghasilkan x, y, z >= 0.

    Rumus (berlaku untuk urutan return apa pun):
        p_min = rx*E + (M - E) * min(ry, rz)
        p_max = rx*E + (M - E) * max(ry, rz)

    Return:
        (p_min, p_max)
    """
    profit_siaga = rx * E
    sisa_modal = M - E
    p_min = profit_siaga + sisa_modal * min(ry, rz)
    p_max = profit_siaga + sisa_modal * max(ry, rz)
    return p_min, p_max


def validate_inputs(
    M: float, E: float, rx: float, ry: float, rz: float
) -> list[str]:
    """Pre-check input sebelum solve (spesifikasi 4A). P dicek terpisah.

    Return:
        Daftar peringatan, boleh kosong. Peringatan muncul jika ry > rz
        atau rx > ry.

    Raise:
        InvalidInputError    M <= 0, E < 0, atau E >= M.
        SingularSystemError  |rz - ry| < EPS_SINGULAR.
    """
    if M <= 0:
        raise InvalidInputError("Modal M harus lebih dari 0.")
    if E < 0:
        raise InvalidInputError("Dana siaga E tidak boleh negatif.")
    if E >= M:
        raise InvalidInputError("Dana siaga E harus lebih kecil dari modal M.")
    if abs(rz - ry) < EPS_SINGULAR:
        raise SingularSystemError(
            "Return SBN dan saham terlalu mirip (selisih < 0,01%), "
            "alokasi keduanya tidak bisa ditentukan."
        )

    peringatan = []
    if ry > rz:
        peringatan.append(
            "Return SBN > saham: target lebih tinggi dialokasikan ke SBN, bukan saham."
        )
    if rx > ry:
        peringatan.append("Return RDPU > SBN: tidak lazim, periksa kembali input return.")
    return peringatan


def clean_near_zero(values: Vector, tol: float = EPS_ROUNDING) -> Vector:
    """Post-check: nilai di [-tol, 0) dijadikan 0.0.

    Raise:
        AssertionError  ada nilai < -tol. Artinya pre-check atau derivasi salah.
    """
    hasil = []
    for v in values:
        assert v >= -tol, f"Nilai {v} negatif di luar toleransi {tol}"
        hasil.append(0.0 if v < 0 else v)
    return hasil


def round_allocation(y: float, z: float, M: int) -> tuple[int, int, int]:
    """Bulatkan alokasi ke rupiah (spesifikasi 4C).

    Rumus:
        y, z = floor(y), floor(z)
        x    = M - y - z        sisa pembulatan masuk ke RDPU

    Return:
        (x, y, z) sebagai int, dengan x + y + z == M tepat.
    """
    y_bulat = math.floor(y)
    z_bulat = math.floor(z)
    x_bulat = M - y_bulat - z_bulat
    return x_bulat, y_bulat, z_bulat


def allocate(
    M: int, P: float, E: int, rx: float, ry: float, rz: float
) -> Allocation:
    """Hitung alokasi lengkap dari input user.

    Langkah:
        1. peringatan = validate_inputs(...)
        2. P di luar feasible_range(...) -> raise InfeasibleTargetError(P, p_min, p_max)
        3. A, b = build_system(...)
        4. solusi = clean_near_zero(gauss_solve(A, b))
        5. x, y, z = round_allocation(...)
        6. p_aktual = rx*x + ry*y + rz*z
        7. Return Allocation(...)
    """
    peringatan = validate_inputs(M, E, rx, ry, rz)

    p_min, p_max = feasible_range(M, E, rx, ry, rz)
    if not p_min <= P <= p_max:
        raise InfeasibleTargetError(P, p_min, p_max)

    A, b = build_system(M, P, E, rx, ry, rz)
    _, y, z = clean_near_zero(gauss_solve(A, b))
    x, y, z = round_allocation(y, z, M)

    p_aktual = rx * x + ry * y + rz * z
    return Allocation(x, y, z, P, p_aktual, peringatan)