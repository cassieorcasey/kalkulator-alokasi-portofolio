"""CLI Kalkulator Alokasi Portofolio.

Lapisan tipis di atas solver.py: baca input, ubah dana siaga menjadi E,
tampilkan hasil. Semua perhitungan ada di solver.py.

Jalankan: python cli.py
"""

from __future__ import annotations

from typing import Callable, TypeVar

from solver import (
    Allocation,
    InfeasibleTargetError,
    PortfolioError,
    allocate,
    feasible_range,
    validate_inputs,
)

T = TypeVar("T")

# Porsi dana siaga dari modal M (spesifikasi bagian 5).
PRESET_SIAGA = {
    "1": ("Konservatif", 0.40),
    "2": ("Moderat", 0.20),
    "3": ("Agresif", 0.10),
}
# Estimasi return bersih tahunan, dalam persen.
RETURN_DEFAULT = (5.0, 6.5, 12.0)


# Parsing dan format: fungsi murni, dites di tests/test_cli.py.

def parse_rupiah(teks: str) -> int:
    """'Rp 100.000.000', '100_000_000', atau '100000000' -> 100000000.

    Raise:
        ValueError  bukan bilangan bulat >= 0.
    """
    bersih = teks.strip().lower().removeprefix("rp")
    for karakter in (".", "_", " "):
        bersih = bersih.replace(karakter, "")
    if not bersih.isdigit():
        raise ValueError("Masukkan angka rupiah bulat, contoh: 100.000.000")
    return int(bersih)


def parse_persen(teks: str) -> float:
    """'6,5', '6.5', atau '6,5%' -> 0.065.

    Raise:
        ValueError  bukan angka.
    """
    bersih = teks.strip().removesuffix("%").strip().replace(",", ".")
    try:
        return float(bersih) / 100
    except ValueError:
        raise ValueError("Masukkan persen, contoh: 6,5") from None


def format_rupiah(nilai: float, desimal: int = 0) -> str:
    """100000000 -> 'Rp100.000.000'. Pemisah ribuan titik, desimal koma."""
    teks = f"{abs(nilai):,.{desimal}f}"
    teks = teks.replace(",", "#").replace(".", ",").replace("#", ".")
    tanda = "-" if nilai < 0 else ""
    return f"{tanda}Rp{teks}"


def format_persen(rasio: float) -> str:
    """0.2091 -> '20,9%'."""
    return f"{rasio * 100:.1f}%".replace(".", ",")


def dana_siaga_preset(M: int, porsi: float) -> int:
    """Nominal dana siaga dari porsi preset, dibulatkan ke rupiah."""
    return round(M * porsi)


# Interaksi dengan user.

def tanya(prompt: str, parser: Callable[[str], T], default: T | None = None) -> T:
    """Ulangi pertanyaan sampai parser berhasil. Enter kosong memakai default."""
    while True:
        teks = input(prompt).strip()
        if not teks and default is not None:
            return default
        try:
            return parser(teks)
        except ValueError as error:
            print(f"  ! {error}")


def tanya_modal() -> int:
    def parser(teks: str) -> int:
        M = parse_rupiah(teks)
        if M <= 0:
            raise ValueError("Modal harus lebih dari 0.")
        return M

    return tanya("Total modal (Rp): ", parser)


def tanya_dana_siaga(M: int) -> int:
    print("\nDana siaga di RDPU:")
    for kunci, (nama, porsi) in PRESET_SIAGA.items():
        print(f"  {kunci}. {nama:<12} {format_persen(porsi):>6}  "
              f"({format_rupiah(dana_siaga_preset(M, porsi))})")
    print("  4. Nominal sendiri, mis. 6 x pengeluaran bulanan")

    pilihan = tanya("Pilih [1-4]: ", _parse_pilihan)
    if pilihan in PRESET_SIAGA:
        return dana_siaga_preset(M, PRESET_SIAGA[pilihan][1])

    def parser(teks: str) -> int:
        E = parse_rupiah(teks)
        if E >= M:
            raise ValueError(f"Dana siaga harus kurang dari modal {format_rupiah(M)}.")
        return E

    return tanya("Nominal dana siaga (Rp): ", parser)


def _parse_pilihan(teks: str) -> str:
    if teks not in ("1", "2", "3", "4"):
        raise ValueError("Pilih 1, 2, 3, atau 4.")
    return teks


def tanya_return() -> tuple[float, float, float]:
    print("\nEstimasi return bersih tahunan (setelah pajak & biaya). Enter = default.")
    nama = ("RDPU", "SBN", "Saham")
    hasil = []
    for label, default in zip(nama, RETURN_DEFAULT):
        prompt = f"  {label:<6} [{str(default).replace('.', ',')}%]: "
        hasil.append(tanya(prompt, parse_persen, default / 100))
    return hasil[0], hasil[1], hasil[2]


def tampilkan(hasil: Allocation, M: int) -> None:
    baris = (("RDPU", hasil.x), ("SBN", hasil.y), ("Saham", hasil.z))
    print("\nAlokasi:")
    for label, nilai in baris:
        print(f"  {label:<6} {format_rupiah(nilai):>18}  {format_persen(nilai / M):>6}")
    print(f"  {'Total':<6} {format_rupiah(hasil.x + hasil.y + hasil.z):>18}")

    print(f"\nTarget profit    {format_rupiah(hasil.p_target)}")
    print(f"Estimasi profit  {format_rupiah(hasil.p_aktual, 2)}"
          f"  (selisih {format_rupiah(hasil.selisih, 2)})")

    for pesan in hasil.peringatan:
        print(f"\nPeringatan: {pesan}")
    print("\nCatatan: profit adalah estimasi dari asumsi return, bukan jaminan.")


def main() -> int:
    print("Kalkulator Alokasi Portofolio (RDPU / SBN / Saham)\n")
    try:
        M = tanya_modal()
        E = tanya_dana_siaga(M)

        while True:
            rx, ry, rz = tanya_return()
            try:
                validate_inputs(M, E, rx, ry, rz)
                break
            except PortfolioError as error:
                print(f"  ! {error} Masukkan ulang return.")

        p_min, p_max = feasible_range(M, E, rx, ry, rz)
        print(f"\nTarget yang bisa dicapai: {format_rupiah(p_min)} – {format_rupiah(p_max)} per tahun")

        while True:
            P = tanya("Target profit per tahun (Rp): ", parse_rupiah)
            try:
                hasil = allocate(M, P, E, rx, ry, rz)
                break
            except InfeasibleTargetError as error:
                print(f"  ! Target harus antara {format_rupiah(error.p_min)} "
                      f"dan {format_rupiah(error.p_max)}.")
    except (KeyboardInterrupt, EOFError):
        print("\nDibatalkan.")
        return 130

    tampilkan(hasil, M)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
