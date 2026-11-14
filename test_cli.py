"""Test cli.py: parsing, format, dan satu sesi lengkap dengan input tersimulasi."""

import pytest

import cli


@pytest.mark.parametrize(
    "teks, hasil",
    [
        ("100000000", 100_000_000),
        ("100.000.000", 100_000_000),
        ("100_000_000", 100_000_000),
        ("Rp 100.000.000", 100_000_000),
        ("rp100.000.000", 100_000_000),
        ("0", 0),
    ],
)
def test_parse_rupiah(teks, hasil):
    assert cli.parse_rupiah(teks) == hasil


@pytest.mark.parametrize("teks", ["", "abc", "-5", "1,5", "12jt"])
def test_parse_rupiah_tidak_valid(teks):
    with pytest.raises(ValueError):
        cli.parse_rupiah(teks)


@pytest.mark.parametrize(
    "teks, hasil",
    [("6,5", 0.065), ("6.5", 0.065), ("6,5%", 0.065), ("12", 0.12), ("0", 0.0)],
)
def test_parse_persen(teks, hasil):
    assert cli.parse_persen(teks) == pytest.approx(hasil)


def test_parse_persen_tidak_valid():
    with pytest.raises(ValueError):
        cli.parse_persen("enam")


@pytest.mark.parametrize(
    "nilai, desimal, hasil",
    [
        (100_000_000, 0, "Rp100.000.000"),
        (8_999_999.934, 2, "Rp8.999.999,93"),
        (-0.065, 2, "-Rp0,07"),
        (0, 0, "Rp0"),
    ],
)
def test_format_rupiah(nilai, desimal, hasil):
    assert cli.format_rupiah(nilai, desimal) == hasil


def test_format_persen():
    assert cli.format_persen(0.2091) == "20,9%"


def test_dana_siaga_preset():
    assert cli.dana_siaga_preset(100_000_000, 0.20) == 20_000_000


def _jalankan(monkeypatch, capsys, jawaban):
    antrean = iter(jawaban)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(antrean))
    kode = cli.main()
    return kode, capsys.readouterr().out


def test_sesi_lengkap_contoh_spesifikasi(monkeypatch, capsys):
    # Modal, preset Moderat, tiga return default, target di luar rentang, target valid.
    kode, out = _jalankan(
        monkeypatch, capsys,
        ["100.000.000", "2", "", "", "", "12.000.000", "9.000.000"],
    )
    assert kode == 0
    assert "Rp6.200.000 – Rp10.600.000" in out
    assert "Target harus antara" in out
    for nominal in ("Rp20.000.001", "Rp29.090.909", "Rp50.909.090", "Rp100.000.000"):
        assert nominal in out
    assert "bukan jaminan" in out


def test_sesi_nominal_dan_input_salah(monkeypatch, capsys):
    # Modal 0 ditolak, pilihan 9 ditolak, dana siaga >= modal ditolak.
    kode, out = _jalankan(
        monkeypatch, capsys,
        ["0", "50.000.000", "9", "4", "60.000.000", "10.000.000",
         "", "", "", "4.000.000"],
    )
    assert kode == 0
    assert "Modal harus lebih dari 0." in out
    assert "Pilih 1, 2, 3, atau 4." in out
    assert "Dana siaga harus kurang dari modal" in out
    assert "Rp10.000.000" in out


def test_return_terlalu_mirip_diminta_ulang(monkeypatch, capsys):
    kode, out = _jalankan(
        monkeypatch, capsys,
        ["100.000.000", "2", "5", "8", "8", "", "", "", "9.000.000"],
    )
    assert kode == 0
    assert "terlalu mirip" in out


def test_ry_lebih_besar_dari_rz_tampil_peringatan(monkeypatch, capsys):
    kode, out = _jalankan(
        monkeypatch, capsys,
        ["100.000.000", "2", "5", "7", "6", "6.000.000"],
    )
    assert kode == 0
    assert "Peringatan:" in out


def test_ctrl_c_keluar_rapi(monkeypatch, capsys):
    def interupsi(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interupsi)
    assert cli.main() == 130
    assert "Dibatalkan." in capsys.readouterr().out
