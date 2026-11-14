# Kalkulator Alokasi Portofolio

Hitung pembagian modal ke RDPU, SBN, dan saham berdasarkan target profit tahunan.

Program ini menjawab pertanyaan seperti ini: modal saya 100 juta, 20 juta harus tetap jadi dana siaga, dan saya ingin profit 9 juta per tahun. Berapa yang perlu saya taruh di setiap instrumen?

Setiap syarat menjadi satu persamaan linear, jadi pertanyaannya berubah menjadi Sistem Persamaan Linear Tiga Variabel (SPLTV). Solver-nya ditulis manual dengan Gaussian elimination, lalu diverifikasi terhadap `numpy` di unit test.

## Demo

```
Total modal (Rp): 100.000.000

Dana siaga di RDPU:
  1. Konservatif   40,0%  (Rp40.000.000)
  2. Moderat       20,0%  (Rp20.000.000)
  3. Agresif       10,0%  (Rp10.000.000)
  4. Nominal sendiri, mis. 6 x pengeluaran bulanan
Pilih [1-4]: 2

Estimasi return bersih tahunan (setelah pajak & biaya). Enter = default.
  RDPU   [5,0%]:
  SBN    [6,5%]:
  Saham  [12,0%]:

Target yang bisa dicapai: Rp6.200.000 – Rp10.600.000 per tahun
Target profit per tahun (Rp): 9.000.000

Alokasi:
  RDPU         Rp20.000.001   20,0%
  SBN          Rp29.090.909   29,1%
  Saham        Rp50.909.090   50,9%
  Total       Rp100.000.000

Target profit    Rp9.000.000
Estimasi profit  Rp8.999.999,93  (selisih -Rp0,07)

Catatan: profit adalah estimasi dari asumsi return, bukan jaminan.
```

Coba naikkan target-nya. Porsi saham ikut membesar karena saham punya return tertinggi. Target lebih tinggi berarti risiko lebih besar.

## Model Matematika

| Variabel | Arti |
|---|---|
| $x$ | dana di Reksa Dana Pasar Uang (RDPU) |
| $y$ | dana di Obligasi Negara (SBN) |
| $z$ | dana di saham |

$$
\begin{bmatrix} 1 & 1 & 1 \\ r_x & r_y & r_z \\ 1 & 0 & 0 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix} =
\begin{bmatrix} M \\ P \\ E \end{bmatrix}
$$

Baris pertama memastikan seluruh modal $M$ terbagi habis. Baris kedua memastikan total profit sama dengan target $P$. Baris ketiga mengunci dana siaga $E$ di RDPU. $r_x, r_y, r_z$ adalah estimasi return bersih tahunan setiap instrumen.

Determinant-nya $\det(A) = r_z - r_y$. Artinya, sistem baru menjadi singular kalau return SBN sama dengan return saham.

Target $P$ hanya menghasilkan alokasi tanpa nilai negatif kalau berada di feasible range ini:

$$
r_x E + (M - E)\min(r_y, r_z) \;\le\; P \;\le\; r_x E + (M - E)\max(r_y, r_z)
$$

Program menghitung range ini sebelum meminta target, lalu menolak target di luar range.

## Installation

Butuh Python 3.10 atau lebih baru.

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python cli.py
```

Input rupiah boleh ditulis `100.000.000`, `Rp 100.000.000`, atau `100000000`. Persen boleh ditulis `6,5` atau `6.5`. Tekan Enter untuk memakai return default, dan Ctrl+C untuk keluar.

## Testing

```bash
pytest              # semua test
pytest -k gauss     # hanya test Gaussian elimination
```

`tests/test_solver.py` membandingkan `det3` dan `gauss_solve` dengan `numpy` pada matriks acak, menguji target yang tepat berada di batas range, dan memastikan total alokasi selalu sama dengan modal untuk 200 input acak. `tests/test_cli.py` menjalankan sesi CLI lengkap dengan input yang disimulasikan.

## Project Structure

```
├── solver.py        # SPLTV: det3, Gaussian elimination, validasi, pembulatan
├── cli.py           # command-line interface, tanpa perhitungan sendiri
├── tests/
│   ├── test_solver.py
│   └── test_cli.py
├── requirements.txt
└── pytest.ini
```

`solver.py` tidak memakai `print`, `input`, maupun `numpy`. Semua perhitungan ada di sana, sehingga interface lain bisa memakai solver yang sama tanpa mengubah logikanya.

## Limitations

Hasilnya hanya sebaik asumsi return yang kamu masukkan. Return saham bisa jauh di bawah estimasi, bahkan negatif. Pakai program ini untuk simulasi dan memahami trade-off antara target dan risiko, bukan sebagai saran investasi.

Alokasi dibulatkan ke rupiah utuh. Program belum memperhitungkan satuan pembelian seperti lot saham atau kelipatan minimum SBN.

## Roadmap

- Web app dengan Streamlit: slider target dan pie chart alokasi
- Scenario testing untuk return saham (pesimis, dasar, optimis)
- Input return kotor ditambah tarif pajak dan biaya
- Pembulatan ke satuan pembelian setiap instrumen
- Instrumen keempat, memanfaatkan `gauss_solve` yang sudah berlaku untuk matriks n×n
