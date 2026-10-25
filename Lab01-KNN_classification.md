# Mission-06C — KNN: Machine Condition Classification on ESP32

> **Mata kuliah:** Machine Learning · **Platform:** ESP32 + MicroPython · **Jenis:** klasifikasi biner · **Perangkat tambahan:** tidak ada  
> **Estimasi waktu:** 80–100 menit · **Prasyarat:** konsep train/test split, KNN, normalisasi, accuracy, confusion matrix, menjalankan MicroPython di Thonny.

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan mengapa KNN dapat berjalan di ESP32 tanpa Scikit-learn dan tanpa fase pelatihan parameter eksplisit.
2. Mengimplementasikan perhitungan jarak, pemilihan **K tetangga terdekat**, dan *majority voting* dengan MicroPython.
3. Menguji pengaruh nilai **K** dan skala fitur pada hasil klasifikasi.
4. Mengevaluasi model dengan **test set terpisah**, accuracy, dan confusion matrix.
5. Mengukur perkiraan *inference latency* dan mengidentifikasi keterbatasan memori perangkat.

## 2. Studi kasus

Bayangkan sebuah mesin industri dipantau berdasarkan dua fitur:

- **Temperature** ($x_1$): temperatur mesin dalam derajat Celsius (°C).
- **Vibration** ($x_2$): tingkat getaran mesin dalam mm/s.

Target klasifikasi:

- **0 = NORMAL**
- **1 = WARNING**

**Penting:** Semua data adalah **sintetik** untuk pembelajaran. Tidak ada sensor yang dihubungkan ke ESP32. Label bukan standar diagnosis kondisi mesin nyata; jangan gunakan model ini untuk keselamatan mesin.

**Alur kerja:** `Data sintetik → penyimpanan di ESP32 → scaling → KNN → prediksi → evaluasi`.

## 3. Alat dan persiapan

- ESP32 yang telah di-*flash* dengan firmware MicroPython.
- Kabel USB data dan komputer dengan Thonny IDE.
- Tidak perlu breadboard, sensor, LED eksternal, koneksi internet, atau paket ML.

Di Thonny, pilih interpreter **MicroPython (ESP32)** dan port serial ESP32. Buat file `main.py` **di perangkat ESP32**. Program akan mencetak hasil di Shell Thonny. Jika sudah ada `main.py` dari praktikum blink, salin/ubah namanya terlebih dahulu agar tidak berjalan bersamaan.

## 4. Ringkasan teori

KNN melakukan klasifikasi dengan mencari **K contoh training yang paling dekat** dengan input baru dan memilih label yang paling banyak muncul di antara tetangga tersebut.

Dengan dua fitur, jarak Euclidean adalah:

$$
d = \sqrt{(x_1-a_1)^2 + (x_2-a_2)^2}
$$

Untuk pengurutan tetangga, akar tidak diperlukan sehingga kita menghitung **jarak kuadrat**:

$$
d^2 = (x_1-a_1)^2 + (x_2-a_2)^2
$$

Keterangan:

- $x_1$, $x_2$: fitur input yang akan diklasifikasikan.
- $a_1$, $a_2$: fitur satu sampel di data training.
- $d$: jarak Euclidean antara kedua sampel.
- $K$: jumlah tetangga yang ikut *voting*.

Karena temperatur dan getaran memiliki rentang angka berbeda, kita gunakan **min–max scaling berdasarkan data training**:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

> Gunakan rentang minimum dan maksimum **hanya dari training set**. Data test dan input baru menggunakan parameter scaling yang sama. Jangan menyesuaikan scaler menggunakan test set. Nilai data baru yang berada di luar rentang training dapat menghasilkan nilai scaling di luar $[0,1]$; hal ini tidak otomatis salah.

**Catatan konsep:** KNN menyimpan sampel referensi saat disebut “training”. Beban utamanya terjadi ketika **inference**, karena harus menghitung jarak ke sampel referensi.

## 5. Dataset

Modul menyediakan **36 sampel training** dan **12 sampel test** dalam kode. Label dibuat dengan aturan simulasi yang memiliki area batas kelas sehingga prediksi tidak selalu 100% benar. Data test **tidak** digunakan sebagai tetangga saat inference.

Format setiap baris:

```python
(temperature, vibration, label)
```

Contoh: `(68.0, 6.1, 1)` berarti temperatur 68°C, getaran 6,1 mm/s, label WARNING.

## 6. Program utama MicroPython

Simpan kode ini sebagai `main.py` pada ESP32, lalu klik **Run**.

```python
# Mission-06D - KNN pada ESP32 (MicroPython)
import time
import gc

# temperature (C), vibration (mm/s), label (0=NORMAL; 1=WARNING)
TRAIN = [
    (29.0, 1.0, 0),
    (42.4, 2.7, 0),
    (55.8, 4.4, 0),
    (68.0, 6.1, 1),
    (31.4, 7.8, 0),
    (44.8, 1.2, 0),
    (57.0, 2.9, 0),
    (70.4, 4.6, 1),
    (33.8, 6.3, 0),
    (46.0, 8.0, 1),
    (59.4, 1.4, 0),
    (72.8, 3.1, 1),
    (35.0, 4.8, 0),
    (48.4, 6.5, 1),
    (61.8, 8.2, 1),
    (74.0, 1.6, 0),
    (37.4, 3.3, 0),
    (50.8, 5.0, 0),
    (63.0, 6.7, 1),
    (76.4, 8.4, 1),
    (39.8, 1.8, 0),
    (52.0, 3.5, 0),
    (65.4, 5.2, 1),
    (78.8, 6.9, 1),
    (41.0, 8.6, 1),
    (54.4, 2.0, 0),
    (67.8, 3.7, 1),
    (30.0, 5.4, 0),
    (43.4, 7.1, 0),
    (56.8, 8.8, 1),
    (69.0, 2.2, 0),
    (32.4, 3.9, 0),
    (45.8, 5.6, 0),
    (58.0, 7.3, 1),
    (71.4, 9.0, 1),
    (34.8, 2.4, 0),
]
TEST = [
    (32, 1.6, 0),
    (39, 3.1, 0),
    (52, 3.0, 0),
    (60, 4.5, 1),
    (47, 6.5, 1),
    (70, 7.5, 1),
    (78, 8.0, 1),
    (35, 7.9, 0),
    (73, 2.5, 1),
    (57, 5.5, 1),
    (46, 4.4, 0),
    (65, 6.2, 1),
]
LABELS = ("NORMAL", "WARNING")

# Parameter scaling dihitung HANYA dari data training.
MIN_T = min(r[0] for r in TRAIN)
MAX_T = max(r[0] for r in TRAIN)
MIN_V = min(r[1] for r in TRAIN)
MAX_V = max(r[1] for r in TRAIN)
RANGE_T = MAX_T - MIN_T
RANGE_V = MAX_V - MIN_V


def predict_knn(temp, vibration, k=3, scaled=True):
    if not isinstance(k, int) or k < 1 or k > len(TRAIN) or k % 2 == 0:
        raise ValueError("Gunakan K ganjil: 1 sampai jumlah data training")
    distances = []
    for idx, (t, v, label) in enumerate(TRAIN):
        dt = temp - t
        dv = vibration - v
        if scaled:
            dt = dt / RANGE_T
            dv = dv / RANGE_V
        distance_sq = dt * dt + dv * dv
        distances.append((distance_sq, idx, label))

    # Tie jarak dipecahkan secara deterministik memakai urutan data.
    distances.sort(key=lambda r: (r[0], r[1]))
    neighbors = distances[:k]
    warning_votes = sum(r[2] for r in neighbors)
    prediction = 1 if warning_votes > k // 2 else 0
    return prediction, neighbors


def evaluate(k=3, scaled=True, verbose=False):
    correct = 0
    tp = tn = fp = fn = 0
    for temp, vib, actual in TEST:
        predicted, _ = predict_knn(temp, vib, k, scaled)
        correct += predicted == actual
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        else:
            fn += 1
        if verbose:
            print("T=%.1f, V=%.1f | actual=%s predicted=%s" % (
                temp, vib, LABELS[actual], LABELS[predicted]))
    accuracy = 100 * correct / len(TEST)
    print("K=%d, scaled=%s | accuracy=%.2f%% (%d/%d) | TP=%d TN=%d FP=%d FN=%d" % (
        k, scaled, accuracy, correct, len(TEST), tp, tn, fp, fn))
    return accuracy


def demo():
    print("TRAIN =", len(TRAIN), "TEST =", len(TEST))
    print("Rentang training: T=%.1f..%.1f; V=%.1f..%.1f" % (
        MIN_T, MAX_T, MIN_V, MAX_V))
    for k in (1, 3, 5, 7):
        evaluate(k)
    print("\nPengaruh scaling untuk K=3")
    evaluate(3, scaled=False)
    evaluate(3, scaled=True)


def interactive():
    print("Ketik q untuk keluar. Input sintetik, BUKAN pembacaan sensor.")
    while True:
        raw = input("Temperature [C] atau q: ").strip()
        if raw.lower() == "q":
            return
        try:
            temp = float(raw)
            vib = float(input("Vibration [mm/s]: ").strip())
            if not (20 <= temp <= 100 and 0 <= vib <= 12):
                print("Input di luar rentang simulasi (T:20..100; V:0..12)")
                continue
            k = int(input("K ganjil [1/3/5/7]: ").strip())
            pred, neighbors = predict_knn(temp, vib, k)
            print("Prediction:", LABELS[pred])
            print("Nearest neighbors (jarak kuadrat, indeks, kelas):")
            for d, idx, label in neighbors:
                print("  %.5f | %d | %s" % (d, idx, LABELS[label]))
        except (ValueError, TypeError) as e:
            print("Input tidak valid:", e)


def benchmark(repetitions=100):
    # Menghindari waktu input/print dalam timing; tanpa dataset generator.
    samples = [(39, 3.1), (57, 5.5), (70, 7.5), (47, 6.5)]
    gc.collect()
    start_free = gc.mem_free()
    for temp, vib in samples:  # warm-up
        predict_knn(temp, vib, 3)
    elapsed = []
    for j in range(repetitions):
        temp, vib = samples[j % len(samples)]
        start = time.ticks_us()
        predict_knn(temp, vib, 3)
        elapsed.append(time.ticks_diff(time.ticks_us(), start))
    elapsed.sort()
    print("Inference median:", elapsed[len(elapsed) // 2], "us")
    print("Free memory before/after:", start_free, gc.mem_free(), "bytes")
    print("Note: these are approximate MicroPython-level measurements")


if __name__ == "__main__":
    demo()
    # Untuk eksperimen lain, panggil interactive() atau benchmark()
```

### Apa yang dilakukan program?

1. `TRAIN` disimpan sebagai data referensi KNN; `TEST` disimpan terpisah untuk evaluasi.
2. `predict_knn()` menghitung jarak kuadrat ke setiap data training, mengurutkan jarak, dan melakukan voting.
3. `evaluate()` membandingkan prediksi dengan label test sehingga menghasilkan accuracy dan confusion matrix.
4. `demo()` menjalankan eksperimen awal secara otomatis. `interactive()` dan `benchmark()` dipanggil saat dibutuhkan.

## 7. Eksperimen A — KNN berjalan di ESP32 (15 menit)

Jalankan `main.py`. Amati keluaran `TRAIN`, `TEST`, dan accuracy untuk K = 1, 3, 5, 7.

**Target pemeriksaan dengan dataset dan kode yang tidak diubah:**

| K | Jumlah benar | Accuracy |
|---:|---:|---:|
| 1 | 10 dari 12 | 83,33% |
| 3 | 9 dari 12 | 75,00% |
| 5 | 10 dari 12 | 83,33% |
| 7 | 9 dari 12 | 75,00% |

Angka di atas merupakan hasil acuan pada Python untuk dataset ini; output MicroPython seharusnya sesuai selama logikanya sama. **K lebih besar tidak otomatis memberi akurasi lebih tinggi.**

## 8. Eksperimen B — Input interaktif melalui Thonny (15 menit)

Di bagian akhir program, ubah:

```python
if __name__ == "__main__":
    demo()
```

menjadi:

```python
if __name__ == "__main__":
    interactive()
```

Jalankan, lalu masukkan temperatur, getaran, dan K melalui Shell Thonny. Coba input berikut:

| Temperatur (°C) | Getaran (mm/s) | K |
|---:|---:|---:|
| 32 | 1.6 | 3 |
| 70 | 7.5 | 3 |
| 57 | 5.5 | 3 |

Perhatikan tiga tetangga terdekat. Tuliskan alasan prediksi berdasarkan **voting**, bukan asumsi ambang temperatur atau getaran.


***jawab*** 

Prediksi ditentukan berdasarkan majority voting (suara terbanyak) dari $K$ tetangga terdekat di ruang fitur, bukan berdasarkan asumsi ambang batas yang kaku, sehingga kelas diputuskan murni dari label terbanyak di antara data terdekat.

---

## 9. Eksperimen C — Pengaruh K dan normalisasi (20 menit)

Pertama, panggil `demo()` lagi dan amati output `scaled=False` dibandingkan `scaled=True`. Kedua, untuk satu input yang sama, uji K = 1, 3, 5, dan 7 melalui `interactive()`.

**Pertanyaan investigasi:** Mengapa fitur temperatur bisa terlalu dominan saat jarak dihitung tanpa scaling? Apakah seluruh input selalu berubah kelas sesudah scaling?

Untuk memperdalam, catat minimal satu contoh yang prediksinya berubah ketika K atau status scaling diubah. Jika tidak menemukan perubahan pada input awal, coba nilai yang lebih dekat dengan batas kelas seperti temperatur 45–65°C dan getaran 4–7 mm/s.

***jawab:***

Mengapa fitur temperatur bisa terlalu dominan saat jarak dihitung tanpa scaling? Tanpa scaling, fitur temperatur mendominasi perhitungan jarak Euclidean karena memiliki rentang nilai angka yang jauh lebih besar dibanding getaran.

Apakah seluruh input selalu berubah kelas sesudah scaling? 

| K | Prediksi | vote |
|---:|---:|---:|
| 1 | NORMAL | 1 NORMAL vs 0 WARNING |
| 3 | NORMAL | 2 NORMAL vs 1 WARNING |
| 5 | WARNING | 2 NORMAL vs 3 WARNING |
| 7 | WARNING | 3 NORMAL vs 4 WARNING |

Pada input di titik perbatasan (Temp = 57 °C, Vib = 5.5 mm/s), perubahan nilai $K$ terbukti dapat mengubah hasil prediksi. Pada $K = 1$ dan $K = 3$, model memprediksi NORMAL. Namun ketika $K$ dinaikkan menjadi $5$ dan $7$, prediksi berubah menjadi WARNING karena mencakup lebih banyak sampel tetangga berlabel WARNING di sekitar titik tersebut. 


## 10. Eksperimen D — Evaluasi pada data yang belum pernah dilihat (15 menit)

Jalankan:

```python
evaluate(k=3, scaled=True, verbose=True)
```

Perhatikan confusion matrix biner yang menggunakan **WARNING sebagai kelas positif**:

| | Prediksi WARNING (1) | Prediksi NORMAL (0) |
|---|---|---|
| Aktual WARNING (1) | TP | FN |
| Aktual NORMAL (0) | FP | TN |

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Bandingkan hasil accuracy dengan angka acuan pada Eksperimen A. Jelaskan mengapa hasil evaluasi tidak boleh hanya dihitung pada `TRAIN`.

***jawab:***

Bandingkan hasil accuracy dengan angka acuan pada Eksperimen A : Hasil akurasi pengujian pada Eksperimen D dengan $K = 3$ dan scaled = True adalah sebesar 75,00% (9 dari 12 data benar). Nilai ini identik dan sesuai dengan angka acuan pada Eksperimen A untuk konfigurasi parameter yang sama ($K = 3$, scaled = True). tentu saja dengan tujuan yang berbeda Eksperimen A melihat perbandingan tingkat akurasi secara keseluruhan dari berbagai variasi nilai $K$ ($1, 3, 5, 7$) sedangkan eksperimen D berfokus pada analisis mendalam menggunakan Confusion Matrix ($TP, TN, FP, FN$) untuk melihat secara rinci jenis kesalahan prediksi model (mana data yang benar-benar salah atau benar).  

Jelaskan mengapa hasil evaluasi tidak boleh hanya dihitung pada TRAIN : Evaluasi pada data TRAIN hanya mengukur seberapa baik model mengingat data yang sudah dipelajari (overfitting). Menguji model dengan data TEST terpisah bertujuan untuk mengukur kemampuan generalisasi model KNN dalam memprediksi kondisi data baru secara obyektif di dunia nyata.
## 11. Eksperimen E — Waktu inference pada ESP32 (15 menit)

Panggil `benchmark(100)` dari Shell Thonny atau ubah pemanggilan pada bagian `__main__`. Fungsi ini:

- Menjalankan pemanasan ringan sebelum pencatatan waktu.
- Mengukur 100 kali waktu eksekusi `predict_knn()` dengan `time.ticks_us()`.
- Melaporkan **median** waktu inference dan memori bebas (`gc.mem_free()`).

**Penting tentang metrik:** median waktu adalah perkiraan dari pengukuran MicroPython, bukan latency real-time deterministik. Memori bebas sebelum/sesudah benchmark **bukan** ukuran persis total konsumsi RAM model; alokasi sementara dan *garbage collection* dapat memengaruhinya. Untuk membandingkan ukuran dataset, gunakan prosedur yang sama dan simpan jumlah titik, latency median, serta memori bebas secara konsisten.

### Tantangan lanjutan (opsional)

Salin beberapa baris `TRAIN` untuk membuat dataset referensi lebih besar, misalnya 72 atau 144 sampel. Ulangi benchmark dengan input dan jumlah repetisi yang sama. Ingat, duplikasi data dapat memengaruhi voting dan **bukan** penambahan informasi baru sehingga jangan menafsirkan peningkatan accuracy sebagai akibat data lebih bermutu.

## 12. Lembar hasil mahasiswa

| Variasi | K | Scaling | Accuracy (%) | Median inference (µs) | Catatan |
|---|---:|---|---:|---:|---|
| A | 1 | Ya | 83,33% | 75,00% | TP=5, TN=5, FP=0, FN=2 |
| B | 3 | Ya | 75,00% | 75,00% | TP=4, TN=5, FP=0, FN=3 |
| C | 5 | Ya | 83,33% | 75,00% | TP=5, TN=5, FP=0, FN=2 |
| D | 7 | Ya | 75,00% | 75,00% | TP=4, TN=5, FP=0, FN=3 |
| E | 3 | Tidak | 83,33% | 75,00% | Tanpa scaling (scaled=False) |

## 13. Pertanyaan analisis

1. Mengapa KNN tidak memerlukan proses pembaruan koefisien seperti regresi linear, tetapi tetap membutuhkan data training?
2. Mengapa normalisasi dapat mengubah prediksi KNN? Jelaskan menggunakan skala temperatur dan getaran.
3. Apakah K yang lebih besar selalu menghasilkan accuracy lebih tinggi? Gunakan hasil percobaan.
4. Apa perbedaan waktu inference dan waktu input melalui terminal serial? Mengapa *timing* tidak dilakukan saat `input()` atau `print()`?
5. Jika data training diperbesar 10 kali, apa yang kemungkinan terjadi pada penggunaan memori dan waktu inference di ESP32? Apa yang perlu diukur untuk membuktikannya?


***jawab***

1.	KNN adalah algoritma berbasis instance (lazy learner) yang tidak membangun model matematika parametrik (tidak mencari koefisien bobot/slope seperti OLS). Meskipun demikian, data training tetap mutlak dibutuhkan karena seluruh data tersebut disimpan mentah-mentah di dalam memori untuk digunakan sebagai titik acuan/referensi saat menghitung jarak ke data baru pada saat inference.



2.	Tanpa normalisasi (scaling), rentang nilai fitur temperatur (misal 30–80 °C) jauh lebih besar dibandingkan getaran (misal 1–9 mm/s). Dalam rumus jarak Euclidean, selisih angka pada temperatur akan mendominasi total kuadrat jarak secara masif, sehingga fitur getaran seolah-olah tidak dianggap oleh model. Normalisasi diperlukan agar kedua fitur memiliki rentang skala yang setara (misalnya antara 0 sampai 1), sehingga kontribusi jarak dari temperatur dan getaran menjadi adil dan seimbang.

3. Tidak selalu. Berdasarkan hasil percobaan yang telah dilakukan, nilai K yang lebih besar tidak menjamin akurasi yang lebih tinggi. Akurasi sangat bergantung pada sebaran data dan struktur batas antar kelas nilai K yang terlalu besar justru dapat mengaburkan batas keputusan (boundary smoothing).

4. •	Waktu inference adalah durasi waktu murni yang dibutuhkan mikrokontroler untuk melakukan komputasi algoritma (menghitung jarak Euclidean, mengurutkan, dan melakukan majority voting).

   • 	Waktu input serial melibatkan proses komunikasi perangkat keras, kecepatan transmisi (baud rate), dan jeda waktu manusia mengetik di terminal.

   •	Timing tidak boleh dilakukan pada input() atau print() karena hal tersebut hanya mengukur kecepatan respons serial dan waktu ketik manusia, bukan performa kecepatan komputasi algoritma yang objektif.

5.	Jika data training diperbesar 10 kali, penggunaan memori RAM di ESP32 akan meningkat secara linier karena algoritma KNN menyimpan seluruh data latih mentah-mentah di dalam memori, dan waktu inference akan menjadi jauh lebih lambat akibat bertambahnya jumlah perhitungan jarak Euclidean yang harus diproses terhadap seluruh titik referensi (O(N)). Untuk membuktikannya secara empiris, hal yang perlu diukur adalah kapasitas memori bebas menggunakan fungsi gc.mem_free() sebelum dan sesudah data diperbesar, serta waktu eksekusi fungsi prediksi menggunakan pewaktu berbasis mikrodetik seperti time.ticks_us().

   
## 14. Kesimpulan dan tugas

**Kesimpulan:** KNN dapat ditulis dengan MicroPython dan dijalankan seluruhnya pada ESP32 tanpa sensor atau library machine learning. Parameter preprocessing dan pemisahan data training/test tetap penting meskipun perangkat sangat sederhana.

**Pengumpulan:** serahkan `main.py`, foto/screenshot output Shell, tabel eksperimen, dan jawaban pertanyaan analisis. Jangan menyerahkan accuracy training sebagai pengganti accuracy test.

---

**Catatan instruktur:** seluruh hasil acuan diuji pada Python standar dengan algoritma identik; performa waktu/memori perlu diukur pada board mahasiswa. Tidak ada hardware sensor maupun koneksi cloud yang terlibat.
