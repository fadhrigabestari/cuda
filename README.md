# CUDA Radix Sort

Dibuat oleh :
 - Yuly Haruka Berliana Gunawan - 13516031
 - M. Fadhriga Bestari - 13516154

## Petunjuk Penggunaan Program
Run this in your terminal or on the server
```
cd cuda/src
make
./radix_sort <number_threads>
```

## Pembagian Tugas
- Yuly Haruka Berliana Gunawan - 13516031
	- Parallel the radix sort using CUDA
	- Main Program
- M. Fadhriga Bestari - 13516154
	- Parallel the radix sort using CUDA
	- Main Program

## Laporan Pengerjaan
### Deskripsi Solusi Paralel
Solusi paralel yang kami implementasikan menggunakan CUDA ialah dengan memparalelkan setiap *looping for* yang ada pada fungsi *generateFlag* dan *generateShouldIndex*. 

### Analisis Solusi
Menurut kami, penerapan CUDA pada radix sort dapat mempercepat sort karena dengan dilakukannya pembagian proses pada setiap task dari core processor, proses pengisian array flag dan proses pencarian index yang dilakukan dalam radix sort dipercepat. Tetapi perlu dilakukan peng-copyan data dari CPU ke GPU dan dari GPU ke CPU karena data yang ada pada CPU dan GPU tidak dapat digunakan pada dua *device* secara langsung.  

### Pemetaan *Thread* 
Jumlah *thread* per blok yang kami gunakan adalah 256 karena CUDA GPU menjalankan kernel dengan menggunakan jumlah blocks of thread yang kelipatan 32, maka dengan itu angka 256 merupakan angka yang masuk akal untuk digunakan karena angka 256 tidak terlalu kecil dan tidak terlalu besar. 

### Pengukuran Kinerja
> N = 5000
<br>
Radix Sort serial : 14803
<br>
Radix Sort Paralel dengan CUDA : 156699

> N = 50.000
<br>
Radix Sort serial : 116493
<br>
Radix Sort Paralel dengan CUDA : 248837

> N = 100.000
<br>
Radix Sort serial : 232686
<br>
Radix Sort Paralel dengan CUDA : 330071

> N = 200.000
<br>
Radix Sort serial : 483184
<br>
Radix Sort Paralel dengan CUDA : 535068

> N = 400.000
<br>
Radix Sort serial : 929879
<br>
Radix Sort Paralel dengan CUDA : 957559

### Analisis Perbandingan Kinerja Serial dan Paralel
Berdasarkan percobaan dan perbandingan yang telah kami lakukan, didapatkan bahwa program paralel yang kami gunakan MPI_Scatter dan MPI_Gather membutuhkan waktu tambahan yaitu waktu untuk melakukan pembagian dan pengumpulan data dari ke empat task yang kami assign. Sehingga program paralel yang kami buat, akan lebih lambat pada n yang relatif kecil. Tetapi pada percobaan dengan n = 1.000.000, program paralel radix sort yang kami buat, memproses radix sort lebih cepat.