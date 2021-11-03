# Tugas Akhir  
Template repositori untuk mendokumentasikan proyek Tugas Akhir (aka Skripsi), ditujukan untuk S1.

Template di tempat ini hanya bersifat panduan. Di tempat lain ada yang menjadi keharusan, termasuk 
penulisan nama direktori/folder, file, variabel, dan strukturnya.

## Nama Repo
tugas-akhir, contoh: bagustris/tugas-akhir

## Struktur Direktori
- `code`: 
   - berisi coding utama untuk mendapatkan data yang dipakai di buku TA, 
   - coding untuk plot
   - diambil dari hasil terbaik di direktori `exp`
   - hendaknya dibagi per bab berdasarkan buku
- `data`: berisi raw data, baik yang digunakan di `code` atau `exp`. Contoh:  
   - X_si.npy
   - X_sd.npy
   - y_si.npy
   - y_sd.npy
- `fig`: berisi gambar
- `book`: berisi file buku TA: Latex, ms word, atau LibreOffice Writer, Google Doc, dll.
- `exp`: direktori **UTAMA** yang berisi experimen berdasarkan waktu, contoh:  
   - 2021:
      - oktober:  
         - klasikasi_unbalance_normal.py  
         - klasifikasi_4_kelas.py  
      - november
      - desember
   - 2022:  
      - januari
      - februari
      - maret  
      - april 
      - mei   
      - juni
      - juli
- `note`:  
  Berisi catatan dan lain-lain:  
  - diskusi dengan dosen pembimbing
  - pertanyan dan jawaban
  - apa yang belum paham
  - apa yang ingin dipelajari
  - permasalahan
  - temuan
  - dll (*like research diary atau journal*)
- README.md: berisi panduan untuk mereplikasi TA, kontak, promosi, dll.
