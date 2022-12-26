# Face-Recognition

## Tentang Repositori

Repositori ini berisi proyek face recognition untuk memprediksi apakah wajah seseorang terdata pada database.
Repositori ini terdiri dari:
- `Face Recognition.ipynb` yang berisi report f1-score dan accuracy
- Folder `foto`. Folder tersebut berisi 2 folder. Folder `registered` yang merupakan foto dari 25 public figur dari Indonesia, Jepang, Thailand, Cina, Vietnam, dan India yang telah terdaftar dan masuk dalam database. Folder `test` merupakan foto lainnya dari 25 public figur tersebut yang digunakan untuk mengetest akurasi program.
- Folder `app` berisi file python dan docker

## Tentang APP
App dibuat dengan Python dengan framework Streamlit. Tediri dari 2 page yakni register untuk mendaftarkan foto ke dalam database dan recognition untuk mengecek apakah wajah seseorang telah terdaftar atau belum.

## Requirements
`numpy`

`opencv-python==4.6.0.66` 

`scikit-learn==1.1.1` 

`streamlit==1.11.1`
