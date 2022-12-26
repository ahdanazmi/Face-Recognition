import streamlit as st
import cv2
import numpy as np
import pickle

def run():
    # Fungsi untuk mendeteksi wajah dan mengembalikan embedding wajah
    def detect_and_embed(image):
        # Mendeteksi wajah dengan OpenCV
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(image, 1.3, 5)
        
        # Jika tidak ada wajah yang terdeteksi, kembalikan None
        if len(faces) == 0:
            return None
        # Jika terdeteksi lebih dari satu wajah, kembalikan "more_than_1"
        elif len(faces) > 1:
            return 'more_than_1'
        # Jika terdeteksi satu wajah, kembalikan embedding wajah
        else:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            embedding = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            return embedding

    # Fungsi untuk menyimpan embedding wajah dan nama ke dalam database
    def save_to_database(embedding, name):
    # Baca data dari database jika sudah ada
        try:
            with open('database.pkl', 'rb') as f:
                data = pickle.load(f)
        except:
            data = []
            
        # Tambahkan embedding wajah dan nama ke dalam database
        data.append((embedding, name))
        
        # Simpan database ke dalam file pickle
        with open('database.pkl', 'wb') as f:
            pickle.dump(data, f)


    # Buat endpoint API dengan Streamlit
    st.title('Register')

    # Input gambar wajah dan nama
    with st.form(key= 'form_parameter'):
        image = st.file_uploader('Upload image', type='jpg')
        name = st.text_input('Name')
        submitted = st.form_submit_button('Register')

    # Jika input gambar wajah dan nama sudah diberikan, proses deteksi wajah dan penyimpanan ke database
    if submitted:
        # Baca gambar dan ubah ke dalam format grayscale
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # Mendeteksi wajah dan mengembalikan embedding wajah
        embedding = detect_and_embed(image)
        
        # Jika tidak ada wajah yang terdeteksi, tampilkan pesan error
        if embedding is None:
            st.error('No face detected')
        # Jika wajah terdeteksi, simpan ke database dan tampilkan pesan sukses
        elif embedding == 'more_than_1':
            st.error('More than one face is detected')

        else:
            save_to_database(embedding, name)
            st.success('Your face has been registered')

if __name__ == '__main__':
    run()

