# Import library yang dibutuhkan
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
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

    # Fungsi untuk mengecek kesamaan wajah dengan wajah di database
    def recognize_face(embedding, database):
        # Inisialisasi nama dan nilai similarity terkecil
        name = 'Your face is not registered'
        max_similarity = -1.0
        
        # Loop setiap data di database
        for data in database:
            # Ambil embedding wajah dan nama dari data
            stored_embedding, stored_name = data
            
            # Hitung similarity antara embedding wajah terdeteksi dengan embedding wajah di database
            similarity = cosine_similarity(embedding.reshape(1, -1), stored_embedding.reshape(1, -1))[0][0]
            
            # Jika similarity lebih besar dari nilai terbesar sebelumnya, update nama dan nilai terbesar
            if similarity > max_similarity:
                max_similarity = similarity
                name = stored_name
    
        # Kembalikan nama yang terdeteksi
        return name

    # Baca database wajah dari file pickle
    with open('database.pkl', 'rb') as file:
        database = pickle.load(file)

    # Buat endpoint API dengan Streamlit
    st.title('Recognition')

    # Input gambar wajah
    with st.form(key= 'form_parameter'):
        image = st.file_uploader('Upload image', type='jpg')
        submitted = st.form_submit_button('Check')

    # Jika input gambar wajah sudah diberikan, proses deteksi wajah dan pengenalan wajah
    if submitted:
        # Baca gambar dan ubah ke dalam format BGR (default OpenCV)
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Mendeteksi wajah dan mengembalikan embedding wajah
        embedding = detect_and_embed(image)
        
        # Jika tidak ada wajah yang terdeteksi, tampilkan pesan error
        if embedding is None:
            st.error('No face detected')
        # Jika terdeteksi lebih dari satu wajah, tampilkan pesan error
        elif embedding == "more_than_1":
            st.error('More than one face is detected')
        # Jika terdeteksi satu wajah, lakukan pengenalan wajah
        else:
            # Cek kesamaan wajah dengan wajah di database
            name = recognize_face(embedding, database)
            # Tampilkan nama yang terdeteksi
            st.success(name)

if __name__ == '__main__':
    run()
