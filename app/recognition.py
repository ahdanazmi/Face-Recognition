import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import pickle

def run():
    # Fungsi untuk mendeteksi wajah dan return embedding wajah
    def detect_and_embed(image):
        # deteksi wajah dengan OpenCV
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(image, 1.3, 5)
        
        # Jika tidak ada wajah yang terdeteksi, return None
        if len(faces) == 0:
            return None
        # Jika tidak ada wajah yang terdeteksi, return None
        elif len(faces) > 1:
            return 'more_than_1'
        # Jika terdeteksi satu wajah, return embedding
        else:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            embedding = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            return embedding

    # Fungsi untuk mengecek kesamaan wajah yang diinput dengan wajah di database
    def recognize_face(embedding, database):
        # Inisialisasi nama dan nilai similarity terkecil
        name = 'Your face is not registered'
        max_similarity = -1.0
        
        # Loop setiap data di database
        for data in database:
            # Ambil embedding wajah dan nama dari data
            stored_embedding, stored_name = data
            
            # Hitung similarity antara wajah yang terdeteksi dengan embedding di database
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

    # Buat endpoint menggunakan Streamlit
    st.title('Recognition')

    # Input gambar wajah dengan form
    with st.form(key= 'form_parameter'):
        image = st.file_uploader('Upload image', type='jpg')
        submitted = st.form_submit_button('Check')

    # Jika tombol Check ditekan maka wajah akan dideteksi dan dibandingkan dengan database
    if submitted:
        # Baca gambar dan ubah ke dalam format BGR
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # menjalankan fungsi detect_and_embed
        embedding = detect_and_embed(image)
        
        # Jika tidak ada wajah yang terdeteksi maka munculkan pesan error
        if embedding is None:
            st.error('No face detected')
        # Jika wajah lebih dari 1 maka munculkan pesan error
        elif embedding == "more_than_1":
            st.error('More than one face is detected')

        # Jika wajah terdeteksi maka munculkan nama
        else:
            # Cek kesamaan wajah dengan wajah di database
            name = recognize_face(embedding, database)
            # Tampilkan nama yang terdeteksi
            st.success(name)

if __name__ == '__main__':
    run()
