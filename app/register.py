import streamlit as st
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
        # Jika terdeteksi satu wajah, return embedding wajah
        else:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            embedding = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            return embedding

    # Fungsi untuk menyimpan embedding wajah dan nama ke database
    def save_to_database(embedding, name):
    # Coba buka database jika sudah ada
        try:
            with open('database.pkl', 'rb') as f:
                data = pickle.load(f)
        except:
            data = []
            
        # Tambahkan embedding wajah dan nama ke database
        data.append((embedding, name))
        
        # Simpan database ke file pickle
        with open('database.pkl', 'wb') as f:
            pickle.dump(data, f)


    # Buat endpoint menggunakan Streamlit
    st.title('Register')

    # Input wajah dan nama menggunakan form
    with st.form(key= 'form_parameter'):
        image = st.file_uploader('Upload image', type='jpg')
        name = st.text_input('Name')
        submitted = st.form_submit_button('Register')

    # Jika tombol register ditekan maka wajah dideteksi dan disimpan ke database
    if submitted:
        # Baca gambar dan ubah ke dalam format grayscale
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        # menjalankan fungsi detect_and_embed
        embedding = detect_and_embed(image)
        
        # Jika tidak ada wajah yang terdeteksi maka munculkan pesan error
        if embedding is None:
            st.error('No face detected')
        # Jika wajah lebih dari 1 maka munculkan pesan error
        elif embedding == 'more_than_1':
            st.error('More than one face is detected')
        # Jika wajah terdeteksi, simpan ke databas
        else:
            save_to_database(embedding, name)
            st.success('Your face has been registered')

if __name__ == '__main__':
    run()

