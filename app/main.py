import streamlit as st
import recognition
import register

navigation = st.sidebar.selectbox('Choose Page : ', ('Recognition','Register'))

if navigation=='Recognition':
    recognition.run()
else:
    register.run()