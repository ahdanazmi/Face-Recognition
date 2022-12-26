FROM python:3.10.5
RUN pip install opencv-python==4.6.0.66 scikit-learn==1.1.1 streamlit==1.11.1
COPY . /app
WORKDIR /app
CMD ["streamlit", "run", "main.py"]