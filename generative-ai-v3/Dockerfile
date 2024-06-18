# Gunakan image dasar Python
FROM python:3.9-slim

# Menetapkan direktori kerja
WORKDIR /app

# Menyalin requirements.txt ke image
COPY requirements.txt requirements.txt

# Menginstal dependensi
RUN pip install -r requirements.txt

# Menyalin seluruh aplikasi ke image
COPY . .

# Menetapkan variabel lingkungan
ENV PORT=8080

# Mengekspos port aplikasi
EXPOSE 8080

# Menjalankan aplikasi
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
