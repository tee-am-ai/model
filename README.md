# Model Latih Tee-Am-AI

Proyek ini bertujuan untuk melatih model GPT-2 menggunakan dataset khusus untuk menghasilkan jawaban berbasis pertanyaan. Di bawah ini adalah langkah-langkah untuk mengatur lingkungan, melatih model, dan menguji model yang telah dilatih.

## 1. Mengatur Virtual Environment

1. **Buat Virtual Environment**

   Untuk membuat virtual environment baru, jalankan perintah berikut:

   ```bash
   python -m venv .venv

   ```

2. Aktivasi virtual environment

   ```bash
   .venv\Scripts\activate
   ```

3. Install semua library

   ```bash
   pip install -r requirements.txt
   ```

4. Running model

   ```bash
   python train.py
   ```

5. Testing model latih

   ```bash
   python test.py
   ```
