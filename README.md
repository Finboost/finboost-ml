<h1 align="center">Finboost Machine Learning</h1>

# Finboost Machine Learning

## Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [File and Folder Structure](#file-and-folder-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Finboost Machine Learning adalah proyek chatbot finansial yang dapat menjawab pertanyaan terkait topik finansial dalam bahasa Indonesia. Proyek ini menggunakan model deep learning dengan fine-tuning pada model `deepset/roberta-base-squad2` dan integrasi dengan RAG (Retrieval-Augmented Generation) untuk menghasilkan jawaban yang lebih akurat dan relevan.

## Tech Stack

- **Python**: Bahasa pemrograman utama yang digunakan.
- **Transformers**: Pustaka dari Hugging Face untuk memanfaatkan model transformer.
- **Datasets**: Pustaka untuk mengelola dataset.
- **PyTorch**: Framework deep learning yang digunakan sebagai backend.
- **FAISS**: Library untuk pencarian vektor yang cepat, digunakan oleh RAG.
- **Pandas**: Pustaka untuk mengelola data dalam bentuk DataFrame.

## Architecture

Proyek ini menggunakan arsitektur berikut:

- **Data Preparation**: Data pertanyaan dan jawaban disiapkan dalam format CSV.
- **Model Fine-Tuning**: Model `deepset/roberta-base-squad2` di-fine-tune menggunakan dataset.
- **RAG (Retrieval-Augmented Generation)**: Digunakan untuk memperkaya jawaban dengan informasi tambahan.
- **Integration**: Hasil dari model fine-tuned dan RAG digabungkan untuk memberikan jawaban akhir.

## File and Folder Structure

| File/Folder Name                        | Description                                                                                                 |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `data/dataset.csv`                      | Dataset contoh dalam format CSV                                                                             |
| `models/fine_tuned_model`               | Direktori untuk menyimpan model yang telah di-fine-tune                                                     |
| `notebooks/finchat.ipynb`               | Notebook utama untuk fine-tuning dan penggunaan RAG                                                         |
| `scripts/fine_tuning.py`                | Script untuk melakukan fine-tuning pada model deepset/roberta-base-squad2 (inggris only, tapi lebih akurat) |
| `scripts/indo_fine_tuning.py`           | Script untuk melakukan fine-tuning pada model dengan model xlm-roberta-base (support bahasa indonesia)      |
| `scripts/rag_integration.py`            | Script untuk mengintegrasikan RAG dengan model fine-tuned                                                   |
| `scripts/context/financial_keywords.py` | Daftar kata kata yang berhubungan dengan finansial                                                          |
| `requirements.txt`                      | Daftar dependencies untuk proyek ini                                                                        |

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/finboost-ml.git
   cd finboost-ml
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Upload the dataset to the appropriate folder:**
   - Place dataset.csv in the data/ directory.
4. **Open the notebook:**
   - Open chatbot_finansial.ipynb in Google Colab or Jupyter Notebook.

## Usage

Run the notebook:

1. Run the notebook:

- Execute all cells in the chatbot_finansial.ipynb notebook to train the model and test the integration with RAG.
  Fine-Tuning:

2. Fine-Tuning:

- The fine-tuning script fine_tuning.py can be used independently to fine-tune the model on new data.
  Integration with RAG:

3. Integration with RAG:

- Use the rag_integration.py script to integrate the fine-tuned model with RAG for enhanced answers.

## Contributing

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

(belum tau)

---

Anda dapat menyimpan konten di atas ke dalam file `README.md` untuk proyek Anda. File ini mencakup semua informasi penting tentang proyek, termasuk struktur folder, langkah-langkah setup, dan cara penggunaan.
