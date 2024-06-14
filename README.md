<h1 align="center">Finboost Machine Learning</h1>

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

Finboost Machine Learning is a financial chatbot project that can answer finance-related questions in Indonesian. The project leverages deep learning models with fine-tuning on the `cahya/gpt2-small-indonesian-522M` model for generative AI and uses an LSTM model for question suggestion.

## Tech Stack

- **Python**: Primary programming language.
- **Transformers**: Hugging Face library for transformer models.
- **Datasets**: Library for managing datasets.
- **TensorFlow**: Deep learning framework used as the backend.
- **Pandas**: Library for managing data in DataFrame format.

## Architecture

This project uses the following architecture:

1. **Data Preparation**: Preparing question and answer data in CSV format.
2. **Model Fine-Tuning**: Fine-tuning the `cahya/gpt2-small-indonesian-522M` model using the dataset for generative AI.
3. **LSTM Model**: Using an LSTM model for question suggestion.

## File and Folder Structure

| File/Folder Name                              | Description                                                            |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `data/generative-ai/finansial-dataset-v2.csv` | Example dataset generative-ai in CSV format                            |
| `data/question-suggestion/data.csv`           | Example dataset question suggestion in CSV format                      |
| `models/fine_tuned_model`                     | Directory to store the fine-tuned generative AI model                  |
| `notebooks/generative_ai.ipynb`               | Notebook for fine-tuning and using the generative AI model             |
| `notebooks/question_suggestion.ipynb`         | Notebook for training and using the LSTM model for question suggestion |
| `preprocessing/combined_dataset.ipynb`        | notebook for preprocessing the collected dataset                       |
| `scripts/`                                    | Folder to save the script in the next feature                          |
| `generative-ai/`                              | Folder to deploy generative-ai model                                   |
| `generative-ai-v2/`                           | Folder to deploy generative-ai-v2 model                                |
| `question-suggestion/`                        | Folder to deploy question-suggestion model                             |
| `requirements.txt`                            | List of dependencies for this project                                  |

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
   - Open generative_ai.ipynb and question_suggestion.ipynb in Google Colab or Jupyter Notebook.

## Usage

Run the notebook:

1. Run the Generative AI notebook:

- Execute all cells in the generative_ai.ipynb notebook to fine-tune the cahya/gpt2-small-indonesian-522M model and generate responses.

2. Run the Question Suggestion notebook:

- Execute all cells in the question_suggestion.ipynb notebook to train the LSTM model and make question suggestions.

3. Fine-Tuning:

- Use the fine-tuning script generative_ai.py to fine-tune the cahya/gpt2-small-indonesian-522M model on new data.

4. Question Suggestion:

- Use the question_suggestion.py script to train and use the LSTM model for question suggestion.

## Contributing

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

(TBC)

---

Anda dapat menyimpan konten di atas ke dalam file `README.md` untuk proyek Anda. File ini mencakup semua informasi penting tentang proyek, termasuk struktur folder, langkah-langkah setup, dan cara penggunaan.
