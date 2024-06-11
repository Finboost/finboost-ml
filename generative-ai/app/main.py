from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering

app = Flask(__name__)

# Load the tokenizer and model
model_name = "Rifky/Indobert-QA"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)
model.load_weights('./models/gen-ai/model_weights_gen_ai.h5')

# Load the dataset
df = pd.read_csv('./data/generative-ai/final_dataset.csv')

def provide_recommendation():
    return "Terima kasih atas pertanyaannya. Saya ingin menjelaskan bahwa peran saya di sini adalah untuk memberikan panduan dan rekomendasi seputar isu keuangan"

def answer_question(question, context):
    if context is None:
        return provide_recommendation()

    inputs = tokenizer(question, context, return_tensors="tf", max_length=384, truncation=True, padding=True)
    outputs = model(inputs)

    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

    if answer_end < answer_start:
        answer_end = answer_start

    max_answer_length = 100
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end + max_answer_length]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).split("[SEP]")[0]

    return answer

def find_context_for_question(question, dataframe):
    max_matched_words = 0
    best_matched_context = None
    context_found = False

    for _, row in dataframe.iterrows():
        if question.strip().lower() in row['question'].strip().lower():
            best_matched_context = row['context']
            context_found = True
            break

    if not context_found:
        for _, row in dataframe.iterrows():
            dataset_tokens = set(row['question'].strip().lower().split())
            matched_words = len(set(question.strip().lower().split()).intersection(dataset_tokens))
            if matched_words > max_matched_words:
                max_matched_words = matched_words
                best_matched_context = row['context']

    return best_matched_context, context_found

def provide_recommendation_for_question(question):
    common_words = {
        "apa", "kenapa", "mengapa", "apa itu", "bagaimana", "siapa",
        "di mana", "kapan", "yang", "adalah", "untuk", "dengan",
        "ke", "dari", "atau", "dan", "jika", "jika", "maka",
        "seperti", "oleh", "agar", "sehingga", "karena", "namun",
        "jadi", "tidak", "adalah", "bahwa", "itu", "dalam", 
        "oleh", "pada", "untuk", "dengan", "tanpa", "saat", "dimana",
        "akan", "sudah", "belum", "pernah", "apakah", "apabila",
        "bagaimanakah", "sebagaimana", "adakah", "bilamana", "mengapakah",
        "kapankah", "dimanakah", "siapakah", "apa sajakah", "berapa", "indonesia"
    }

    def filter_common_words(tokens):
        return [word for word in tokens if word not in common_words]

    max_matched_words = 0
    best_matched_context = None

    question_tokens = set(filter_common_words(question.strip().lower().split()))

    for _, row in df.iterrows():
        dataset_tokens = set(filter_common_words(row['question'].strip().lower().split()))
        matched_words = len(question_tokens.intersection(dataset_tokens))
        if matched_words > max_matched_words:
            max_matched_words = matched_words
            best_matched_context = row['context']

    if best_matched_context:
        return best_matched_context
    else:
        return provide_recommendation()

def is_complex_prompt(prompt):
    # Define criteria for complex prompts
    threshold_length = 10
    return len(prompt.split()) > threshold_length

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data.get('question', '')
    
    is_expert = is_complex_prompt(question)
    
    if is_expert:
        return jsonify({
            'answer': (
                "Terima kasih atas pertanyaannya. Pertanyaan Anda terlalu kompleks untuk dijawab secara langsung. "
                "Namun, berikut beberapa rekomendasi yang mungkin dapat membantu mengelola keuangan Anda:\n\n"
                
                "1. **Buat anggaran:** Pantau pengeluaran Anda untuk mengidentifikasi area penghematan dan alokasi dana yang lebih efisien.\n\n"
                
                "2. **Cari pendapatan tambahan:** Pertimbangkan pekerjaan sampingan atau proyek paruh waktu untuk meningkatkan pendapatan Anda.\n\n"
                
                "3. **Investasi bijak:** Pelajari opsi investasi seperti saham, obligasi, atau properti. Konsultasikan dengan profesional keuangan jika perlu.\n\n"
                
                "4. **Kembangkan keterampilan:** Investasikan dalam pengembangan keterampilan dan pertimbangkan memonetisasi hobi atau minat Anda.\n\n"
                
                "5. **Perencanaan jangka panjang:** Buat rencana pensiun dan pastikan Anda memiliki perlindungan asuransi untuk risiko finansial.\n\n"
                
                "Semoga rekomendasi ini bermanfaat. Jika Anda membutuhkan bantuan lebih lanjut, jangan ragu untuk bertanya atau menggunakan fitur konsultasi dengan pakar keuangan kami. silahkan cek list expert di aplikasi kami."
            ),
            'isExpert': True
        })

    best_context, context_found = find_context_for_question(question, df)
    if not context_found:
        answer = provide_recommendation_for_question(question)
    else:
        answer = answer_question(question, best_context)
    
    return jsonify({
        'answer': answer,
        'isExpert': False
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
