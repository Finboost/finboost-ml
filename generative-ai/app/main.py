from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForQuestionAnswering
import requests

app = Flask(__name__)

# Load the tokenizer and model
model_name = "Rifky/Indobert-QA"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)
model.load_weights('./models/gen-ai/model_weights_gen_ai.h5')

# URL of the model weights
# weights_url = 'https://storage.googleapis.com/finboost-generative-ai-model/my_model_weights.h5'

# # Download the model weights
# local_weights_file = 'my_model_weights.h5'
# response = requests.get(weights_url)

# with open(local_weights_file, 'wb') as f:
#     f.write(response.content)

# # Load the model weights
# model.load_weights(local_weights_file)

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
        "oleh", "pada", "untuk", "dengan", "tanpa", "saat",
        "akan", "sudah", "belum", "pernah", "apakah", "apabila",
        "bagaimanakah", "sebagaimana", "adakah", "bilamana", "mengapakah",
        "kapankah", "dimanakah", "siapakah", "apa sajakah", "berapa"
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
    
    if is_complex_prompt(question):
        return jsonify({
            'answer': ("Terima kasih atas pertanyaannya. Namun pertanyaan anda terlalu kompleks. Meskipun saya tidak memberikan jawaban "
                    "yang langsung terkait dengan pertanyaan Anda, saya berharap rekomendasi berikut dapat membantu Anda "
                    "mengelola keuangan Anda dengan lebih baik:\n\n"
                    
                    "1. Pertama-tama, langkah yang paling penting adalah membangun anggaran yang terperinci dan "
                    "memantau pengeluaran Anda secara cermat. Dengan demikian, Anda dapat mengidentifikasi area di mana Anda "
                    "dapat menghemat dan mengalokasikan dana dengan lebih efisien.\n\n"
                    
                    "2. Selain mengelola pengeluaran, pertimbangkan untuk mencari peluang pendapatan tambahan melalui "
                    "pekerjaan sampingan atau proyek-proyek paruh waktu. Hal ini dapat membantu meningkatkan pendapatan Anda "
                    "dan memperluas sumber pendapatan.\n\n"
                    
                    "3. Selanjutnya, luangkan waktu untuk mempelajari opsi investasi yang tersedia dan alokasikan dana Anda "
                    "dengan bijak. Mungkin Anda ingin mempertimbangkan investasi dalam instrumen keuangan seperti saham, "
                    "obligasi, atau properti. Namun, pastikan untuk melakukan riset yang teliti dan berkonsultasi dengan "
                    "profesional keuangan jika diperlukan.\n\n"
                    
                    "4. Selain itu, penting untuk terus mengembangkan keterampilan yang bernilai tinggi dalam karier Anda. "
                    "Pertimbangkan untuk memonetisasi hobi atau minat Anda sebagai sumber pendapatan tambahan. Hal ini dapat "
                    "membantu meningkatkan potensi penghasilan Anda di masa mendatang.\n\n"
                    
                    "5. Terakhir, namun tidak kalah pentingnya, pastikan Anda memiliki perencanaan keuangan jangka panjang "
                    "yang tepat. Ini termasuk perencanaan pensiun yang baik serta perlindungan asuransi untuk melindungi Anda "
                    "dari risiko finansial yang tidak terduga.\n\n"
                    
                    "Semoga rekomendasi ini memberikan arahan yang berguna bagi Anda dalam memulai atau meningkatkan "
                    "perjalanan keuangan Anda. Jika Anda memiliki pertanyaan lebih lanjut atau membutuhkan bantuan tambahan, "
                    "jangan ragu untuk bertanya. Saya siap membantu Anda dalam segala hal terkait keuangan Anda.\n\n"
                    
                    "Jika Anda memerlukan penjelasan lebih lanjut atau bantuan dari seorang ahli, Anda dapat menggunakan fitur "
                    "konsultasi dengan pakar keuangan kami [di sini](#https://finboost-waitlist.vercel.app/).")
        })

    best_context, context_found = find_context_for_question(question, df)
    if not context_found:
        answer = provide_recommendation_for_question(question)
    else:
        answer = answer_question(question, best_context)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
