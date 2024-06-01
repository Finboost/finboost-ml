import pandas as pd
from datasets import Dataset
from transformers import TFBertForQuestionAnswering, BertTokenizerFast, DefaultDataCollator, create_optimizer
from transformers import TrainingArguments, Trainer
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import tf_keras

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

df = pd.read_csv('./data/final_dataset.csv')
dataset = Dataset.from_pandas(df)

model_name = "Rifky/Indobert-QA"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = TFBertForQuestionAnswering.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, (answer, offset) in enumerate(zip(answers, offset_mapping)):
        start_char = examples["answer_start"][i]
        end_char = start_char + len(answer)

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_idx = context_start
            while start_idx <= context_end and offset[start_idx][0] <= start_char:
                start_idx += 1
            start_positions.append(start_idx - 1)

            end_idx = context_start
            while end_idx <= context_end and offset[end_idx][1] < end_char:
                end_idx += 1
            end_positions.append(end_idx - 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

train_test_split = tokenized_datasets.train_test_split(test_size=0.01)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

training_args = TrainingArguments(
    output_dir="./models/fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,  # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    max_steps=1000,
)

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = train_dataset.shuffle(seed=42).to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'],
    batch_size=4,  # Reduced batch size
    collate_fn=data_collator,
    shuffle=True,
)
tf_eval_dataset = eval_dataset.to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'],
    batch_size=4,  # Reduced batch size
    collate_fn=data_collator,
    shuffle=False,
)

num_train_steps = len(tf_train_dataset) * training_args.num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=training_args.learning_rate,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=training_args.weight_decay,
)

def start_logits_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32))

def end_logits_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32))

model.compile(optimizer=optimizer, metrics=[start_logits_accuracy, end_logits_accuracy])

class LossAccuracyLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LossAccuracyLogger, self).__init__()
        self.epoch_loss = []
        self.start_logits_accuracy = []
        self.end_logits_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_loss.append(logs['loss'])
        self.start_logits_accuracy.append(logs.get('start_logits_accuracy'))
        self.end_logits_accuracy.append(logs.get('end_logits_accuracy'))
        print(f"Epoch {epoch + 1} - Loss: {logs['loss']}, Start Logits Accuracy: {logs.get('start_logits_accuracy')}, End Logits Accuracy: {logs.get('end_logits_accuracy')}")
        print(f"Logged data so far: Loss: {self.epoch_loss}, Start Logits Accuracy: {self.start_logits_accuracy}, End Logits Accuracy: {self.end_logits_accuracy}")

    def plot(self):
        epochs = range(1, len(self.epoch_loss) + 1)
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.epoch_loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.start_logits_accuracy, label='Start Logits Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Start Logits Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.end_logits_accuracy, label='End Logits Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('End Logits Accuracy')
        plt.legend()
        
        plt.show()
        
logger = LossAccuracyLogger()

class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, eval_dataset):
        super(MetricsLogger, self).__init__()
        self.epoch_f1 = []
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, epoch, logs=None):
        y_true_start, y_true_end = [], []
        y_pred_start, y_pred_end = [], []

        for batch in self.eval_dataset:
            inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = {k: v for k, v in batch.items() if k in ['start_positions', 'end_positions']}
            start_true = labels['start_positions'].numpy()
            end_true = labels['end_positions'].numpy()
            outputs = self.model.predict(inputs)
            start_pred = np.argmax(outputs['start_logits'], axis=-1)
            end_pred = np.argmax(outputs['end_logits'], axis=-1)

            y_true_start.extend(start_true)
            y_true_end.extend(end_true)
            y_pred_start.extend(start_pred)
            y_pred_end.extend(end_pred)

        start_f1 = f1_score(y_true_start, y_pred_start, average='micro')
        end_f1 = f1_score(y_true_end, y_pred_end, average='micro')
        overall_f1 = (start_f1 + end_f1) / 2
        self.epoch_f1.append(overall_f1)
        print(f"Epoch {epoch + 1} - F1 Score: {overall_f1}")
        print(f"Logged F1 data so far: {self.epoch_f1}")

    def plot(self):
        epochs = range(1, len(self.epoch_f1) + 1)
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.epoch_f1, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.show()
        
metrics_logger = MetricsLogger(tf_eval_dataset)

class GradientAccumulation(tf.keras.callbacks.Callback):
    def __init__(self, accum_steps=2):
        super(GradientAccumulation, self).__init__()
        self.accum_steps = accum_steps
        self.step = 0
        self.accumulated_grads = None

    def on_train_batch_begin(self, batch, logs=None):
        if self.step % self.accum_steps == 0:
            self.accumulated_grads = [tf.zeros_like(var) for var in self.model.trainable_variables]

    def on_train_batch_end(self, batch, logs=None):
        grads = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_variables)
        self.accumulated_grads = [accum_grad + grad / self.accum_steps for accum_grad, grad in zip(self.accumulated_grads, grads)]
        if self.step % self.accum_steps == 0:
            self.model.optimizer.apply_gradients(zip(self.accumulated_grads, self.model.trainable_variables))
            self.model.optimizer.set_weights(self.accumulated_grads)
        self.step += 1

gradient_accumulation = GradientAccumulation(accum_steps=2)  # Adjust the accumulation steps as needed

# Train the model with gradient accumulation
history = model.fit(tf_train_dataset, epochs=training_args.num_train_epochs, callbacks=[logger, metrics_logger, gradient_accumulation], validation_data=tf_eval_dataset)
# Train the model
# history = model.fit(tf_train_dataset, epochs=training_args.num_train_epochs, callbacks=[logger, metrics_logger], validation_data=tf_eval_dataset)


def provide_recommendation():
    recommendation = ("Terima kasih atas pertanyaannya. Saya ingin menjelaskan bahwa peran saya di sini adalah untuk "
                      "memberikan panduan dan rekomendasi seputar isu keuangan. Meskipun saya tidak memberikan jawaban "
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
                      "yang solid. Ini termasuk perencanaan pensiun yang baik serta perlindungan asuransi untuk melindungi Anda "
                      "dari risiko finansial yang tidak terduga.\n\n"
                      
                      "Semoga rekomendasi ini memberikan arahan yang berguna bagi Anda dalam memulai atau meningkatkan "
                      "perjalanan keuangan Anda. Jika Anda memiliki pertanyaan lebih lanjut atau membutuhkan bantuan tambahan, "
                      "jangan ragu untuk bertanya. Saya siap membantu Anda dalam segala hal terkait keuangan Anda.")
    
    return recommendation

def answer_question(question, context):
    if context is None:
        return provide_recommendation()  # Return recommendation if context not found

    inputs = tokenizer(question, context, return_tensors="tf", max_length=384, truncation=True, padding=True)

    outputs = model(inputs)

    # Get the start and end positions of the answer
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

    # Ensure the answer_end is greater than or equal to answer_start
    if answer_end < answer_start:
        answer_end = answer_start

    # Extract the answer tokens, extending to a reasonable length if necessary
    max_answer_length = 100
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end + max_answer_length]

    # Convert tokens to text, skipping special tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Strip out any trailing special tokens (like [SEP])
    answer = answer.split("[SEP]")[0]

    return answer

def find_context_for_question(question, dataframe):
    max_matched_words = 0
    best_matched_context = None
    context_found = False

    # Iterate through the rows of the dataframe
    for _, row in dataframe.iterrows():
        if question.strip().lower() in row['question'].strip().lower():
            best_matched_context = row['context']
            context_found = True
            break  # Break the loop if a matching context is found

    # If no exact match is found, find the best matching context based on the number of common words
    if not context_found:
        for _, row in dataframe.iterrows():
            dataset_tokens = set(row['question'].strip().lower().split())
            matched_words = len(set(question.strip().lower().split()).intersection(dataset_tokens))
            if matched_words > max_matched_words:
                max_matched_words = matched_words
                best_matched_context = row['context']

    return best_matched_context, context_found

user_question = "apa itu cuaca"

context, context_found = find_context_for_question(user_question, df)

def provide_recommendation_for_question(question):
    # Define the common words to ignore
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

    # Function to filter out common words from a list of tokens
    def filter_common_words(tokens):
        return [word for word in tokens if word not in common_words]

    max_matched_words = 0
    best_matched_context = None

    # Tokenize and filter the question
    question_tokens = set(filter_common_words(question.strip().lower().split()))

    # Iterate through the rows of the dataset
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

if context_found:
    answer = answer_question(user_question, context)
    print(f"Q: {user_question}\nA: {answer}")
else:
    answer = provide_recommendation_for_question(user_question)
    print(f"Q: {user_question}\nA: {answer}")
    
model.save('./models/my_model.h5')