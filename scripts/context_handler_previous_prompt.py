from context.financial_keywords import FINANCIAL_KEYWORDS

# Inisialisasi variabel untuk menyimpan konteks
previous_prompt = ""

# Fungsi untuk menyimpan prompt sebelumnya
def save_previous_prompt(prompt):
    global previous_prompt
    previous_prompt = prompt

# Fungsi untuk mengambil prompt sebelumnya
def get_previous_prompt():
    global previous_prompt
    return previous_prompt

# Fungsi untuk menentukan apakah prompt baru berkaitan dengan prompt sebelumnya
def is_related_to_previous(prompt):
    previous_prompt = get_previous_prompt()
    # Contoh analisis sederhana: jika prompt baru mengandung kata kunci yang sama dengan prompt sebelumnya, maka dianggap berkaitan
    if previous_prompt and any(keyword in prompt.lower() for keyword in FINANCIAL_KEYWORDS):
        return True
    return False

# Fungsi untuk merespons prompt baru berdasarkan konteks
def respond_to_prompt(prompt):
    # Simpan prompt sebelumnya sebelum melakukan pengecekan konteks
    previous_prompt = get_previous_prompt()
    save_previous_prompt(prompt)
    if is_related_to_previous(prompt):
        # Lakukan sesuatu berdasarkan konteks yang terdeteksi
        response = "Ini terkait dengan pertanyaan sebelumnya tentang finansial."
    else:
        # Tanggapi prompt baru secara independen
        response = "Ini adalah pertanyaan baru yang tidak terkait dengan konteks sebelumnya."
    return response


# Prompt pertama
prompt1 = "Apa itu finansial?"
response1 = respond_to_prompt(prompt1)
print(response1)  # Output: Ini adalah pertanyaan baru yang tidak terkait dengan konteks sebelumnya.

# Prompt kedua
prompt2 = "Lalu kenapa dinamakan begitu?"
response2 = respond_to_prompt(prompt2)
print(response2)  # Output: Ini adalah pertanyaan baru yang tidak terkait dengan konteks sebelumnya.
