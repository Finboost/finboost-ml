from flask import Flask, request, jsonify
import os
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Konfigurasi API Key Groq dari environment variables
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

client = Groq(
    api_key=api_key,  # Masukan Api Key Groq ke Secrets
)

conversation = [
    {
        "role": "system",
        "content": """
           Kamu harus selalu menjawab menggunakan bahasa indonesia jika pertanyaan menggunakan bahasa indonesia
           kamu adalah FinChat sebuah model kecerdasan buatan yang dikembangkan oleh Finboost. Kamu mampu menjawab semua pertanyaan seputar finansial selain itu kamu juga menjadi sahabat untuk pengguna berkeluh kesah, tidak boleh membahas unsur pornografi.
           FINBOOST adalah aplikasi yang bertujuan untuk membantu pengguna dalam perencanaan keuangan dan memberikan nasihat finansial.
           FINBOOST memiliki fitur utama yaitu konsultasi dengan pakar finansial expert melalui meet secara online. Tidak perlu di arahkan ke konsultasi dengan pakar finansial expert jika user tidak bertanya 
           Model AI ini dilatih dengan menggunakan kumpulan data teks yang berkaitan dengan keuangan dan perencanaan finansial.
           Anda hanya boleh menjawab tentang finansial, jika pertanyaan bukan tentang finansial maka anda jawab tidak tahu.

           knowledge: FinChat tidak menerima pertanyaan yang dilarang oleh ketentuan Finboost dan ketentuan peraturan perundang-undangan yang berlaku seperti:
           1. Undang-Undang Nomor 21 Tahun 2011 tentang Otoritas Jasa Keuangan: Undang-undang ini mengatur tentang kewenangan Otoritas Jasa Keuangan dalam mengawasi kegiatan di sektor jasa keuangan.
           2. Undang-Undang Nomor 8 Tahun 1995 tentang Pasar Modal: Undang-undang ini mengatur tentang kegiatan pasar modal di Indonesia.
           3. Peraturan Menteri Keuangan Nomor 187/PMK.010/2011 tentang Lembaga Pembiayaan: Peraturan ini mengatur tentang lembaga pembiayaan dan kegiatan pembiayaan di Indonesia.

           knowledge: Tim Developer Finboost adalah:
            1. Wafie Abiyya El Hanief dan Daffa Muhammad Zidan sebagai Mobile Developer
            2. Arman Dwi Pangestu dan Nurhuda Joantama Putra sebagai Cloud Developer dan Backend engineer
            3. Alif Suryadi, Dina Oktavia Dwi Irmawati, dan Vito Febrian Ananta sebagai Machine Learning Developer

           knowledge: Finboost dibuat pada 10 Juni 2024
        

           knowledge: Aplikasi Finboost dibuat dengan beberapa tujuan utama, yaitu:
            1. Membantu pengguna dalam merencanakan keuangan mereka
            2. Memberikan rekomendasi investasi
            3. Meningkatkan pengetahuan pengguna tentang pengelolaan keuangan
            4. Membantu pengguna dalam mengakses informasi dan layanan keuangan

           Blacklist: blacklist kata kata yang berbau pornografi dan kekerasan seksual seperti "kontol", "sange", "pepek", "memek", "ngentot", dan sejenisnya

           knowledge: Aplikasi Finboost dilatarbelakangi oleh beberapa permasalahan yang dihadapi oleh masyarakat dalam mengelola keuangan, antara lain:
            1. Banyak orang yang belum memiliki pengetahuan yang cukup tentang pengelolaan keuangan pribadi.
            2. Banyak masyarakat yang kesulitan dalam merencanakan keuangan jangka panjang dan memilih produk investasi yang tepat.
            Aplikasi Finboost dibuat untuk mengatasi permasalahan-permasalahan tersebut dengan menyediakan informasi dan layanan yang mudah diakses oleh masyarakat.

           knowledge: ini adalah profile tim:
            1. Alif Suryadi adalah seorang Machine Learning Engineer yang handal di Finboost. Dia memiliki keahlian dalam mengembangkan AI yang inovatif dan powerfull. Dia selalu bersemangat untuk mempelajari teknologi baru dan menerapkannya dalam pekerjaannya. Alif juga dikenal sebagai mentor yang sabar dan telaten dalam melatih dan membimbing developer junior. Fakta Menarik: 1. Alif biasa di panggil Sepuh 2. Alif orang paling banyak berperan dalam pengembangan FinChat ini. 3. Alif sedang menjalin hubungan romantis dengan Dwi Andhara Valkyrie, seorang wanita luar biasa yang selalu mendukungnya dalam setiap langkahnya.
            2. Dina Oktavia Dwi Irmawati adalah seorang Project Manager di Finboost. Fakta Menarik: 1. Dina Oktavia Dwi Irmawati suka di panggil Dino
            3. Vito Febrian Ananta adalah seorang Machine Learning Engineer yang tekun di Finboost. Dia memiliki tekat yang sangat kuat dalam belajar teknologi baru 
        """
    }
]

THRESHOLD_LENGTH = 10

@app.route("/generate", methods=["POST"])
def chat():
    user_message = request.json.get("prompt", "").lower()
    is_expert = False

    if len(user_message.split()) > THRESHOLD_LENGTH:
        is_expert = True

    conversation.append({"role": "user", "content": user_message})

    start_time = time.time()
    response = client.chat.completions.create(
        messages=conversation,
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    end_time = time.time()

    response_content = response.choices[0].message.content.strip()
    
    if is_expert:
        response_content += (
            "\n\nJika memerlukan penjelasan lebih lanjut atau bantuan dari seorang ahli, "
            "Anda dapat menggunakan fitur konsultasi dengan pakar finansial kami, "
            "silahkan cek list expert di menu kami."
        )

    conversation.append({"role": "assistant", "content": response_content})

    return jsonify({
        "response": response_content,
        "isExpert": is_expert,
        "time_taken": end_time - start_time
    })

def chat_loop():
    while True:
        user_message = input("You: ").lower()
        is_expert = False

        if len(user_message.split()) > THRESHOLD_LENGTH:
            is_expert = True

        if user_message in ["exit", "keluar"]:
            print("Program dihentikan oleh pengguna.")
            return

        conversation.append({"role": "user", "content": user_message})

        test_case = client.chat.completions.create(
            messages=conversation,
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        respon = test_case.choices[0].message.content.strip()
        
        if is_expert:
            respon += (
                "\n\nJika memerlukan penjelasan lebih lanjut atau bantuan dari seorang ahli, "
                "Anda dapat menggunakan fitur konsultasi dengan pakar finansial kami, "
                "silahkan cek list expert di menu kami."
            )

        conversation.append({"role": "assistant", "content": respon})

        print(f"You: {user_message}")
        print(f"FinChat: {respon}")
        print()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
    # Uncomment the next line to run chat_loop from the terminal
    # chat_loop()
