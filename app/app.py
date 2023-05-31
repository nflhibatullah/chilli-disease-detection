'''
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Activation, Dropout, LeakyReLU
from PIL import Image
from fungsi import make_model

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

model = None

NUM_CLASSES = 5
classes = ["healthy", "leaf curl", "leaf spot", "whitefly", "yelowish"]


# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
    return render_template('index.html')


# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['image']
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yg dipilih untuk diupload
    if filename != '':

        # Set/mendapatkan extension dan path dari file yg diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        # Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:

            # Simpan Gambar
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            # Memuat Gambar
            test_image = Image.open('.' + gambar_prediksi)

            # Mengubah Ukuran Gambar
            test_image_resized = test_image.resize((224, 224))

            # Konversi Gambar ke Array
            image_array = np.array(test_image_resized)
            test_image_x_count = (image_array / 255) - 0.5
            test_image_x = np.array([test_image_x_count])

            # Prediksi Gambar
            y_pred_test_single = model.predict(test_image_x)
            y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)


            hasil_prediksi = classes[y_pred_test_classes_single[0]]
            response = {}



            if hasil_prediksi == "healthy":
                response["disease"] = "healthy"
            elif hasil_prediksi == "whitefly":
                response["disease"] = "Whitefly (Serangga Kutu Kebul)"
                response["descriptionDisease"] = "Whitefly adalah serangga kecil berwarna putih yang menyerang " \
                                                 "tanaman dan menghisap cairan pada daunnya. Serangan whitefly dapat " \
                                                 "menyebabkan kerusakan pada tanaman dan penyebaran penyakit virus."
                response["tips"] = "Menggunakan insektisida yang efektif terhadap whitefly. Menggunakan penutup " \
                                   "tanaman berupa jaring serangga untuk mencegah masuknya whitefly.Menerapkan " \
                                   "praktik sanitasi yang baik, seperti membuang tanaman yang terinfeksi dan daun " \
                                   "yang terinfestasi whitefly."
            elif hasil_prediksi == "leaf curl":
                response["disease"] = "Leaf Curl (Keriting Daun)"
                response["descriptionDisease"] = "Leaf curl adalah penyakit yang menyebabkan daun tanaman keriting " \
                                                 "dan menggulung ke atas. Penyebab utama leaf curl adalah infeksi " \
                                                 "oleh virus. Penyakit ini biasanya menyebar melalui serangga " \
                                                 "perantara, seperti kutu daun atau trips."
                response["tips"] = "Mencabut dan membuang tanaman yang terinfeksi secara menyeluruh untuk mencegah " \
                                   "penyebaran penyakit. Mengendalikan serangga vektor yang membawa virus dengan " \
                                   "menggunakan insektisida.Menerapkan praktik sanitasi yang baik, " \
                                   "seperti membersihkan alat-alat taman yang terkontaminasi."
            elif hasil_prediksi == "leaf spot":
                response["disease"] = "Leaf Spot (Bercak Daun)"
                response["descriptionDisease"] = "Leaf spot adalah penyakit yang ditandai dengan munculnya " \
                                                 "bercak-bertekstur berbeda pada daun tanaman. Penyakit ini umumnya " \
                                                 "disebabkan oleh infeksi jamur atau bakteri."
                response["tips"] = "Membuang dan memusnahkan daun yang terinfeksi. Menghindari penyiraman daun pada " \
                                   "saat malam hari atau kondisi lembab yang berlebihan, karena kelembaban tinggi " \
                                   "memfasilitasi pertumbuhan penyakit. Menggunakan fungisida atau bakterisida yang " \
                                   "sesuai untuk mengendalikan infeksi jamur atau bakteri."
            elif hasil_prediksi == "yelowish":
                response["disease"] = "Yellowish (Kuning)"
                response["descriptionDisease"] = "Yellowish adalah penyakit pada tanaman yang ditandai dengan " \
                                                 "perubahan warna daun menjadi kuning. Penyebab utama yellowish " \
                                                 "adalah kekurangan nutrisi, khususnya zat besi. Tanaman yang " \
                                                 "mengalami yellowish sering tampak lemah dan pertumbuhannya terhambat."
                response["tips"] = "Cara mengatasi yellowish adalah dengan memberikan pupuk yang kaya zat besi kepada " \
                                   "tanaman. Pupuk dengan kandungan besi tinggi dapat membantu tanaman untuk kembali " \
                                   "menghasilkan daun yang sehat."

            # Return hasil prediksi dengan format JSON
            return jsonify(response)
        else:
            # Return hasil prediksi dengan format JSON
            gambar_prediksi = '(none)'
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })


# =[Main]========================================

if __name__ == '__main__':
    # Load model yang telah ditraining
    model = make_model()
    model.load_weights("model_penyakitcabai_tf_mobilenet.h5")

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
