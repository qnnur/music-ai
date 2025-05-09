from flask import Flask, render_template, request
from models import train_models, knn_predict, other_models_predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    # Получаем модели и песни
    data = train_models()
    models = data["models"]
    songs = data["songs"]

    if request.method == "POST":
        # Получаем рейтинги песен, которые пользователь оценил
        ratings = [
            request.form.get(f"song_{i+1}", type=int) for i in range(10)
        ]
        
        # Прогнозируем оценки для песен с помощью KNN
        if "predict_knn" in request.form:
            knn_result = knn_predict(models["KNN"], ratings)
            return render_template("index.html", songs=songs, knn_result=knn_result)
        
        # Прогнозируем нравится ли песня с помощью других моделей
        if "predict_other" in request.form:
            other_result = other_models_predict(models, ratings)
            return render_template("index.html", songs=songs, other_result=other_result)

    return render_template("index.html", songs=songs)

if __name__ == "__main__":
    app.run(debug=True)
