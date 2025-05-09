# music-ai
html/css/js/py
Music Recommender Web App
Этот проект — простой веб-приложение на Flask, использующее 8 алгоритмов машинного обучения для рекомендаций песен и предсказаний, понравится ли песня пользователю.

Описание
Главная страница отображает 10 популярных песен.

Пользователь может выставить оценки от 1 до 10 для любых песен.

Нажатие на кнопку "Предсказать оценки (KNN)" прогнозирует оценки для неоценённых песен с помощью алгоритма K-Nearest Neighbors.

Кнопка "Предсказать другие модели" использует 7 других алгоритмов (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, SVM, Gradient Boosting, Linear Regression) для предсказания, понравится ли песня на основе имеющихся оценок.

Используемые алгоритмы
K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Random Forest

Naive Bayes

Support Vector Machine (SVM)

Gradient Boosting

Linear Regression

Установка и запуск
Клонируйте репозиторий:

bash
Копировать
Редактировать
git clone https://github.com/yourusername/music-recommender-app.git
cd music-recommender-app
Установите зависимости:

bash
Копировать
Редактировать
pip install -r requirements.txt
Запустите приложение:

bash
Копировать
Редактировать
python app.py
Откройте в браузере:

arduino
Копировать
Редактировать
http://localhost:5000
📁 Структура проекта
csharp
Копировать
Редактировать
music-recommender-app/
├── app.py               Основной Flask сервер
├── models.py            Модель машинного обучения
├── templates/
│   └── index.html       HTML шаблон
├── static/
│   └── style.css        Стили
├── requirements.txt     Зависимости
└── README.md            Документация
📋 Примечания
Все данные сгенерированы случайно, но можно адаптировать под настоящие данные.

Нет зависимости от количества оценок — модель будет работать даже с 1-2 выставленными оценками.

Лицензия
MIT License
