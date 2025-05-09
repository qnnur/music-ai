# models.py
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

def train_models():
    songs = [
        "Shape of You - Ed Sheeran",
        "Blinding Lights - The Weeknd",
        "Levitating - Dua Lipa",
        "Stay - Justin Bieber & The Kid LAROI",
        "Good 4 U - Olivia Rodrigo",
        "Save Your Tears - The Weeknd",
        "Industry Baby - Lil Nas X & Jack Harlow",
        "Peaches - Justin Bieber",
        "Watermelon Sugar - Harry Styles",
        "Bad Habit - Steve Lacy"
    ]

    X_train = np.random.randint(1, 11, size=(100, 10))
    y_clf = (X_train[:, 7] > 5).astype(int)

    models = {
        "KNN": KNeighborsRegressor().fit(X_train, X_train)
    }

    for i in range(1, 10):
        models[f"Logistic Regression ({i})"] = LogisticRegression(max_iter=1000).fit(X_train[:, :i], y_clf)
        models[f"Decision Tree ({i})"] = DecisionTreeClassifier().fit(X_train[:, :i], y_clf)
        models[f"Random Forest ({i})"] = RandomForestClassifier().fit(X_train[:, :i], y_clf)
        models[f"Naive Bayes ({i})"] = GaussianNB().fit(X_train[:, :i], y_clf)
        models[f"SVM ({i})"] = SVC().fit(X_train[:, :i], y_clf)
        models[f"Gradient Boosting ({i})"] = GradientBoostingClassifier().fit(X_train[:, :i], y_clf)
        models[f"Linear Regression ({i})"] = LinearRegression().fit(X_train[:, :i], y_clf)

    return {"models": models, "songs": songs}

def knn_predict(model, ratings):
    input_data = [r if r is not None else 0 for r in ratings]
    input_data = np.array(input_data).reshape(1, -1)
    y_pred = model.predict(input_data)[0]
    result = {}
    for i, r in enumerate(ratings):
        if r is None:
            result[f"{i+1}"] = f"{y_pred[i]:.2f}"
    return result

def other_models_predict(models, ratings):
    known_ratings = [r for r in ratings if r is not None]
    if not known_ratings:
        return {"Ошибка": "Нужно хотя бы 1 оценка"}

    X = np.array(known_ratings).reshape(1, -1)
    results = {}

    for name, model in models.items():
        if name == "KNN":
            continue
        try:
            n = int(name.split("(")[-1].split(")")[0])
            if len(known_ratings) < n:
                continue
            pred = model.predict(X[:, :n])[0]
            if "Linear Regression" in name:
                results[name] = "Нравится" if pred >= 0.5 else "Не нравится"
            else:
                results[name] = "Нравится" if pred == 1 else "Не нравится"
        except:
            results[name] = "Ошибка"
    return results
