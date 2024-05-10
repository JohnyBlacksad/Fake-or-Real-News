import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go

# Загрузка данных
data = pd.read_csv('fake_news.csv')

# Разделение данных на обучающую и тестовую выборки
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Извлечение признаков из текста с помощью TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(X_train_vectorized, y_train)

# Предсказание на тестовой выборке
y_pred = classifier.predict(X_test_vectorized)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

# Построение матрицы ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
labels = ['Поддельные', 'Настоящие']
annotations = [
    [f'Правильно классифицированы как поддельные: {conf_matrix[0][0]}', f'Неправильно классифицированы как поддельные: {conf_matrix[0][1]}'],
    [f'Неправильно классифицированы как настоящие: {conf_matrix[1][0]}', f'Правильно классифицированы как настоящие: {conf_matrix[1][1]}']
]

fig = go.Figure(data=go.Heatmap(z=conf_matrix, x=labels, y=labels, colorscale='Viridis', text=annotations, hoverinfo='text'))
fig.update_layout(title='Матрица ошибок', xaxis_title='Предсказанный класс', yaxis_title='Истинный класс')
fig.show()

# Матрица важности слов
feature_names = vectorizer.get_feature_names_out()
coefs = classifier.coef_[0]
importance_scores = list(zip(feature_names, coefs))
importance_scores.sort(key=lambda x: x[1], reverse=True)

# Визуализация важных слов
word_scores = [score for word, score in importance_scores[:20]]
word_names = [word for word, score in importance_scores[:20]]

fig = go.Figure(data=[go.Bar(x=word_names, y=word_scores)])
fig.update_layout(title='Топ-20 важных слов', xaxis_title='Слово', yaxis_title='Важность')
fig.show()
