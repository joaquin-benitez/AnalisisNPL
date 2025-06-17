import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import nltk
import re
import spacy
from collections import Counter
from sklearn.decomposition import PCA

# --- Descargas necesarias ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cargar modelo de spaCy (puede tardar la primera vez)
try:
    nlp = spacy.load("en_core_web_md")
except:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# --- Cargar y preparar datos ---
df = pd.read_csv('ChatGPT_Reviews.csv')
text_col = 'Review'
df = df.dropna(subset=[text_col])

# --- Preprocesamiento ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned'] = df[text_col].apply(clean_text)

# Muestra aleatoria para acelerar procesamiento
df = df.sample(300, random_state=42).reset_index(drop=True)

# --- Wordcloud ---
all_words = ' '.join(df['cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras')
plt.show()

# --- Análisis de Sentimiento ---
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df[text_col].apply(get_sentiment)

plt.hist(df['sentiment'], bins=20, color='skyblue')
plt.title("Distribución de Sentimientos")
plt.xlabel("Polaridad")
plt.ylabel("Frecuencia")
plt.show()

# Clasificación de sentimiento
def classify_sentiment(score):
    if score > 0.1:
        return 'positivo'
    elif score < -0.1:
        return 'negativo'
    else:
        return 'neutral'

df['sentiment_class'] = df['sentiment'].apply(classify_sentiment)
print(df['sentiment_class'].value_counts())

# --- Palabras más frecuentes ---
word_freq = Counter(all_words.split())
common_words = word_freq.most_common(10)

plt.bar([w[0] for w in common_words], [w[1] for w in common_words], color='orange')
plt.title('Top 10 Palabras Más Frecuentes')
plt.xticks(rotation=45)
plt.show()

# --- Word Embeddings ---
def get_vector(text):
    doc = nlp(text)
    return doc.vector

df['vector'] = df['cleaned'].apply(get_vector)

# Reducción a 2D con PCA para graficar
X = df['vector'].tolist()
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

df['pca_x'] = X_reduced[:, 0]
df['pca_y'] = X_reduced[:, 1]

# Visualización
colors = {'positivo': 'green', 'neutral': 'gray', 'negativo': 'red'}

plt.figure(figsize=(8, 6))
for sentiment in df['sentiment_class'].unique():
    subset = df[df['sentiment_class'] == sentiment]
    plt.scatter(subset['pca_x'], subset['pca_y'], label=sentiment, alpha=0.6, c=colors[sentiment])
plt.title('Distribución de Reseñas según Word Embeddings')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.show()

# --- Guardar resultados ---
df.to_csv('reviews_analizadas.csv', index=False)

# --- 1. Codificar etiquetas ---
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
valid_df['label'] = le.fit_transform(valid_df['sentiment_class'])
y = to_categorical(valid_df['label'])

# --- 2. Dividir datos ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Crear el modelo ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 clases: positivo, negativo, neutral

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Entrenar ---
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# --- 5. Evaluar ---
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# --- 6. Reporte de métricas ---
print(classification_report(y_true, y_pred_classes, target_names=le.classes_))

# --- 7. Matriz de Confusión ---
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.title("Matriz de Confusión - Red Neuronal")
plt.show()
