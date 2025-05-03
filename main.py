import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

#DESCRIÇÃO
#“Como se deu a evolução dos principais temas das manchetes de Economia no Brasil nos últimos 30 dias?”

#COLETA DE DADOS (NewsData.io)
load_dotenv()
api_key = os.getenv("API_KEY")
url = f"https://newsdata.io/api/1/latest?country=br&category=business&apikey={api_key}"
resp = requests.get(url)
resp.raise_for_status()
items = resp.json().get('results', [])
df = pd.DataFrame(items)
df['pubDate'] = pd.to_datetime(df['pubDate'])
now = pd.Timestamp.today()
df = df[df['pubDate'] >= now - pd.Timedelta(days=30)]

#PRÉ-PROCESSAMENTO
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('portuguese'))

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zà-ú\s]', '', txt)
    return ' '.join(w for w in txt.split() if w not in stop_words)

df['clean'] = df['title'].apply(clean_text)

#ANÁLISE EXPLORATÓRIA
print("Total de manchetes:", len(df))
print(df['clean'].str.split().str.len().describe())
wc = WordCloud(width=800, height=400, collocations=False)
wc.generate(' '.join(df['clean']))
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#TAREFA DE MINERAÇÃO DE DADOS
vec = TfidfVectorizer()
X = vec.fit_transform(df['clean'])
km = KMeans(n_clusters=3, random_state=42).fit(X)
df['cluster'] = km.labels_

for c in range(3):
    print(f"\nCluster {c}:")
    print(df[df['cluster']==c]['title'].head(5).to_list())

#VISUALIZAÇÃO DE RESULTADOS
counts = df['cluster'].value_counts().sort_index()
counts.plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Manchetes')
plt.title('Distribuição por Cluster')
plt.show()
df.to_csv('manchetes_economia_clusterizadas.csv', index=False)