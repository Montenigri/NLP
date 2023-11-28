import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

nltk.download('stopwords')
nltk.download('punkt')

italianStopWords = stopwords.words("italian")

#Raccolgo un testo online
Testo = requests.get("https://it.wikipedia.org/wiki/Elaborazione_del_linguaggio_naturale").text
#Testo = requests.get("https://it.wikipedia.org/wiki/Speciale:PaginaCasuale").text

testoPulito = []

#Isolo un singolo paragrafo
Testo = Testo.split("<p>")[1].split("</p>")[0]
print(Testo)
lenTesto = len(Testo)
#Divido la stringa in sottostringhe divise dagli spazi
Testo = word_tokenize(Testo)


for parola in Testo:
    if not parola in italianStopWords:
        testoPulito.append(parola)

#Riunisco l'array in una sola stringa  
 
testoPulito = " ".join(testoPulito)

print(f"testo senza stopwords:\n {testoPulito}\n\n")
print(f"differenza di lunghezza {lenTesto - len(testoPulito)}")
