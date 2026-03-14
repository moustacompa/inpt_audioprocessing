import numpy as np
import pandas as pd
import os
import librosa
from sentence_transformers import SentenceTransformer 

tmp = pd.read_csv("data/documents/ss-corpus-fr.tsv", sep="\t")
data = tmp[tmp["transcription"].notna()]
#print(data.head())
#print(data.info())

audio_folder = os.path.join("data", "audio_queries")
#row = data.iloc[2]
#audio_path = os.path.join(audio_folder, row["audio_file"])[:-3]+"wav"
#sentence = row["transcription"]
#print("audio_path"+audio_path)
#print(sentence)
#audio, sr = librosa.load(audio_path, sr=16000)
sentences = data["transcription"].tolist()
#print(sentences)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
#print(embeddings.shape)
np.save(os.path.join("embeddings","text_embeddings.npy"), embeddings)

def getSentences():
    tmp = pd.read_csv("data/documents/ss-corpus-fr.tsv", sep="\t")
    data = tmp[tmp["transcription"].notna()]
    sentences = data["transcription"].tolist()
    return sentences