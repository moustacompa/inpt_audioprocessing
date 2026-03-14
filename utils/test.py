import faiss_index as fi
import search as s
import create_embeddings as ce

phrases = ce.getSentences()
txt = "Dix-huit heures, tout rond."
rep = s.search(query=txt)
#print(rep)
indices = rep[0][0]
distances = rep[1][0]
for  i in range(len(indices)):
    r = indices[i]
    d = distances[i]
    p = phrases[r]
    print("Resutat "+str(i+1)+" : "+str(r)+"; distance : "+str(d)+"; p="+p)