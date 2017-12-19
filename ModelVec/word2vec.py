import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("/usr/vectors.bin", binary=True,unicode_errors="ignore")

# s=model.wmdistance("今天天气不错","今天天气还可以")
# print(s)

# s1=model.most_similar_cosmul(positive=["中国",'北京'],negative=["美国"])
# print(s1)

vec=model["美国"]-model["首都"]
s1=model.similar_by_vector(vec)
print(s1)