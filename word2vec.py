from gensim.models import word2vec

# reference_url : https://www.cnblogs.com/pinard/p/7278324.html


def main():
    """
        num_features = 300  # Word vector dimensionality
    min_word_count = 10  # Minimum word count
    num_workers = 16  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    sentences = word2vec.Text8Corpus("seg201708.txt")

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save("model201708")

    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('/tmp/mymodel')
    # model.train(more_sentences)
    :return:
    """
    sentences = word2vec.LineSentence('./cut.txt')

    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
    req_count = 5
    similar_name = []
    print("Similarity with 李达康 ::")
    for key in model.wv.similar_by_word('李达康', topn=100):
        if len(key[0]) == 3:
            req_count -= 1
            print(key[0], key[1], model.wv[key[0]])    # vector = model.wv['computer']  # numpy vector of a word
            similar_name.append(key[0])
            if req_count == 0:
                break
    print(model.wv.vocab['人民'])
    print(len(model.wv.vocab))


if __name__ == "__main__":
    main()
