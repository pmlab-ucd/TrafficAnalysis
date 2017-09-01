from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


ngram_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(6, 8))
counts = ngram_vectorizer.fit_transform(['appverify', 'clientnew', 'www'])
print ngram_vectorizer.get_feature_names()