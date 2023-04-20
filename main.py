"""
#N19DCCN094- Le Tuan Khoi
#N19DCCN017- Tran Nguyen Gia Bao
#N19DCCN037 - Phan Cong Dang
"""
import io
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Đọc dữ liệu từ file
with io.open('npl/doc-text', 'r', encoding='utf-8') as f:
    text = f.read()

# Tiền xử lý văn bản
sentences = text.split('/')
doclist = []
for i, sentence in enumerate(sentences):
    # Loại bỏ các ký tự đặc biệt và chuyển đổi chữ in thành chữ thường
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    # Loại bỏ các từ không cần thiết
    sentence = ' '.join(word for word in sentence.split() if word not in ['a', 'an', 'the', 'is', 'are', 'and', 'or', 'of'])
    if sentence:
        # Lưu câu vào danh sách nếu nó không rỗng
        doclist.append(sentence.strip())

# Đọc dữ liệu từ file
with io.open('npl/query-text', 'r', encoding='utf-8') as f:
    text = f.read()
# Tiền xử lý truy van
sentences = text.split('/')
querylist = []
for i, sentence in enumerate(sentences):
    # Loại bỏ các ký tự đặc biệt và chuyển đổi chữ in thành chữ thường
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    # Loại bỏ các từ không cần thiết
    sentence = ' '.join(word for word in sentence.split() if word not in ['a', 'an', 'the', 'is', 'are', 'and', 'or', 'of'])
    if sentence:
        # Lưu câu vào danh sách nếu nó không rỗng
        querylist.append(sentence.strip())
# Đọc dữ liệu từ file
with open('npl/term-vocab', 'r') as f:
    text = f.read()

# Tiền xử lý dữ liệu
sentences = text.split('/')
keywordlist = []
for i, sentence in enumerate(sentences):
    # Loại bỏ các ký tự đặc biệt và chuyển đổi chữ in thành chữ thường
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    sentence = sentence.strip()
    if sentence:
        # Lưu câu vào danh sách nếu nó không rỗng
        keywordlist.append(sentence)
# Tiền xử lý dữ liệu
words = re.findall(r'\b\w+\b', text)
keywordlist = [word.lower() for word in words[1::2] if word.lower() not in ['a', 'an', 'the', 'is', 'are', 'and', 'or', 'of']]





# Khởi tạo đối tượng CountVectorizer với từ điển là dictionary
vectorizer = CountVectorizer(vocabulary=keywordlist)

# Tạo ma trận tần số
X = vectorizer.fit_transform(doclist)

# Khởi tạo đối tượng TfidfTransformer
tfidf_transformer = TfidfTransformer()

# Tính toán giá trị tf-idf
X_tfidf = tfidf_transformer.fit_transform(X)


# Tạo vector biểu diễn của câu truy vấn
query_vecs = tfidf_transformer.transform(vectorizer.transform(querylist))

# Tính toán độ tương đồng giữa câu truy vấn và các câu trong văn bản
similarity_scores = cosine_similarity(query_vecs, X_tfidf)

# Sắp xếp các câu theo độ tương đồng giảm dần và trả về năm câu có giá trị cao nhất tương ứng với từng câu truy vấn
for i in range(len(querylist)):
    print("Câu truy vấn {}".format(querylist[i]))
    ranked_sentences = np.argsort(similarity_scores[i])[::-1][:3]
    for j in ranked_sentences:
        print("   Văn bản  {}".format(doclist[j]))









