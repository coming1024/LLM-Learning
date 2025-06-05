import subprocess
import sys

# ---- 自动安装依赖 ----
required_packages = ["google-genai", "faiss-cpu", "numpy"]

for pkg in required_packages:
    try:
        __import__(pkg.split("-")[0])  # 简单尝试导入
    except ImportError:
        print(f"正在安装依赖库: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ---- 进入主程序 ----
import tkinter as tk
from tkinter import filedialog
import faiss
import numpy as np
from google import genai

API_KEY = "AIzaSyBetoZgp-M1JAl0knqqGFHAlOAZKNgRimw"

client = genai.Client(api_key=API_KEY)

def select_file():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="请选择希望AI回答相关问题的的文本文件",
        filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
    )
    return file_path

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500):
    # 简单按字符切分，500字符一块，可以根据需要调整
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_texts(text_chunks):
    embeddings = []
    print(f"开始为 {len(text_chunks)} 个文本块生成向量...")
    for i, chunk in enumerate(text_chunks):
        res = client.models.embed_content(
            model="models/embedding-001",
            contents=chunk
        )
        embeddings.append(res.embeddings[0].values)
        if (i + 1) % 10 == 0 or i == len(text_chunks) - 1:
            print(f"已完成 {i + 1}/{len(text_chunks)} 个块的向量生成")
    return embeddings

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    print(f"FAISS 索引构建完成，包含 {index.ntotal} 个向量")
    return index

def embed_query(query):
    res = client.models.embed_content(
        model="models/embedding-001",
        contents=query
    )
    return res.embeddings[0].values

def retrieve_similar_chunks(query_vec, index, chunks, top_k=3):
    D, I = index.search(np.array([query_vec]).astype("float32"), top_k)
    results = [chunks[i] for i in I[0]]
    print(f"检索到 {len(results)} 个相关文本块")
    return results

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"以下是相关文档内容：\n{context}\n请基于上述内容回答问题：{query}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

def main():
    print("请选择一个文本文件...")
    file_path = select_file()
    if not file_path:
        print("未选择文件，程序退出")
        return
    print(f"你选择的文件路径是：{file_path}")

    text = read_file(file_path)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)
    print(chunks)
    print(embeddings)

    print("索引构建完成，可以开始提问，输入 exit 退出。")
    while True:
        query = input("请输入你的问题：")
        if query.strip().lower() == "exit":
            print("退出程序")
            break
        query_vec = embed_query(query)
        relevant_chunks = retrieve_similar_chunks(query_vec, index, chunks, top_k=3)
        answer = generate_answer(query, relevant_chunks)
        print("回答：", answer)
        print("-" * 50)

if __name__ == "__main__":
    main()

# 输入文本 - 切分文档得到chunks - 得到每个chunks对应的embedding - 构建faiss索引 - 用户输入query -
# 把query转变为chunk - 通过faiss索引找到相关的chunk - 把query和chunk进行拼接向大模型提问 - 得到答案