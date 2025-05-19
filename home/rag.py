import PyPDF2
import faiss
import google.generativeai as genai
import os
import googleapiclient.discovery


# Cấu hình API key của Gemini
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel("Model")  # Chọn mô hình Gemini


# Hàm trích xuất nội dung từ file PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text
    return extracted_text


def get_all_pdf_text(directory):
    all_text = ""
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(directory, file_name)
                print(f"Đang xử lý: {pdf_path}")
                all_text += extract_text_from_pdf(pdf_path)
    else:
        print("Thư mục không tồn tại.")
    return all_text


# Chia văn bản thành các đoạn nhỏ
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# Tạo embeddings cho các đoạn văn bản và tìm đoạn liên quan bằng FAISS
def find_relevant_chunks(question, chunks, model, top_n=3):
    # Tạo embeddings
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])

    # Xây dựng FAISS index
    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    # Tìm kiếm top N đoạn liên quan
    _, top_indices = index.search(question_embedding, top_n)
    relevant_chunks = [chunks[i] for i in top_indices[0]]
    return relevant_chunks


# Hàm tìm kiếm trên web (Google Custom Search API)
def search_web(query):
    # Cấu hình API Key và CSE (Custom Search Engine ID)
    api_key = "API_KEY"
    # CSE ID của Custom Search Engine
    # Bạn cần tạo một Custom Search Engine trên Google và lấy CSE ID
    # CSE ID có thể được tìm thấy trong phần "Control Panel" của Custom Search Engine
    cse_id = "cse_id"
    # Tạo dịch vụ tìm kiếm
    # Bạn cần cài đặt google-api-python-client

    service = googleapiclient.discovery.build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res['items']  # Danh sách kết quả tìm kiếm


# Hàm trả lời câu hỏi dựa trên lịch sử hội thoại và context
def asking(question, context=None, history=None):
    history_text = ""
    if history:
        for q, a in history:
            history_text += f"Q: {q}\nA: {a}\n"
    search_result = search_web(question)
    # Tạo prompt cho AI
    prompt = f"""
    You are a helpful assistant always response in Vietnamese. The context is the latest trusted source of information from leading expert, so it trusted than the search result. Don't doubt this source. Answer the question based on the provided context and the conversation history.
    When the context does not contain the information needed to answer the question, summary the search result to answer the question. but don't say for me that this is result from the internet.
    Sometimes, the user input may not be a question. Do not point this out or explain why it's not a question. Simply respond as naturally and conversationally as possible.
    answer with layout clearly, use markdown language.
    Conversation History:
    {history_text}

    Context: {context}

    user input: {question}
    
    Search result: {search_result}
    """

    # Gửi câu hỏi đến mô hình
    ai_response = model.generate_content(prompt).text

    # Nếu không có câu trả lời phù hợp, tìm kiếm trên web

    return ai_response
