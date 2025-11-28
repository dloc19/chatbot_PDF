import PyPDF2
import faiss
import google.generativeai as genai
import os
import googleapiclient.discovery
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# Cấu hình API key của Gemini từ environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment. Set it via .env or environment variables.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-1.5-flash (newer, faster model) instead of deprecated gemini-pro
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    logger.warning(f"gemini-1.5-flash not available: {e}. Trying gemini-pro...")
    model = genai.GenerativeModel("gemini-pro")


# Hàm trích xuất nội dung từ file PDF
def extract_text_from_pdf(pdf_file):
    """Trích xuất văn bản từ file PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text
        return extracted_text
    except Exception as e:
        logger.error(f"Lỗi trích xuất PDF: {e}")
        return ""


def get_all_pdf_text(directory):
    """Lấy toàn bộ văn bản từ tất cả PDF trong một thư mục."""
    all_text = ""
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(directory, file_name)
                logger.info(f"Đang xử lý: {pdf_path}")
                all_text += extract_text_from_pdf(pdf_path)
    else:
        logger.warning(f"Thư mục không tồn tại: {directory}")
    return all_text


# Chia văn bản thành các đoạn nhỏ
def split_text_into_chunks(text, chunk_size=1000):
    """
    Chia văn bản thành các đoạn nhỏ (chunks) có kích thước định sẵn.
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước mỗi đoạn (số ký tự)
        
    Returns:
        Danh sách các chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Chỉ lưu chunks không rỗng
            chunks.append(chunk)
    return chunks


# Tạo embeddings cho các đoạn văn bản và tìm đoạn liên quan bằng FAISS
def find_relevant_chunks(question, chunks, model, top_n=3):
    """
    Tìm các chunks liên quan nhất đến câu hỏi bằng FAISS.
    
    Args:
        question: Câu hỏi
        chunks: Danh sách các chunks cần tìm
        model: Mô hình embedding
        top_n: Số lượng chunks hàng đầu cần trả về
        
    Returns:
        Danh sách chunks phù hợp nhất
    """
    try:
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
    except Exception as e:
        logger.error(f"Lỗi tìm chunks liên quan: {e}")
        return []


# Hàm tìm kiếm trên web (Google Custom Search API)
def search_web(query):
    """
    Tìm kiếm trên web bằng Google Custom Search API.
    Nếu API key không hợp lệ, trả về danh sách rỗng (fallback).
    
    Args:
        query: Chuỗi tìm kiếm
        
    Returns:
        Danh sách kết quả tìm kiếm hoặc danh sách rỗng nếu lỗi.
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    # Nếu không có key, skip search
    if not api_key or not cse_id:
        logger.debug("GOOGLE_API_KEY or GOOGLE_CSE_ID not set. Skipping web search.")
        return []
    
    # Nếu key là placeholder (not configured), skip
    if api_key.startswith('your_') or cse_id.startswith('your_'):
        logger.debug("Google API credentials not configured. Skipping web search.")
        return []
    
    try:
        service = googleapiclient.discovery.build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id).execute()
        results = res.get('items', [])
        logger.info(f"Web search found {len(results)} results for: {query}")
        return results
    except Exception as e:
        logger.warning(f"Web search error (API key may be invalid): {e}")
        # Return empty list instead of crashing
        return []


# Hàm trả lời câu hỏi dựa trên lịch sử hội thoại và context
def asking(question, context=None, history=None):
    """
    Tạo câu trả lời bằng Gemini dựa trên context, lịch sử và kết quả tìm kiếm web.
    
    Args:
        question: Câu hỏi của user
        context: Văn bản context liên quan từ documents
        history: Lịch sử hội thoại trước đó
        
    Returns:
        Câu trả lời từ mô hình AI
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not configured. Cannot generate response.")
        return "Lỗi: Chưa cấu hình API key Gemini. Vui lòng kiểm tra file .env."
    
    try:
        history_text = ""
        if history:
            for q, a in history:
                history_text += f"Q: {q}\nA: {a}\n"
        
        # Thử tìm kiếm web (nếu API key hợp lệ)
        search_results = search_web(question)
        search_result_text = ""
        if search_results:
            search_result_text = "\n".join([
                f"- {item.get('title', '')}: {item.get('snippet', '')}" 
                for item in search_results[:3]  # Top 3 results
            ])
        
        # Nếu không có context (không có document), chủ yếu dùng search + general knowledge
        if not context or context.strip() == "":
            logger.warning("No document context available. Using web search + general knowledge.")
            prompt = f"""
You are a helpful Vietnamese assistant answering questions based on general knowledge and web search results.

Conversation History:
{history_text}

Web Search Results:
{search_result_text if search_result_text else "No web results available."}

User Question: {question}

Please provide a helpful answer in Vietnamese. If you don't have enough information, be honest about it."""
        else:
            # Có context từ document - ưu tiên context
            prompt = f"""
You are a helpful Vietnamese assistant. Answer questions based on the provided context first.
The context contains trusted information from uploaded documents and is your primary source.
If context doesn't have enough info, supplement with web search results or general knowledge.
Don't explicitly mention sources - respond naturally.
Format your response clearly using markdown.

Conversation History:
{history_text}

Context from Documents:
{context}

Web Search Results (supplementary):
{search_result_text if search_result_text else "No web results available."}

User Question: {question}

Answer in Vietnamese:"""

        # Gửi câu hỏi đến mô hình
        ai_response = model.generate_content(prompt).text
        logger.info(f"Generated response for question: {question[:50]}...")
        return ai_response
        
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {e}")
        return f"Lỗi: Không thể tạo câu trả lời. {str(e)}"
