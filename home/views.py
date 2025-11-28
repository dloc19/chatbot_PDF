from django.shortcuts import render, redirect, get_object_or_404
from home.models import Document, Answer, ProcessedDocument
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from home.forms import DocumentForm, AnswerForm
from home.rag import split_text_into_chunks, asking, extract_text_from_pdf
import logging
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

logger = logging.getLogger(__name__)

# Mô hình Sentence Transformer cho embedding
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHUNK_SIZE = 1000  # Kích thước chunk cho split_text_into_chunks


def process_new_documents():
    """
    Xử lý các tài liệu PDF chưa được xử lý:
    - Trích xuất văn bản từ PDF
    - Chia thành chunks
    - Tạo embeddings
    - Lưu vào database
    """
    try:
        unprocessed_docs = Document.objects.filter(is_processed=False)
        
        if not unprocessed_docs.exists():
            logger.info("Không có tài liệu chưa xử lý.")
            return

        for doc in unprocessed_docs:
            try:
                logger.info(f"Đang xử lý tài liệu: {doc.description or doc.document.name}")
                
                file_path = doc.document.path
                text_content = extract_text_from_pdf(file_path)
                
                if not text_content:
                    logger.warning(f"Không thể trích xuất văn bản từ {doc.document.name}")
                    continue
                
                # Chia thành chunks (sử dụng CHUNK_SIZE constant)
                chunks = split_text_into_chunks(text_content, chunk_size=CHUNK_SIZE)
                
                if not chunks:
                    logger.warning(f"Không có chunks hợp lệ cho {doc.document.name}")
                    continue
                
                # Tạo embeddings
                chunk_embeddings = np.array(EMBEDDING_MODEL.encode(chunks))
                
                # Lưu vào ProcessedDocument
                ProcessedDocument.objects.create(
                    file_name=doc.description or doc.document.name,
                    text_content=text_content,
                    embeddings=pickle.dumps(chunk_embeddings),
                    document=doc
                )
                
                # Đánh dấu tài liệu đã xử lý
                doc.is_processed = True
                doc.save()
                
                logger.info(f"Xử lý thành công tài liệu: {doc.description or doc.document.name}")
                
            except Exception as e:
                logger.error(f"Lỗi xử lý tài liệu {doc.id}: {e}")
                
    except Exception as e:
        logger.error(f"Lỗi trong process_new_documents: {e}")


def making_context(question):
    """
    Tạo context cho câu hỏi bằng cách:
    - Load tất cả embeddings từ ProcessedDocument
    - Xây dựng FAISS index
    - Tìm top-5 chunks liên quan nhất
    - Ghép nó thành một chuỗi context
    
    Args:
        question: Câu hỏi của user
        
    Returns:
        Chuỗi context ghép từ các chunks liên quan
    """
    try:
        processed_docs = ProcessedDocument.objects.all()
        
        if not processed_docs.exists():
            logger.warning("Không có tài liệu đã xử lý. Trả về context rỗng.")
            return ""

        # Khởi tạo FAISS index
        dimension = EMBEDDING_MODEL.encode(["sample"]).shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)

        all_chunks = []

        # Load embeddings từ database
        for doc in processed_docs:
            if not doc.embeddings:
                continue
                
            try:
                embeddings = pickle.loads(doc.embeddings)
                
                if not isinstance(embeddings, np.ndarray):
                    logger.warning(f"Embeddings của doc {doc.id} không phải numpy array. Bỏ qua.")
                    continue
                
                # Thêm vào FAISS index
                faiss_index.add(embeddings)
                
                # Lấy chunks từ text_content (phải dùng cùng CHUNK_SIZE)
                chunks = split_text_into_chunks(doc.text_content, chunk_size=CHUNK_SIZE)
                all_chunks.extend(chunks)
                
            except pickle.UnpicklingError as e:
                logger.error(f"Lỗi giải mã pickle cho doc {doc.id}: {e}")

        if faiss_index.ntotal == 0:
            logger.warning("FAISS index rỗng. Không có dữ liệu để tìm kiếm.")
            return ""

        # Tìm kiếm top-5 chunks liên quan
        question_embedding = EMBEDDING_MODEL.encode([question])
        distances, top_indices = faiss_index.search(question_embedding, k=5)

        if top_indices.shape[1] == 0:
            logger.warning(f"Không tìm thấy chunks liên quan cho câu hỏi: {question}")
            return ""

        # Ghép các chunks liên quan
        relevant_chunks = [all_chunks[i] for i in top_indices[0] if i < len(all_chunks)]
        context = " ".join(relevant_chunks)
        
        logger.info(f"Tạo context thành công từ {len(relevant_chunks)} chunks")
        return context
        
    except Exception as e:
        logger.error(f"Lỗi trong making_context: {e}")
        return ""


def chatGoD(request):
    """
    Xử lý trang chat chính:
    - Lấy/duy trì lịch sử hội thoại trong session
    - Xử lý câu hỏi: tạo context, gọi AI, lưu answer
    - Hỗ trợ xóa lịch sử
    """
    history = request.session.get("chat_history", [])
    
    if request.method == "POST":
        if "clear_history" in request.POST:
            request.session.pop("chat_history", None)
            if request.user.is_authenticated:
                Answer.objects.filter(uploaded_by=request.user).delete()
            return render(request, 'home/chatGoD.html', {"answer": None})

        if "delete_answer" in request.POST:
            try:
                answer_id = request.POST.get("id")
                answer = get_object_or_404(Answer, id=answer_id)
                
                if answer.uploaded_by == request.user or request.user.is_staff:
                    answer.delete()
                    messages.success(request, "Xóa lịch sử hội thoại thành công!")
                    logger.info(f"User {request.user.username} xóa answer {answer_id}")
                else:
                    messages.error(request, "Bạn không có quyền xóa lịch sử này.")
                    logger.warning(f"User {request.user.username} cố xóa answer của user khác")
                    
            except Exception as e:
                logger.error(f"Lỗi xóa answer: {e}")
                messages.error(request, "Có lỗi khi xóa lịch sử.")
            return redirect('home')

        question = request.POST.get("question", "").strip()
        if not question:
            messages.warning(request, "Vui lòng nhập một câu hỏi.")
            return render(request, 'home/chatGoD.html', {"answer": Answer.objects.last()})

        try:
            # Tạo context từ documents
            context = making_context(question)
            
            # Gọi AI để tạo câu trả lời
            answer_text = asking(question, context, history)
            
            if not answer_text:
                messages.error(request, "Không thể tạo câu trả lời.")
                return render(request, 'home/chatGoD.html', {"answer": Answer.objects.last()})
            
            # Cập nhật lịch sử
            history.append((question, answer_text))
            request.session["chat_history"] = history

            # Lưu vào database nếu user đã đăng nhập
            if request.user.is_authenticated:
                answer_obj = Answer(
                    ask_content=question,
                    answer_content=answer_text,
                    context=context,
                    uploaded_by=request.user
                )
                answer_obj.save()
                logger.info(f"Lưu answer cho user {request.user.username} (answer_id={answer_obj.id})")
            else:
                messages.warning(request, "Bạn cần đăng nhập để lưu lịch sử trò chuyện.")
                
        except Exception as e:
            logger.error(f"Lỗi trong chatGoD: {e}")
            messages.error(request, f"Lỗi: {str(e)}")

    # Lấy lịch sử hội thoại của user nếu đã đăng nhập
    if request.user.is_authenticated:
        answers = Answer.objects.filter(uploaded_by=request.user).order_by('-ask_at')[:10]
    else:
        answers = []

    return render(request, 'home/chatGoD.html', {"answer": Answer.objects.last(), "answers": answers})


def admin_check(user):
    return user.is_staff


@user_passes_test(admin_check, login_url='home')
def upload(request):
    """
    Xử lý upload PDF:
    - Upload tài liệu mới (trigger process_new_documents)
    - Xóa tài liệu
    - Cập nhật mô tả tài liệu
    """
    if request.method == 'POST':
        if "delete_document" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                
                # Xóa file khỏi hệ thống
                if document.document and os.path.exists(document.document.path):
                    try:
                        os.remove(document.document.path)
                        logger.info(f"Xóa file: {document.document.path}")
                    except Exception as e:
                        logger.warning(f"Không thể xóa file {document.document.path}: {e}")
                
                # Xóa bản ghi
                document.delete()
                # Cũng xóa ProcessedDocument tương ứng (automatic via CASCADE)
                
                messages.success(request, "Tài liệu đã được xóa thành công!")
                logger.info(f"Xóa document {document_id}")
                
            except Exception as e:
                logger.error(f"Lỗi xóa tài liệu: {e}")
                messages.error(request, "Có lỗi khi xóa tài liệu.")
            return redirect('upload')

        if "update_note" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                new_description = request.POST.get("input-req", "").strip()
                
                document.description = new_description
                document.save()
                
                messages.success(request, "Cập nhật mô tả thành công!")
                logger.info(f"Cập nhật mô tả document {document_id}")
                
            except Exception as e:
                logger.error(f"Lỗi cập nhật mô tả: {e}")
                messages.error(request, "Có lỗi khi cập nhật mô tả.")
            return redirect('upload')

        # Xử lý upload PDF mới
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            try:
                document.uploaded_by = request.user
                document.save()
                
                # Xử lý document mới (trích xuất, embedding, etc.)
                process_new_documents()
                
                messages.success(request, "Tải lên thành công!")
                logger.info(f"Upload document {document.id} thành công")
                
            except Exception as e:
                logger.error(f"Lỗi upload: {e}")
                messages.error(request, f"Có lỗi khi tải lên: {str(e)}")
                
                # Rollback: xóa document và file
                try:
                    if document.document and os.path.exists(document.document.path):
                        os.remove(document.document.path)
                    document.delete()
                except Exception as cleanup_e:
                    logger.error(f"Lỗi cleanup sau upload failed: {cleanup_e}")
                    
            return redirect('upload')

    # GET request: hiển thị danh sách tài liệu
    documents = Document.objects.all().order_by('-uploaded_at')
    return render(request, 'admin/uploadManage.html', {'documents': documents})


def logout_view(request):
    """Đăng xuất."""
    logout(request)
    logger.info(f"User đăng xuất")
    messages.success(request, 'Đăng xuất thành công.')
    return redirect('login')


def select_files(request):
    """Trang chọn files (placeholder)."""
    return render(request, 'home/select_files.html')


def admin_base(request):
    """Trang admin dashboard."""
    users = User.objects.all()
    total_docs = Document.objects.count()
    processed_docs = Document.objects.filter(is_processed=True).count()
    
    context = {
        'users': users,
        'total_docs': total_docs,
        'processed_docs': processed_docs,
    }
    return render(request, 'admin/adminBase.html', context)


@login_required(login_url='login')
@user_passes_test(admin_check, login_url='home')
def account(request):
    """
    Quản lý tài khoản user:
    - Xóa tài khoản
    - Cập nhật quyền (user/staff/superadmin)
    """
    users = User.objects.all().order_by('-date_joined')
    
    if "delete_account" in request.POST:
        try:
            account_id = request.POST.get("id")
            account = get_object_or_404(User, id=account_id)
            username = account.username
            
            account.delete()
            
            messages.success(request, f"Tài khoản {username} đã được xóa thành công!")
            logger.info(f"Xóa user account {username} (id={account_id})")
            
        except Exception as e:
            logger.error(f"Lỗi xóa account: {e}")
            messages.error(request, "Có lỗi xảy ra khi xóa tài khoản.")
        return redirect('account')
        
    if "update_auth" in request.POST:
        try:
            account_id = request.POST.get("id")
            account = get_object_or_404(User, id=account_id)
            new_role = request.POST.get("newauth", "").strip()
            
            if new_role == "Superadmin":
                account.is_superuser = True
                account.is_staff = True
            elif new_role == "Staff":
                account.is_superuser = False
                account.is_staff = True
            else:
                account.is_superuser = False
                account.is_staff = False
            
            account.save()
            messages.success(request, f"Cập nhật quyền cho {account.username} thành công!")
            logger.info(f"Cập nhật quyền user {account.username} → {new_role}")
            
        except Exception as e:
            logger.error(f"Lỗi cập nhật quyền: {e}")
            messages.error(request, "Có lỗi xảy ra khi cập nhật quyền. Vui lòng thử lại!")
        return redirect('account')
    
    return render(request, 'admin/account.html', {'users': users})


def admin_base(request):
    users = User.objects.all()
    return render(request, 'admin/adminBase.html', {'users': users})


def register_view(request):
    """
    Xử lý đăng ký tài khoản mới.
    """
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        
        # Kiểm tra mật khẩu khớp
        if password != confirm_password:
            messages.error(request, 'Mật khẩu không khớp!')
            return render(request, 'home/register.html')

        # Kiểm tra username đã tồn tại
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Tên tài khoản đã tồn tại!')
            return render(request, 'home/register.html')

        # Validate mật khẩu
        try:
            validate_password(password)
        except ValidationError as e:
            messages.error(request, f'Mật khẩu không hợp lệ: {", ".join(e.messages)}')
            return render(request, 'home/register.html')

        try:
            # Tạo user mới
            user = User.objects.create_user(username=username, password=password)
            logger.info(f"Đăng ký thành công user: {username}")
            
            messages.success(request, 'Đăng ký thành công! Bạn có thể đăng nhập ngay.')
            return redirect('login')
        except Exception as e:
            logger.error(f"Lỗi tạo user: {e}")
            messages.error(request, f'Lỗi tạo tài khoản: {str(e)}')
            return render(request, 'home/register.html')

    return render(request, 'home/register.html')


def login_view(request):
    """
    Xử lý đăng nhập.
    """
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            logger.info(f"User {username} đăng nhập thành công")
            
            # Redirect dựa trên quyền
            if user.is_staff:
                return redirect('account')
            else:
                return redirect('home')
        else:
            logger.warning(f"Đăng nhập thất bại cho username: {username}")
            messages.error(request, 'Tên tài khoản hoặc mật khẩu không đúng.')

    return render(request, 'home/login.html')
