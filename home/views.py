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
# from django.utils.timezone import now
# from datetime import datetime

logger = logging.getLogger('django')
# Mô hình Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def process_new_documents():
    # Lấy các tài liệu chưa xử lý
    unprocessed_docs = Document.objects.filter(is_processed=False)

    for doc in unprocessed_docs:
        file_path = doc.document.path
        text_content = extract_text_from_pdf(file_path)  # Trích xuất nội dung từ file
        logger.error(1)
        # Chia nhỏ thành các đoạn
        chunks = split_text_into_chunks(text_content)
        # Tạo embedding cho các đoạn văn bản
        chunk_embeddings = np.array(embedding_model.encode(chunks))
        # logger.error("chunks: %s, chunks_emb: %s",chunks, chunk_embeddings)

        # Lưu thông tin vào bảng ProcessedDocument, lưu chunk_embeddings thay vì faiss_index
        ProcessedDocument.objects.create(
            file_name=doc,
            text_content=text_content,
            embeddings=pickle.dumps(chunk_embeddings), # Lưu chunk_embeddings
            document=doc
        )
        # Đánh dấu tài liệu đã xử lý
        doc.is_processed = True
        doc.save()
        logger.error(3)


def making_context(question):
    processed_docs = ProcessedDocument.objects.all()

    dimension = embedding_model.encode(["sample"]).shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    all_chunks = []
    all_embeddings = [] # Thêm danh sách để lưu trữ embeddings

    for doc in processed_docs:
        if doc.embeddings:
            try:
                embeddings = pickle.loads(doc.embeddings)
                if isinstance(embeddings, np.ndarray): # Kiểm tra xem embeddings có phải là mảng NumPy không
                    faiss_index.add(embeddings)
                    all_chunks.extend(split_text_into_chunks(doc.text_content))
                    all_embeddings.append(embeddings) # Lưu trữ embeddings
                else:
                    print(f"doc.embeddings không phải là mảng NumPy: {type(embeddings)}")
            except pickle.UnpicklingError:
                print(f"Lỗi giải mã pickle cho doc.embeddings")

    if faiss_index.ntotal == 0:
        print("FAISS index rỗng. Không có dữ liệu để tìm kiếm.")
        return ""

    question_embedding = embedding_model.encode([question])
    distances, top_indices = faiss_index.search(question_embedding, 5)

    if top_indices.shape[1] == 0:
        print("Không tìm thấy kết quả phù hợp.")
        return ""

    relevant_chunks = [all_chunks[i] for i in top_indices[0]]
    return " ".join(relevant_chunks)


def chatGoD(request):
    history = request.session.get("chat_history", [])
    if request.method == "POST":
        if "clear_history" in request.POST:
            request.session.pop("chat_history", None)
            Answer.objects.all().delete()
            return render(request, 'home/chatGoD.html', {"answer": None})

        question = request.POST.get("question", "")
        if question:
            context = making_context(question)
            print("context:", context)
            answer = asking(question, context, history)
            history.append((question, answer))
            request.session["chat_history"] = history

            form_data = {
                "ask_content": question,
                "answer_content": answer
            }
            form = AnswerForm(form_data)
            if form.is_valid():
                ask = form.save(commit=False)
                if request.user.is_authenticated:
                    ask.uploaded_by = request.user
                else:
                    messages.error(request, "Bạn cần đăng nhập để tiếp tục.")
                    return redirect('login')
                ask.save()

    return render(request, 'home/chatGoD.html', {"answer": Answer.objects.last()})


def admin_check(user):
    return user.is_staff


@user_passes_test(admin_check, login_url='home')
def upload(request):
    if request.method == 'POST':
        if "delete_document" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                if document.document and os.path.exists(document.document.path):
                    os.remove(document.document.path)
                document.delete()
                messages.success(request, "Tài liệu đã được xóa thành công!")
            except Exception as e:
                messages.error(request, "Có lỗi khi xóa tài liệu.")
            return redirect('upload')

        if "update_note" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                document.description = request.POST.get("input-req")
                document.save()
                messages.success(request, "Cập nhật mô tả thành công!")
            except Exception as e:
                messages.error(request, "Có lỗi khi cập nhật mô tả.")
            return redirect('upload')

        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            try:
                document.uploaded_by = request.user
                document.save()
                process_new_documents()
                messages.success(request, "Tải lên thành công!")
            except Exception as e:
                messages.error(request, "Có lỗi khi tải lên.")
                logger.error("e:", e)
                document.delete()
                os.remove(document.document.path)
            return redirect('upload')

    documents = Document.objects.all()
    return render(request, 'admin/uploadManage.html', {'documents': documents})


def select_files(request):
    return render(request, 'home/select_files.html')


def logout_view(request):
    logout(request)
    messages.success(request, 'Đăng xuất thành công.')
    return redirect('login')


@login_required(login_url='login')
@user_passes_test(admin_check, login_url='home')
def account(request):
    users = User.objects.all()
    if "delete_account" in request.POST:
        account_id = request.POST.get("id")  # Lấy ID từ form
        acc = get_object_or_404(User, id=account_id)
        acc.delete()  # Xóa tài liệu trong database
        messages.success(request, "Tài khoản đã được xóa thành công!")
        return redirect('account')
    if "update_auth" in request.POST:
        try:
            account_id = request.POST.get("id")
            account = get_object_or_404(User, id=account_id)
            if request.POST.get("newauth") == "Superadmin":
                account.is_superuser = True
                account.is_staff = True
            elif request.POST.get("newauth") == "Staff":
                account.is_superuser = False
                account.is_staff = True
            else:
                account.is_superuser = False
                account.is_staff = False
            account.save()
            messages.success(request, "Cập nhật thành công!")
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật mô tả: {e}")
            messages.error(request, "Có lỗi xảy ra khi cập nhật mô tả. Vui lòng thử lại!")
        return redirect('account')
    return render(request, 'admin/account.html', {'users': users})


def admin_base(request):
    users = User.objects.all()
    return render(request, 'admin/adminBase.html', {'users': users})


def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        if password != confirm_password:
            messages.error(request, 'Mật khẩu không khớp!')
            return render(request, 'home/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Tên tài khoản đã tồn tại!')
            return render(request, 'home/register.html')

        try:
            validate_password(password)
        except ValidationError as e:
            messages.error(request, e)
            return render(request, 'home/register.html')

        user = User.objects.create_user(username=username, password=password)
        user.save()
        messages.success(request, 'Đăng ký thành công!')
        return redirect('login')

    return render(request, 'home/register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # messages.success(request, f'Chào mừng, {user.username}! Đăng nhập thành công!')
            return redirect('account')
        else:
            messages.error(request, 'Tên tài khoản hoặc mật khẩu không đúng.')

    return render(request, 'home/login.html')
