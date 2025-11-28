from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    """
    Model lưu trữ tài liệu PDF được upload.
    """
    description = models.CharField(max_length=255, blank=True, help_text="Mô tả tài liệu")
    document = models.FileField(upload_to='documents/', help_text="File PDF")
    uploaded_at = models.DateTimeField(auto_now_add=True, help_text="Thời gian upload")
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, help_text="User upload")
    is_processed = models.BooleanField(default=False, help_text="Đã xử lý (trích xuất, embedding)?")

    def __str__(self):
        return self.description or self.document.name

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Tài liệu"
        verbose_name_plural = "Tài liệu"


class Answer(models.Model):
    """
    Model lưu trữ lịch sử Q&A của user.
    """
    ask_content = models.TextField(help_text="Nội dung câu hỏi")
    ask_at = models.DateTimeField(auto_now_add=True, help_text="Thời gian hỏi")
    answer_content = models.TextField(help_text="Nội dung câu trả lời")
    answer_at = models.DateTimeField(auto_now_add=True, help_text="Thời gian trả lời")
    uploaded_file = models.FileField(upload_to='', blank=True, null=True, help_text="File được tham chiếu (optional)")
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, help_text="User đặt câu hỏi")

    def __str__(self):
        return f"Q: {self.ask_content[:50]}... | {self.ask_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['-ask_at']
        verbose_name = "Câu hỏi & Trả lời"
        verbose_name_plural = "Câu hỏi & Trả lời"


class ProcessedDocument(models.Model):
    """
    Model lưu trữ thông tin đã xử lý của Document:
    - Văn bản trích xuất từ PDF
    - Embeddings (pickle serialized numpy array) cho FAISS search
    """
    file_name = models.CharField(max_length=255, help_text="Tên file gốc")
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True, help_text="Liên kết tài liệu gốc")
    text_content = models.TextField(help_text="Nội dung văn bản đã trích xuất")
    embeddings = models.BinaryField(null=True, blank=True, help_text="Embeddings (pickle numpy array)")
    uploaded_at = models.DateTimeField(auto_now_add=True, help_text="Thời gian xử lý")

    def __str__(self):
        return f"Processed: {self.file_name}"

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Tài liệu đã xử lý"
        verbose_name_plural = "Tài liệu đã xử lý"

