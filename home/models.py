from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    is_processed = models.BooleanField(default=False)


class Answer(models.Model):
    answer_at = models.DateTimeField(auto_now_add=True)
    answer_content = models.TextField()
    ask_at = models.DateTimeField(auto_now_add=True)
    ask_content = models.TextField()
    uploaded_file = models.FileField(upload_to='')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)


class ProcessedDocument(models.Model):
    file_name = models.CharField(max_length=255)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    text_content = models.TextField()
    embeddings = models.BinaryField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

