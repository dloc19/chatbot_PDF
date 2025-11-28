from django.contrib import admin
from home.models import Document, Answer, ProcessedDocument


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('description', 'uploaded_by', 'uploaded_at', 'is_processed')
    list_filter = ('is_processed', 'uploaded_at')
    search_fields = ('description',)
    readonly_fields = ('uploaded_at',)
    fieldsets = (
        ("Thông tin cơ bản", {
            'fields': ('description', 'document')
        }),
        ("Xử lý", {
            'fields': ('is_processed',)
        }),
        ("Metadata", {
            'fields': ('uploaded_by', 'uploaded_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ('ask_content_preview', 'uploaded_by', 'ask_at', 'answer_length_preview')
    list_filter = ('ask_at', 'uploaded_by')
    search_fields = ('ask_content', 'answer_content')
    readonly_fields = ('ask_at', 'answer_at')
    fieldsets = (
        ("Câu hỏi", {
            'fields': ('ask_content', 'ask_at', 'uploaded_by')
        }),
        ("Câu trả lời", {
            'fields': ('answer_content', 'answer_at')
        }),
        ("Context (Chunks liên quan)", {
            'fields': ('context',),
            'classes': ('collapse',)
        }),
        ("File tham chiếu", {
            'fields': ('uploaded_file',),
            'classes': ('collapse',)
        }),
    )
    
    def ask_content_preview(self, obj):
        return obj.ask_content[:50] + "..." if len(obj.ask_content) > 50 else obj.ask_content
    ask_content_preview.short_description = "Câu hỏi"
    
    def answer_length_preview(self, obj):
        length = len(obj.answer_content)
        if length > 200:
            return f"Dài ({length} ký tự)"
        elif length > 100:
            return f"Trung bình ({length} ký tự)"
        else:
            return f"Ngắn ({length} ký tự)"
    answer_length_preview.short_description = "Độ dài câu trả lời"


@admin.register(ProcessedDocument)
class ProcessedDocumentAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'document', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('file_name',)
    readonly_fields = ('uploaded_at', 'embeddings')
    fieldsets = (
        ("Thông tin", {
            'fields': ('file_name', 'document')
        }),
        ("Nội dung", {
            'fields': ('text_content',)
        }),
        ("Embeddings (Binary)", {
            'fields': ('embeddings',),
            'classes': ('collapse',)
        }),
        ("Metadata", {
            'fields': ('uploaded_at',),
            'classes': ('collapse',)
        }),
    )
