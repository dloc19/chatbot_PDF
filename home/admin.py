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
    list_display = ('ask_content_preview', 'uploaded_by', 'ask_at')
    list_filter = ('ask_at', 'uploaded_by')
    search_fields = ('ask_content', 'answer_content')
    readonly_fields = ('ask_at', 'answer_at')
    
    def ask_content_preview(self, obj):
        return obj.ask_content[:50] + "..." if len(obj.ask_content) > 50 else obj.ask_content
    ask_content_preview.short_description = "Câu hỏi"


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
