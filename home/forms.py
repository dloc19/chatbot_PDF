from django import forms
from home.models import Document, Answer
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document', )


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ('ask_content', 'answer_content')





