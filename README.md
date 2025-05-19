# Django Project README

## **Hướng dẫn chạy dự án**
1. **Khởi động server**:
   - Lệnh chạy server:
     ```bash
     python manage.py runserver
     ```
     - Nếu muốn chỉ định cổng khác (ví dụ: 8080), dùng lệnh:
       ```bash
       python manage.py runserver 8080
       ```
     - Nếu không chỉ định cổng, cổng mặc định là `8000`.

2. **Tạo file migration** (khi có thay đổi trong models):
   ```bash
   python manage.py makemigrations
   ```
3. **Áp dụng migration** (tạo bảng trong cơ sở dữ liệu):
   ```bash
   python manage.py migrate
   ```
4. **Tạo tài khoản admin**:
   ```bash
    python manage.py createsuperuser
   ```
5. **Cài đặt các thư viện còn thiếu** (có thể chạy runserver để biết thư viện thiếu):
   ```bash
   pip install [tên thư viện]
   ```
6. **Cài đặt kết nối database** (sử dụng mysql):
    #### *Thay đổi trong file pythonweb/settings.py*
7. **API key Gemini**: 
    #### *Thay đổi trong file pythonweb/rag.py*