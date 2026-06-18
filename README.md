Vehicle Speed Detector — Streamlit deployment

Hướng dẫn nhanh để deploy trên Streamlit Cloud

1. Đẩy toàn bộ repository lên GitHub.
2. Trên Streamlit Cloud: tạo app mới và chọn:
   - Repository: (chọn repo của bạn)
   - Branch: (ví dụ main)
   - File path: app/streamlit_app.py
3. Đảm bảo file requirements.txt nằm ở thư mục gốc (đã thêm sẵn) — Streamlit Cloud sẽ cài đặt phụ thuộc tự động.
4. Lưu ý: YOLO (ultralytics) có thể tải model lớn (yolov8n.pt) khi chạy; nếu cần demo nhẹ, báo để tạo chế độ mock (không cần model).

Nếu muốn, sẽ thêm README tiếng Anh hoặc mẫu .github/workflows để CI/deploy tự động.