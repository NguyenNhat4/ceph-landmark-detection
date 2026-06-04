backend/
├── app/
│   ├── api/            # Định nghĩa các endpoints (routers)
│   │   ├── v1/
│   │   │   ├── inference.py  # API /predict, /feedback
│   │   │   └── patient.py    # API quản lý bệnh nhân, X-quang
│   ├── core/           # Cấu hình chung, security, error handling
│   │   ├── config.py         # Đọc biến môi trường bằng Pydantic BaseSettings
│   │   └── lifespan.py       # Quản lý event startup/shutdown (Load Model)
│   ├── db/             # Kết nối database và các ORM models
│   │   ├── database.py       # Khởi tạo connection pool (SQLAlchemy)
│   │   └── models.py         # Định nghĩa các bảng patients, images...
│   ├── ml/             # Logic mô hình AI (Độc lập với Web framework)
│   │   └── landmark_detector.py # Class bọc ONNX Runtime/PyTorch
│   ├── schemas/        # Pydantic models (Validate request/response)
│   │   └── landmark.py       # VD: Tọa độ JSON (x, y)
│   ├── services/       # Logic nghiệp vụ xử lý dữ liệu (Business Logic)
│   │   ├── storage.py        # Tương tác với GCS (Tạo Signed URL)
│   │   └── analysis.py       # Xử lý luồng lưu DB sau khi inference
│   └── main.py         # Entry point khởi tạo FastAPI app
├── requirements.txt
└── Dockerfile