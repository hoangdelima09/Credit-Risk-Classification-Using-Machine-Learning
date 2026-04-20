# Credit Risk Classification Using Machine Learning

## Giới thiệu dự án
Dự án này là **Bài tập lớn môn Khai phá dữ liệu (CO3029)** tại **Trường Đại học Bách Khoa - ĐHQG TP.HCM**.

Mục tiêu chính là xây dựng một hệ thống học máy có khả năng **dự đoán xác suất khách hàng vỡ nợ (loan default)** dựa trên dữ liệu lịch sử từ **Lending Club**.

Bài toán được tiếp cận dưới dạng:
- **Phân loại nhị phân (Binary Classification)**

Kết quả của mô hình giúp:
- Tối ưu hóa quy trình xét duyệt khoản vay  
- Giảm thiểu rủi ro tín dụng cho tổ chức tài chính  

---

## Thành viên thực hiện
- **Đặng Vũ Anh Khoa (MSSV: 2311578)**  
  - Tiền xử lý  
  - EDA  
  - Viết báo cáo  

- **Nguyễn Trần Đức Hoàng (MSSV: 2311064)**  
  - Modeling  
  - Viết báo cáo  
  - Slide  

- **Giảng viên hướng dẫn:** Đỗ Thanh Thái  

---

## Dữ liệu sử dụng
- **Nguồn:** Lending Club Loan Data (Kaggle)  
- **Quy mô:** > 600.000 bản ghi, 105 thuộc tính ban đầu  

### Các nhóm thông tin chính:
- Đặc điểm khoản vay  
- Thông tin người vay  
- Lịch sử tín dụng  
- Mục đích vay vốn  

---

## Quy trình thực hiện (KDD Process)

### 1. Tiền xử lý dữ liệu (Data Preprocessing)
- Loại bỏ các biến có tỷ lệ thiếu > 70%  
- Loại bỏ các biến không liên quan  

#### Data Leakage Handling:
Loại bỏ các biến chứa thông tin phát sinh sau khi giải ngân:
- `recoveries`  
- `total_rec_int`  
- `last_pymnt_d`  

- Xử lý giá trị thiếu:
  - **Median** (số)
  - **Mode** (categorical)

---

### 2. Phân tích khám phá dữ liệu (EDA)
- Trực quan hóa mối quan hệ giữa:
  - Grade (hạng tín dụng)
  - Income (thu nhập)
  - DTI (tỷ lệ nợ)

→ Đánh giá ảnh hưởng đến khả năng vỡ nợ  

---

### 3. Feature Engineering
- **Ordinal Encoding:** `grade`, `sub_grade`  
- **One-Hot Encoding:** `home_ownership`, `purpose`  
- **Log Transformation:**  
  - `log_income` (giảm skewness)

---

## Mô hình & Kết quả

### Baseline Model
- **Decision Tree**
- ROC-AUC: **0.7162**

### Advanced Model
- **XGBoost (Gradient Boosting)**
- ROC-AUC: **0.7418**
- Tuning bằng **RandomizedSearchCV**

### Nhận xét
- XGBoost vượt trội hơn Decision Tree  
- Mô hình ổn định hơn  
- Feature quan trọng :
  - **`grade` (Hạng thẻ tín dụng)** 
  - **`int_rate` (lãi suất)**  

---

## Triển khai (Deployment)
Mô hình được triển khai dưới dạng **Dashboard tương tác** sử dụng **Streamlit**.

### Chức năng:
- Nhập thông tin tài chính người vay  
- Dự đoán:
  - Xác suất vỡ nợ  
  - Mức độ rủi ro  

---

## 📂 Cấu trúc thư mục
```
/data # Dataset hoặc link tải
/notebooks # EDA & Feature Engineering
/preprocessing # Scripts tiền xử lý
/models # Training models
Dashboard.py # Streamlit dashboard
README.md
xgboost_dashboard_model.pkl
```

## Setup
**Run Dashboard:**
   ```bash
   streamlit run Dashboard.py
   ```