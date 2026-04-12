### 1. The "Human-in-the-Loop" Features (Crucial for Medical AI & MLOps)
This is the most important feature set for an MLOps thesis. It proves you understand the lifecycle of data.

*   **Interactive Landmark Adjustment:** The AI predicts the 8 facial or 19+ cephalometric points. The frontend plots them on a Canvas. The dentist must be able to **click and drag** any point that the AI got slightly wrong to its correct position. (It looks like you are already planning this in `ui-tooth-ai`).
*   **The Data Flywheel (Active Learning):** When a dentist clicks "Save/Approve Analysis", the system takes the *corrected* coordinates and saves them to your database (PostgreSQL/MinIO) flagged as `verified_ground_truth`. 
*   **Automated Retraining Trigger:** Once you collect, say, 50 new corrected images, your system triggers an MLflow/Airflow pipeline to fine-tune the HRNet model on the new data, creating a smarter "Version 2" of the model. This is the holy grail of MLOps!

### 2. Clinical & Business Logic Features (The Product)
These features make your application a real "product" rather than just a machine learning demo.

*   **Patient & Record Management:** 
    *   Create patient profiles (Name, Age, ID).
    *   Upload and store multiple records over time (Pre-treatment X-ray vs. Post-treatment X-ray).
*   **Automated Cephalometric Diagnostics (Math Engine):** 
    *   Don't just show the points. Use the `core/` folder in your frontend (or calculate it in the backend) to automatically compute standard orthodontic angles.
    *   *Examples:* **SNA angle** (maxilla to cranial base), **SNB angle** (mandible to cranial base), and **ANB angle** (maxilla to mandible relationship).
    *   Provide a clinical diagnosis based on these angles (e.g., "Class II Malocclusion").
*   **Automated Report Generation:** A button to export the analysis (Images with points drawn + Tables of Angles/Distances + AI Diagnosis) into a beautiful PDF format that the dentist can hand to the patient.

### 3. Engineering & System Monitoring Features (The "Wow" Factor)
These features will impress your grading panel and show your software engineering maturity.

*   **AI Confidence Heatmaps:** Instead of just returning `(x, y)` coordinates, have your FastAPI return the Gaussian heatmap or a "Confidence Score" (0-100%) for each point. If the AI is only 40% confident about the "Glabella," highlight that point in **red** on the frontend so the dentist knows to check it carefully.
*   **Model Version Selector:** In the UI settings, allow the admin to switch between AI models (e.g., `HRNet-W18-Fast` vs `HRNet-W32-Accurate`, or `v1.0` vs `v2.0-finetuned`). This demonstrates your Model Registry works.
*   **System Health Dashboard:** A separate admin page (or Grafana dashboard) showing:
    *   Total images analyzed today.
    *   Average AI processing time (e.g., 120ms).
    *   How many times dentists had to manually correct the AI (Error rate tracking).

### How to Prioritize (MVP Plan):

**Must-Have (Your Minimum Viable Product):**
1. Upload X-Ray / Photo.
2. AI predicts points.
3. Frontend draws points.
4. Dentist can drag/adjust points.
5. System displays calculated angles/distances based on the points.

**Nice-to-Have (Do this after the MVP is working):**
1. Patient Profile Management.
2. PDF Report Export.
3. Saving the adjusted points to the database (Data Flywheel).

**Extra Credit (If you have time before graduation):**
1. CI/CD pipeline.
2. Confidence score heatmaps.
3. Automated retraining pipeline.