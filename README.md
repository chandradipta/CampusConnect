# CampusConnect — Smart Campus Management Platform

CampusConnect is a full-stack campus management application for students, teachers, and administrators. It combines JWT authentication, role-based workflows, SQL database persistence, NLP-powered complaint classification, facility booking, announcements, notifications, profile management, and admin analytics.

## What was upgraded

- Real frontend ↔ FastAPI integration — no localStorage-only demo workflow.
- JWT authentication with password hashing and token expiry.
- Student, teacher, and admin role-based access.
- Database-backed complaints and status history.
- TF-IDF + Logistic Regression complaint classifier with 150 labeled examples.
- Automatic category, priority, and department routing.
- Facility booking with conflict prevention and cancellation.
- Database-backed announcements and user notifications.
- Admin user management and analytics.
- Profile editing, password update, avatar support, deactivation and deletion.
- Input validation, CORS configuration, `.env.example`, and production-friendly project structure.
- Responsive retro/pixel-inspired UI retained and upgraded for the real API.

## Architecture

```text
CampusConnect/
├── frontend/
│   └── index.html              # Responsive web UI
├── backend/
│   ├── main.py                 # FastAPI API, models, auth and business logic
│   └── requirements.txt
├── ml/
│   ├── dataset.csv             # 150 labeled complaint examples
│   ├── train.py                # Model training/evaluation
│   └── complaint_classifier.joblib
├── .env.example
├── .gitignore
└── README.md
```

## Tech stack

**Frontend:** HTML5, CSS3, JavaScript, Fetch API, responsive UI

**Backend:** Python, FastAPI, SQLAlchemy, Pydantic, JWT, Passlib/Bcrypt

**Database:** SQLite by default; PostgreSQL supported through `DATABASE_URL`

**Machine Learning:** Pandas, scikit-learn, TF-IDF, Logistic Regression, Joblib

## Run locally

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Train the complaint classifier

Run from the project root:

```bash
python ml/train.py
```

This creates `ml/complaint_classifier.joblib` and prints accuracy plus a classification report.

### 4. Start FastAPI

From the project root:

```bash
uvicorn backend.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

### 5. Start the frontend

Do **not** open `frontend/index.html` using `file://` if your browser blocks API requests. Use a simple local server instead.

From the project root:

```bash
python -m http.server 5500 --directory frontend
```

Open:

```text
http://localhost:5500
```

## Demo accounts

| Role | Email | Password |
|---|---|---|
| Student | `student@demo.com` | `demo123` |
| Teacher | `teacher@demo.com` | `demo123` |
| Admin | `admin@demo.com` | `demo123` |

## Main API areas

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/reactivate`
- `GET/PATCH /api/me`
- `POST /api/classify`
- `GET/POST /api/complaints`
- `PATCH /api/complaints/{id}/status`
- `GET/POST/DELETE /api/bookings`
- `GET/POST /api/announcements`
- `GET /api/notifications`
- `GET /api/admin/users`
- `GET /api/admin/analytics`

## ML workflow

1. User enters an issue title and description.
2. Backend combines the text and sends it through the trained TF-IDF + Logistic Regression pipeline.
3. The model predicts one of five categories: IT, Water, Electrical, Civil, or Security.
4. A confidence score is returned.
5. Keyword-based safety rules raise priority for urgent terms such as fire, exposed wiring, leak, unsafe, or danger.
6. The predicted category maps to the responsible campus department.
7. The complaint and its routing decision are persisted in the database.

## Database

SQLite is used automatically for local development. To use PostgreSQL, set:

```env
DATABASE_URL=postgresql+psycopg2://username:password@host:5432/campusconnect
```

Also set a strong `SECRET_KEY` and the frontend origin in the environment.

## Resume-ready project summary

**CampusConnect | Python, FastAPI, SQLAlchemy, SQLite/PostgreSQL, JavaScript, NLP, scikit-learn**

- Built a role-based campus management platform with JWT authentication, complaint tracking, facility booking, announcements, notifications, and admin analytics using FastAPI and SQLAlchemy.
- Implemented an NLP complaint-routing pipeline using TF-IDF and Logistic Regression to classify issues, estimate confidence/priority, and automatically route complaints to campus departments.

## Future enhancements

- React + Vite frontend migration for larger-scale component architecture.
- Image evidence upload and computer-vision issue tagging.
- Email/push notifications.
- Redis caching and background workers.
- Docker deployment and CI/CD.
