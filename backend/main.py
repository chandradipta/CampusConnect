import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "complaint_classifier.joblib"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'campusconnect.db'}")
SECRET_KEY = os.getenv("SECRET_KEY", "campusconnect-change-this-secret")
ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "120"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="student")
    dept = Column(String(120), default="")
    active = Column(Boolean, default=True)
    phone = Column(String(30), default="")
    bio = Column(String(500), default="")
    location = Column(String(120), default="")
    website = Column(String(255), default="")
    avatar = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)

class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True)
    ticket = Column(String(30), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(180), nullable=False)
    description = Column(Text, nullable=False)
    location = Column(String(180), default="Campus")
    category = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False, default="Low")
    department = Column(String(100), nullable=False)
    status = Column(String(30), nullable=False, default="Submitted")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    user = relationship("User")
    updates = relationship("ComplaintUpdate", cascade="all, delete-orphan", order_by="ComplaintUpdate.created_at")

class ComplaintUpdate(Base):
    __tablename__ = "complaint_updates"
    id = Column(Integer, primary_key=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(30), nullable=False)
    comment = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Facility(Base):
    __tablename__ = "facilities"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    location = Column(String(120), default="Campus")
    capacity = Column(Integer, default=20)

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True)
    facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    booking_date = Column(String(10), nullable=False)
    slot = Column(String(20), nullable=False)
    purpose = Column(String(180), default="Campus activity")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    facility = relationship("Facility")
    user = relationship("User")

class Announcement(Base):
    __tablename__ = "announcements"
    id = Column(Integer, primary_key=True)
    title = Column(String(180), nullable=False)
    body = Column(Text, nullable=False)
    audience = Column(String(30), default="all")
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    author = relationship("User")

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(160), nullable=False)
    message = Column(Text, nullable=False)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CampusConnect API", version="2.0.0", description="Smart campus management with NLP complaint routing.")
origins = [x.strip() for x in FRONTEND_ORIGIN.split(",") if x.strip()] if FRONTEND_ORIGIN != "*" else ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=FRONTEND_ORIGIN != "*", allow_methods=["*"], allow_headers=["*"])

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    role: str = Field(default="student", pattern="^(student|teacher)$")
    dept: str = Field(default="", max_length=120)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ComplaintIn(BaseModel):
    title: str = Field(min_length=3, max_length=180)
    description: str = Field(min_length=8, max_length=3000)
    location: str = Field(default="Campus", max_length=180)

class StatusIn(BaseModel):
    status: str = Field(pattern="^(Submitted|Under Review|In Progress|Resolved|Rejected)$")
    comment: str = Field(default="", max_length=1000)

class ProfileIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    dept: str = Field(default="", max_length=120)
    phone: str = Field(default="", max_length=30)
    location: str = Field(default="", max_length=120)
    website: str = Field(default="", max_length=255)
    bio: str = Field(default="", max_length=500)
    avatar: str = Field(default="", max_length=500000)
    new_password: Optional[str] = Field(default=None, min_length=6, max_length=128)

class BookingIn(BaseModel):
    facility_id: int
    booking_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    slot: str = Field(pattern=r"^(09:00|10:00|11:00|12:00|14:00|16:00)$")
    purpose: str = Field(default="Campus activity", max_length=180)

class AnnouncementIn(BaseModel):
    title: str = Field(min_length=3, max_length=180)
    body: str = Field(min_length=3, max_length=2000)
    audience: str = Field(default="all", pattern="^(all|students|teachers)$")


def db():
    s = SessionLocal()
    try: yield s
    finally: s.close()

def token_for(u: User):
    exp = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    return jwt.encode({"sub": str(u.id), "role": u.role, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def user_out(u: User):
    return {"id": u.id, "name": u.name, "email": u.email, "role": u.role, "dept": u.dept, "active": u.active, "phone": u.phone or "", "bio": u.bio or "", "location": u.location or "", "website": u.website or "", "avatar": u.avatar or "", "created_at": u.created_at.isoformat() if u.created_at else ""}

def get_user(token: str = Depends(oauth2), s: Session = Depends(db)):
    try: uid = int(jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("sub"))
    except (JWTError, ValueError, TypeError): raise HTTPException(status_code=401, detail="Invalid or expired token")
    u = s.get(User, uid)
    if not u or not u.active: raise HTTPException(status_code=401, detail="Account is inactive or unavailable")
    return u

def admin(u: User):
    if u.role != "admin": raise HTTPException(status_code=403, detail="Admin access required")

def seed(s: Session):
    if s.query(User).count(): return
    users = [
        User(name="Asha Rao", email="student@demo.com", password_hash=pwd.hash("demo123"), role="student", dept="B.Tech CSE"),
        User(name="Dr. Mehta", email="teacher@demo.com", password_hash=pwd.hash("demo123"), role="teacher", dept="Computer Science"),
        User(name="Campus Admin", email="admin@demo.com", password_hash=pwd.hash("demo123"), role="admin", dept="Campus Office"),
    ]
    s.add_all(users)
    for n in ["IT Support", "Water & Sanitation", "Electrical", "Civil", "Security"]: s.add(Department(name=n))
    for n, loc, cap in [("Seminar Hall","Academic Block",80),("Computer Lab","Block B",40),("Auditorium","Main Block",250),("Sports Ground","Sports Complex",100)]: s.add(Facility(name=n, location=loc, capacity=cap))
    s.commit()
    s.refresh(users[2])
    s.add_all([
        Announcement(title="Library open till 10 PM during exams", body="Extended study hours are available during the examination period.", audience="all", author_id=users[2].id),
        Announcement(title="Unit Test 2 syllabus uploaded", body="Please check your department communication channel for the latest syllabus.", audience="students", author_id=users[1].id),
    ])
    s.commit()

with SessionLocal() as s: seed(s)

@app.get("/api/health")
def health(): return {"status":"ok", "service":"CampusConnect API", "version":"2.0.0"}

@app.post("/api/auth/register")
def register(x: RegisterIn, s: Session = Depends(db)):
    email = str(x.email).lower()
    if s.query(User).filter(User.email == email).first(): raise HTTPException(409, "Email already registered")
    u = User(name=x.name.strip(), email=email, password_hash=pwd.hash(x.password), role=x.role, dept=x.dept.strip())
    s.add(u); s.commit(); s.refresh(u)
    return {"access_token": token_for(u), "token_type":"bearer", "user":user_out(u)}

@app.post("/api/auth/login")
def login(x: LoginIn, s: Session = Depends(db)):
    u = s.query(User).filter(User.email == str(x.email).lower()).first()
    if not u or not pwd.verify(x.password, u.password_hash): raise HTTPException(401, "Incorrect email or password")
    if not u.active: raise HTTPException(403, "Account is deactivated. Use the Reactivate option with your credentials.")
    return {"access_token": token_for(u), "token_type":"bearer", "user":user_out(u)}

@app.post("/api/auth/reactivate")
def reactivate(x: LoginIn, s: Session = Depends(db)):
    u = s.query(User).filter(User.email == str(x.email).lower()).first()
    if not u or not pwd.verify(x.password, u.password_hash): raise HTTPException(401, "Incorrect email or password")
    u.active=True; s.commit(); return {"access_token": token_for(u), "token_type":"bearer", "user":user_out(u)}

@app.get("/api/me")
def me(u: User = Depends(get_user)): return user_out(u)

@app.patch("/api/me")
def update_me(x: ProfileIn, u: User = Depends(get_user), s: Session = Depends(db)):
    for k in ["name","dept","phone","location","website","bio","avatar"]: setattr(u, k, getattr(x,k))
    if x.new_password: u.password_hash = pwd.hash(x.new_password)
    s.commit(); s.refresh(u); return user_out(u)

@app.post("/api/me/deactivate")
def deactivate(u: User = Depends(get_user), s: Session = Depends(db)):
    u.active=False; s.commit(); return {"message":"Account deactivated"}

@app.delete("/api/me")
def delete_me(u: User = Depends(get_user), s: Session = Depends(db)):
    s.query(Notification).filter(Notification.user_id==u.id).delete(synchronize_session=False)
    s.query(Booking).filter(Booking.user_id==u.id).delete(synchronize_session=False)
    s.query(Complaint).filter(Complaint.user_id==u.id).delete(synchronize_session=False)
    s.query(Announcement).filter(Announcement.author_id==u.id).delete(synchronize_session=False)
    s.delete(u); s.commit(); return {"message":"Account deleted"}

@app.post("/api/classify")
def classify(x: ComplaintIn, u: User = Depends(get_user)):
    text = f"{x.title} {x.description}".strip()
    if not MODEL_PATH.exists(): raise HTTPException(503, "ML model is not trained. Run: python ml/train.py")
    model = joblib.load(MODEL_PATH)
    category = str(model.predict([text])[0])
    confidence = float(max(model.predict_proba([text])[0])) if hasattr(model, "predict_proba") else 0.0
    high_words = ("fire", "exposed", "spark", "leak", "unsafe", "danger", "broken glass", "electric shock")
    priority = "High" if any(w in text.lower() for w in high_words) else ("Medium" if confidence < 0.62 else "Low")
    departments = {"IT":"IT Support","Water":"Water & Sanitation","Electrical":"Electrical","Civil":"Civil","Security":"Security"}
    return {"category":category,"confidence":round(confidence,4),"priority":priority,"department":departments.get(category,category)}

@app.post("/api/complaints")
def create_complaint(x: ComplaintIn, u: User = Depends(get_user), s: Session = Depends(db)):
    result = classify(x,u)
    count = s.query(Complaint).count()+1
    c = Complaint(ticket=f"CC-{datetime.now().year}-{count:05d}", user_id=u.id, title=x.title.strip(), description=x.description.strip(), location=x.location.strip() or "Campus", category=result["category"], priority=result["priority"], department=result["department"], status="Submitted")
    s.add(c); s.flush(); s.add(ComplaintUpdate(complaint_id=c.id,status="Submitted",comment="Issue submitted and automatically routed.")); s.add(Notification(user_id=u.id,title="Issue submitted",message=f"{c.ticket} was routed to {c.department}.")); s.commit(); s.refresh(c)
    return complaint_out(c)

def complaint_out(c: Complaint):
    return {"id":c.id,"ticket":c.ticket,"title":c.title,"description":c.description,"location":c.location,"category":c.category,"priority":c.priority,"department":c.department,"status":c.status,"created_at":c.created_at.isoformat(),"updates":[{"status":x.status,"comment":x.comment,"created_at":x.created_at.isoformat()} for x in c.updates]}

@app.get("/api/complaints")
def complaints(u: User = Depends(get_user), s: Session = Depends(db)):
    q = s.query(Complaint).order_by(Complaint.created_at.desc()) if u.role in ("teacher","admin") else s.query(Complaint).filter(Complaint.user_id==u.id).order_by(Complaint.created_at.desc())
    return [complaint_out(c) for c in q.all()]

@app.patch("/api/complaints/{cid}/status")
def update_status(cid: int, x: StatusIn, u: User = Depends(get_user), s: Session = Depends(db)):
    c=s.get(Complaint,cid)
    if not c: raise HTTPException(404,"Complaint not found")
    if u.role not in ("teacher","admin"): raise HTTPException(403,"Only staff can update complaints")
    if u.role=="teacher" and x.status not in ("Under Review","In Progress","Resolved"): raise HTTPException(403,"Teachers cannot use this status")
    c.status=x.status; s.add(ComplaintUpdate(complaint_id=c.id,status=x.status,comment=x.comment or f"Status changed to {x.status}.")); s.add(Notification(user_id=c.user_id,title=f"{c.ticket} updated",message=f"Status: {x.status}")); s.commit(); s.refresh(c); return complaint_out(c)

@app.get("/api/facilities")
def facilities(s: Session=Depends(db)): return [{"id":f.id,"name":f.name,"location":f.location,"capacity":f.capacity} for f in s.query(Facility).all()]

@app.get("/api/bookings")
def bookings(u: User=Depends(get_user), s: Session=Depends(db)):
    q=s.query(Booking).order_by(Booking.booking_date,Booking.slot)
    if u.role=="student": q=q.filter(Booking.user_id==u.id)
    return [{"id":b.id,"facility":b.facility.name,"facility_id":b.facility_id,"date":b.booking_date,"slot":b.slot,"purpose":b.purpose,"user_id":b.user_id} for b in q.all()]

@app.post("/api/bookings")
def create_booking(x: BookingIn,u: User=Depends(get_user),s: Session=Depends(db)):
    if not s.get(Facility,x.facility_id): raise HTTPException(404,"Facility not found")
    if s.query(Booking).filter_by(facility_id=x.facility_id,booking_date=x.booking_date,slot=x.slot).first(): raise HTTPException(409,"That slot is already booked")
    b=Booking(facility_id=x.facility_id,user_id=u.id,booking_date=x.booking_date,slot=x.slot,purpose=x.purpose);s.add(b);s.add(Notification(user_id=u.id,title="Facility booked",message=f"Your booking is confirmed for {x.booking_date} at {x.slot}."));s.commit();s.refresh(b);return {"id":b.id,"message":"Booking confirmed"}

@app.delete("/api/bookings/{bid}")
def cancel_booking(bid:int,u:User=Depends(get_user),s:Session=Depends(db)):
    b=s.get(Booking,bid)
    if not b: raise HTTPException(404,"Booking not found")
    if b.user_id!=u.id and u.role!="admin": raise HTTPException(403,"Not allowed")
    s.delete(b);s.commit();return {"message":"Booking cancelled"}

@app.get("/api/announcements")
def announcements(u:User=Depends(get_user),s:Session=Depends(db)):
    q=s.query(Announcement).order_by(Announcement.created_at.desc()).all()
    return [{"id":a.id,"title":a.title,"body":a.body,"audience":a.audience,"author":a.author.name,"created_at":a.created_at.isoformat()} for a in q if a.audience=="all" or a.audience==u.role+"s"]

@app.post("/api/announcements")
def create_announcement(x:AnnouncementIn,u:User=Depends(get_user),s:Session=Depends(db)):
    if u.role not in ("teacher","admin"): raise HTTPException(403,"Staff access required")
    a=Announcement(title=x.title.strip(),body=x.body.strip(),audience=x.audience,author_id=u.id);s.add(a);s.flush()
    targets=s.query(User).filter(User.active==True)
    for target in targets:
        if target.id!=u.id and (x.audience=="all" or x.audience==target.role+"s"): s.add(Notification(user_id=target.id,title=x.title,message=x.body[:240]))
    s.commit();return {"message":"Announcement published"}

@app.get("/api/notifications")
def notifications(u:User=Depends(get_user),s:Session=Depends(db)):
    q=s.query(Notification).filter(Notification.user_id==u.id).order_by(Notification.created_at.desc()).limit(30).all()
    return [{"id":n.id,"title":n.title,"message":n.message,"read":n.read,"created_at":n.created_at.isoformat()} for n in q]

@app.post("/api/notifications/read-all")
def read_notifications(u:User=Depends(get_user),s:Session=Depends(db)):
    s.query(Notification).filter(Notification.user_id==u.id,Notification.read==False).update({"read":True});s.commit();return {"message":"Notifications marked as read"}

@app.delete("/api/admin/users/{uid}")
def delete_user(uid:int,u:User=Depends(get_user),s:Session=Depends(db)):
    admin(u)
    target=s.get(User,uid)
    if not target: raise HTTPException(404,"User not found")
    if target.id==u.id: raise HTTPException(400,"You cannot delete yourself here")
    s.query(Notification).filter(Notification.user_id==uid).delete(synchronize_session=False)
    s.query(Booking).filter(Booking.user_id==uid).delete(synchronize_session=False)
    s.query(Complaint).filter(Complaint.user_id==uid).delete(synchronize_session=False)
    s.query(Announcement).filter(Announcement.author_id==uid).delete(synchronize_session=False)
    s.delete(target); s.commit(); return {"message":"User deleted"}

@app.get("/api/admin/users")
def admin_users(u:User=Depends(get_user),s:Session=Depends(db)):
    admin(u);return [user_out(x) for x in s.query(User).order_by(User.created_at.desc()).all()]

@app.patch("/api/admin/users/{uid}/active")
def toggle_user(uid:int,u:User=Depends(get_user),s:Session=Depends(db)):
    admin(u);target=s.get(User,uid)
    if not target: raise HTTPException(404,"User not found")
    if target.id==u.id: raise HTTPException(400,"You cannot deactivate yourself here")
    target.active=not target.active;s.commit();return {"active":target.active}

@app.get("/api/admin/analytics")
def analytics(u:User=Depends(get_user),s:Session=Depends(db)):
    admin(u); total=s.query(Complaint).count();
    def grouped(col): return [{"name":k,"count":v} for k,v in s.query(col,func.count(Complaint.id)).group_by(col).all()]
    return {"total_complaints":total,"open_complaints":s.query(Complaint).filter(Complaint.status.notin_(["Resolved","Rejected"])).count(),"resolved":s.query(Complaint).filter(Complaint.status=="Resolved").count(),"high_priority":s.query(Complaint).filter(Complaint.priority=="High").count(),"active_users":s.query(User).filter(User.active==True).count(),"categories":grouped(Complaint.category),"statuses":grouped(Complaint.status),"departments":grouped(Complaint.department)}
