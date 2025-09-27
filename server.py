from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt
import jwt
import aiofiles
from bson import ObjectId
from mangum import Mangum

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db
    
    # --- Code that runs on application startup ---
    print("Connecting to MongoDB...")
    mongo_url = os.environ.get('MONGO_URL')
    if not mongo_url:
        raise RuntimeError("MONGO_URL environment variable is not set.")
        
    try:
        client = AsyncIOMotorClient(mongo_url)
        db_name = os.environ.get('DB_NAME')
        if not db_name:
            raise RuntimeError("DB_NAME environment variable is not set.")
        db = client[db_name]
        print("Connected to MongoDB successfully!")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")


app = FastAPI(title="MindMate - Student Mental Health Platform", lifespan=lifespan)
api_router = APIRouter(prefix="/api")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create uploads directory
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# MongoDB connection
client = None
db = None

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'your-secret-key-change-this')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# This is the new lifespan context manager
# It replaces both the startup and shutdown event decorators.

        # You might want to raise an exception here to stop the app from starting.
        # For now, we'll just log it.

    # Initialize sample data if it doesn't already exist
    print("Initializing sample data...")
    sample_quizzes = [
        {
            "id": str(uuid.uuid4()),
            "title": "Anxiety Assessment",
            "description": "Evaluate your anxiety levels",
            "category": "anxiety",
            "created_by": "system",
            "questions": [
                {
                    "question": "How often do you feel nervous or anxious?",
                    "options": ["Never", "Sometimes", "Often", "Always"],
                    "type": "multiple_choice"
                },
                {
                    "question": "Do you have trouble concentrating?",
                    "options": ["Never", "Sometimes", "Often", "Always"],
                    "type": "multiple_choice"
                }
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Depression Screening",
            "description": "Screen for depression symptoms",
            "category": "depression",
            "created_by": "system",
            "questions": [
                {
                    "question": "How often do you feel down or hopeless?",
                    "options": ["Never", "Rarely", "Sometimes", "Often"],
                    "type": "multiple_choice"
                },
                {
                    "question": "Have you lost interest in activities you used to enjoy?",
                    "options": ["Not at all", "A little", "Somewhat", "Very much"],
                    "type": "multiple_choice"
                }
            ],
            "created_at": datetime.now(timezone.utc)
        }
    ]
    
    for quiz in sample_quizzes:
        existing = await db.quizzes.find_one({"title": quiz["title"]})
        if not existing:
            await db.quizzes.insert_one(quiz)
            print(f"Inserted sample quiz: {quiz['title']}")
            
    # The yield keyword is crucial. It hands control back to the application.
    yield

    # --- Code that runs on application shutdown ---
    print("Closing MongoDB connection...")
    if client:
        client.close()

# Create the main app instance and pass the lifespan function

# Security
security = HTTPBearer()

# Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    username: str
    full_name: str
    user_type: str  # "student" or "counsellor"
    profile_image: Optional[str] = None
    bio: Optional[str] = None
    specializations: Optional[List[str]] = []  # for counsellors
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    password: str
    user_type: str
    bio: Optional[str] = None
    specializations: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: User

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    student_id: str
    counsellor_id: str
    title: str
    description: str
    session_date: datetime
    duration_minutes: int
    status: str = "scheduled"  # scheduled, completed, cancelled
    session_type: str = "video"  # video, audio, chat
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SessionCreate(BaseModel):
    counsellor_id: str
    title: str
    description: str
    session_date: datetime
    duration_minutes: int = 60
    session_type: str = "video"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: Optional[str] = None
    group_id: Optional[str] = None
    content: str
    message_type: str = "text"  # text, image, file
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Group(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    group_type: str = "support"  # support, therapy, general
    created_by: str
    members: List[str] = []
    is_public: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ResourceFile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    file_path: Optional[str] = None # Internal path, not for frontend
    file_url: Optional[str] = None # Public URL for playback
    file_type: str  # video, audio
    uploaded_by: str
    category: str
    is_public: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Quiz(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    questions: List[Dict[str, Any]]
    category: str
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MoodEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    mood: str
    emoji: str
    date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authentication Routes
@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"$or": [{"email": user_data.email}, {"username": user_data.username}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    hashed_password = hash_password(user_data.password)
    user_dict = user_data.dict(exclude={"password"})
    user = User(**user_dict)
    
    # Store in database
    user_doc = user.dict()
    user_doc["password"] = hashed_password
    await db.users.insert_one(user_doc)
    
    # Create token
    access_token = create_access_token(data={"sub": user.id})
    
    return TokenResponse(access_token=access_token, token_type="bearer", user=user)

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    user_doc = await db.users.find_one({"email": user_data.email})
    if not user_doc or not verify_password(user_data.password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = User(**user_doc)
    access_token = create_access_token(data={"sub": user.id})
    
    return TokenResponse(access_token=access_token, token_type="bearer", user=user)

@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# User Routes
@api_router.get("/users/search")
async def search_users(q: str, current_user: User = Depends(get_current_user)):
    users = await db.users.find({
        "$or": [
            {"username": {"$regex": q, "$options": "i"}},
            {"full_name": {"$regex": q, "$options": "i"}}
        ],
        "id": {"$ne": current_user.id}
    }).to_list(20)
    return [User(**user) for user in users]

@api_router.get("/users/counsellors")
async def get_counsellors():
    counsellors = await db.users.find({"user_type": "counsellor", "is_active": True}).to_list(50)
    return [User(**counsellor) for counsellor in counsellors]

# Session Booking Routes
@api_router.post("/sessions", response_model=Session)
async def create_session(session_data: SessionCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type != "student":
        raise HTTPException(status_code=403, detail="Only students can book sessions")
    
    session = Session(**session_data.dict(), student_id=current_user.id)
    await db.sessions.insert_one(session.dict())
    return session

@api_router.get("/sessions", response_model=List[Session])
async def get_user_sessions(current_user: User = Depends(get_current_user)):
    query = {}
    if current_user.user_type == "student":
        query["student_id"] = current_user.id
    elif current_user.user_type == "counsellor":
        query["counsellor_id"] = current_user.id
    
    sessions = await db.sessions.find(query).to_list(100)
    return [Session(**session) for session in sessions]

# Community Forum Routes
@api_router.post("/groups", response_model=Group)
async def create_group(name: str = Form(...), description: str = Form(...), 
                       group_type: str = Form("support"), is_public: bool = Form(True),
                       current_user: User = Depends(get_current_user)):
    group = Group(name=name, description=description, group_type=group_type, 
                  created_by=current_user.id, members=[current_user.id], is_public=is_public)
    await db.groups.insert_one(group.dict())
    return group

@api_router.get("/groups", response_model=List[Group])
async def get_groups(current_user: User = Depends(get_current_user)):
    groups = await db.groups.find({"$or": [{"is_public": True}, {"members": current_user.id}]}).to_list(50)
    return [Group(**group) for group in groups]

@api_router.post("/groups/{group_id}/join")
async def join_group(group_id: str, current_user: User = Depends(get_current_user)):
    group = await db.groups.find_one({"id": group_id})
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if current_user.id not in group["members"]:
        await db.groups.update_one({"id": group_id}, {"$push": {"members": current_user.id}})
    
    return {"message": "Joined group successfully"}

@api_router.get("/groups/{group_id}/messages", response_model=List[Message])
async def get_group_messages(group_id: str, current_user: User = Depends(get_current_user)):
    messages = await db.messages.find({"group_id": group_id}).sort("created_at", 1).to_list(100)
    return [Message(**message) for message in messages]

@api_router.post("/messages", response_model=Message)
async def send_message(content: str = Form(...), receiver_id: str = Form(None), 
                       group_id: str = Form(None), current_user: User = Depends(get_current_user)):
    message = Message(sender_id=current_user.id, receiver_id=receiver_id, 
                      group_id=group_id, content=content)
    await db.messages.insert_one(message.dict())
    return message

@api_router.get("/messages/conversations")
async def get_conversations(current_user: User = Depends(get_current_user)):
    # Get direct messages
    messages = await db.messages.find({
        "$or": [{"sender_id": current_user.id}, {"receiver_id": current_user.id}],
        "group_id": None
    }).sort("created_at", -1).to_list(100)
    
    # Group by conversation
    conversations = {}
    for message in messages:
        other_user_id = message["receiver_id"] if message["sender_id"] == current_user.id else message["sender_id"]
        if other_user_id not in conversations:
            other_user = await db.users.find_one({"id": other_user_id})
            conversations[other_user_id] = {
                "user": User(**other_user) if other_user else None,
                "last_message": Message(**message),
                "messages": []
            }
        conversations[other_user_id]["messages"].append(Message(**message))
    
    return list(conversations.values())

# Resource Hub Routes
@api_router.post("/resources/upload")
async def upload_resource(
    title: str = Form(...),
    description: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/mov", "audio/mp3", "audio/wav", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Create unique filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Create resource record
    resource = ResourceFile(
        title=title,
        description=description,
        file_path=str(file_path),
        file_type="video" if file.content_type.startswith("video") else "audio",
        uploaded_by=current_user.id,
        category=category,
        # IMPORTANT: The file_url is not saved to the DB here. It's generated on retrieval.
        file_url=None
    )
    await db.resources.insert_one(resource.dict(exclude_none=True))
    
    return resource

@api_router.get("/resources", response_model=List[ResourceFile])
async def get_resources(category: str = None):
    query = {"is_public": True}
    if category:
        query["category"] = category
    
    resources_cursor = db.resources.find(query).sort("created_at", -1)
    
    response_resources = []
    async for resource in resources_cursor:
        # Generate the public URL from the stored file_path
        file_url = f"/uploads/{Path(resource['file_path']).name}"
        
        # Create a new dictionary to add the file_url
        resource_data = {
            **resource,
            "file_url": file_url
        }
        
        response_resources.append(ResourceFile(**resource_data))
        
    return response_resources

# Quiz Routes
@api_router.get("/quizzes", response_model=List[Quiz])
async def get_quizzes():
    quizzes = await db.quizzes.find().to_list(20)
    return [Quiz(**quiz) for quiz in quizzes]

# Mood Tracking
@api_router.post("/mood", response_model=MoodEntry)
async def record_mood(mood: str = Form(...), emoji: str = Form(...), 
                       current_user: User = Depends(get_current_user)):
    mood_entry = MoodEntry(user_id=current_user.id, mood=mood, emoji=emoji)
    await db.mood_entries.insert_one(mood_entry.dict())
    return mood_entry

@api_router.get("/mood/history")
async def get_mood_history(current_user: User = Depends(get_current_user)):
    entries = await db.mood_entries.find({"user_id": current_user.id}).sort("date", -1).to_list(30)
    return [MoodEntry(**entry) for entry in entries]

# Static file serving
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include router
app.include_router(api_router)

# CORS middleware
cors_origins = os.environ.get('CORS_ORIGINS', '').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

handler = Mangum(app) check this
