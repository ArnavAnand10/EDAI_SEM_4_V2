from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware  # Add this import

from models.doctor_model import DoctorDocument
from models.patient_model import PatientDocument
from models.case_model import CaseDocument
from models.report_model import ReportDocument
from db.mongoDb import connectToDB
from controllers.doctor_controller import router as doctor_router
from controllers.patient_controller import router as patient_router
from controllers.case_controller import router as case_router
from controllers.report_controller import router as report_router
from controllers.patient_progress import router as progress_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncIOMotorClient(uri)
    try:
        await client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(f"Error: {e}")

    database = client['EDAI_4']

    try:
        await init_beanie(database, document_models=[DoctorDocument, PatientDocument, CaseDocument, ReportDocument])
        print("Beanie has been initialized successfully.")
    except Exception as e:
        print(f"Error initializing Beanie: {e}")
    
    yield  
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uri = "mongodb+srv://arnavanand710:anand@cluster0.ikgfxgh.mongodb.net/?appName=Cluster0"



app.include_router(doctor_router)
app.include_router(patient_router)
app.include_router(case_router)
app.include_router(report_router)
app.include_router(progress_router)

@app.get("/")
def root():
    return {"message": "Medical AI Backend API is live"}



