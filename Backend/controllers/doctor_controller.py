from fastapi import APIRouter, HTTPException, status
from models.doctor_model import DoctorDocument
from schemas.doctor_schema import DoctorCreate, DoctorOut, DoctorLogin
from utils.hashing import hash_password, verify_password

router = APIRouter(prefix="/doctors", tags=["Doctors"])

@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=DoctorOut)
async def register_doctor(payload: DoctorCreate):
    print('API called for signup:', payload.email)
    
    existing = await DoctorDocument.find_one({"email": payload.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Doctor already exists."
        )

    hashed_pw = hash_password(payload.password)
    doctor = DoctorDocument(**payload.dict(exclude={"password"}), hashed_password=hashed_pw)
    await doctor.insert()

    return DoctorOut(id=str(doctor.id), email=doctor.email)


@router.post("/login", status_code=status.HTTP_200_OK)
async def login_doctor(payload: DoctorLogin):
    doctor = await DoctorDocument.find_one({"email": payload.email})

    if not doctor or not verify_password(payload.password, doctor.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    return {
        "status": "success",
        "message": "Login successful",
        "data": {
            "doctor_id": str(doctor.id),
            "email": doctor.email
        }
    }
