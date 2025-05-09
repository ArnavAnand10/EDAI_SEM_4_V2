from passlib.context import CryptContext
from pydantic import BaseModel
from beanie import Document
from typing import List
from beanie import PydanticObjectId
from bson import ObjectId 

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class Doctor(BaseModel):
    email: str

class DoctorDocument(Doctor, Document):
    hashed_password: str
    patients: List = []

    def set_password(self, password: str) -> None:
        self.hashed_password = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

    class Settings:
        name = "doctors"


async def register_doctor(payload: Doctor):
    existing = await DoctorDocument.find_one(DoctorDocument.email == payload.email)
    if existing:
        pass
    else:
        pass
