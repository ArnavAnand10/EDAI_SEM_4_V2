from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional, List
from bson import ObjectId 

class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    address: Optional[str] = None
    case_ids: Optional[List[PydanticObjectId]] = []  

class PatientDocument(PatientBase, Document):
    doctor_id: PydanticObjectId 

    class Settings:
        name = "patients" 
