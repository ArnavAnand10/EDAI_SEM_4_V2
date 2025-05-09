from pydantic import BaseModel
from typing import Optional, List

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    address: Optional[str] = None
    case_ids: Optional[List[str]] = [] 


class PatientOut(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    address: Optional[str]
    doctor_id: str
    case_ids: List[str] 

    class Config:
            from_attributes = True

