from pydantic import BaseModel
from typing import Optional,List
from datetime import datetime

class CaseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: Optional[str] = "active"
    report_ids: Optional[List[str]] = [] 

class CaseOut(BaseModel):
    id: str
    title: str
    description: Optional[str]
    status: str
    patient_id: str
    created_at: datetime

    class Config:
        from_attributes = True
