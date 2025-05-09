from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional,List
from datetime import datetime

class CaseBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: Optional[str] = "active"
    report_ids: Optional[List[PydanticObjectId]] = []  


class CaseDocument(CaseBase, Document):
    patient_id: PydanticObjectId
    created_at: datetime = datetime.utcnow()

    class Settings:
        name = "cases"
