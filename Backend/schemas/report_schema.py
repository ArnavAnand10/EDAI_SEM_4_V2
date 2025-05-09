from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class ReportCreate(BaseModel):
    file_url: str
    analysis_json: Optional[Dict] = None

class ReportOut(BaseModel):
    id: str
    file_url: str
    analysis_summary: Optional[str]
    analysis_json: Optional[Dict]
    case_id: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ReportDocument(ReportCreate, Document):
    case_id: PydanticObjectId
    uploaded_at: datetime = datetime.utcnow()

    class Settings:
        name = "reports"
