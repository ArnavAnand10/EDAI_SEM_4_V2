from beanie import Document, PydanticObjectId
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class ReportBase(BaseModel):
    file_url: str
    analysis_summary: Optional[str] = None
    analysis_json: Optional[Dict] = None 
    uploaded_at: datetime = datetime.utcnow()

class ReportDocument(ReportBase, Document):
    case_id: PydanticObjectId

    class Settings:
        name = "reports"
    