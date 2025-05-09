from fastapi import APIRouter, HTTPException
from models.report_model import ReportDocument
from schemas.report_schema import ReportCreate, ReportOut
from models.case_model import CaseDocument
from typing import List
from services.imageAnalysis import analyze_medical_image
from bson import ObjectId
from beanie import PydanticObjectId

router = APIRouter(prefix="/reports", tags=["Reports"])
import json 

@router.post("/")
async def create_report(payload: ReportCreate, case_id: str):
    case = await CaseDocument.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    payload = payload.dict()
    file_url = payload['file_url']
    print(file_url,payload)
    try:
        analysis_result_str = analyze_medical_image(file_url)
        analysis_result_dict = json.loads(analysis_result_str)  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

    report = ReportDocument(
        file_url=file_url,
        analysis_json=analysis_result_dict,
        case_id=case.id
    )
    await report.insert()

    case.report_ids.append(report.id)
    await case.save()

    print("new report generated", report.id, type(report.id))
    print(type(report.id))
    return {
         "report_id":str(report.id)
    }

@router.get("/")
async def get_reports(case_id: str):
    case_id =  case_id.replace(" ","")
    print(case_id,len(case_id))
    object_id = ObjectId(case_id)
    print(object_id)
   
    case = await CaseDocument.find_one({"_id": object_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Find all reports that have this case_id
    reports = await ReportDocument.find({"case_id": object_id}).to_list()
    
    # Convert to output format
    return [
        {
            "report_id": str(report.id),
            "file_url": report.file_url
        }
        for report in reports
    ]

@router.get("/report")
async def get_report_by_id(report_id: str):
    report_id = report_id.strip()
    
    # Validate the ObjectId
    if not ObjectId.is_valid(report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    object_id = ObjectId(report_id)
    
    report = await ReportDocument.get(object_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "report_id": str(report.id),
        "file_url": report.file_url,
        "analysis_json": report.analysis_json,
        "case_id": str(report.case_id)
    }
