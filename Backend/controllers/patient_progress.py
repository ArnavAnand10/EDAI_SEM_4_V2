from fastapi import APIRouter, HTTPException
from models.report_model import ReportDocument
from models.case_model import CaseDocument
from beanie import PydanticObjectId
from typing import List
import google.generativeai as genai
import json
import re  # For JSON extraction if needed

router = APIRouter(prefix="/progress", tags=["Progress"])

# Configure Gemini
genai.configure(api_key="AIzaSyC_B4ybRuHtbbF9qAneIvBjCawpZOUS1cw")
model = genai.GenerativeModel('gemini-2.0-flash')

def extract_json_from_response(text: str) -> dict:
    """Safely extract JSON from Gemini's response text"""
    try:
        # First try direct JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from markdown or code blocks
            json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find raw JSON in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError("No valid JSON found in response")
        except Exception as e:
            raise ValueError(f"Could not extract JSON: {str(e)}")

def generate_gemini_prompt(reports: List[dict]) -> str:
    """Generate a prompt that strongly enforces JSON output"""
    return f"""
    Analyze these medical reports and provide a progress summary.
    You MUST respond with ONLY valid JSON in this exact format:

    {{
        "progress_summary": {{
            "overview": "Brief overall assessment",
            "trend": "improving/stable/declining",
            "key_findings": ["list", "of", "key", "points"],
            "next_steps": ["recommended", "actions"]
        }}
    }}

    Reports to analyze:
    {json.dumps(reports, indent=2)}
    """

@router.get("/")
async def get_progress_analysis(case_id: str):
    try:
        # Validate case ID
        case_id = case_id.strip()
        object_id = PydanticObjectId(case_id)
        
        # Verify case exists
        case = await CaseDocument.find_one({"_id": object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Fetch reports in order
        reports = await ReportDocument.find(
            {"case_id": object_id},
            sort=[("created_at", 1)]
        ).to_list()

        if not reports:
            return {"progress_summary": "No reports available for analysis"}
        
        # Prepare clean report data
        report_data = [{"summary": r.analysis_json} for r in reports if r.analysis_json]
        
        # Get Gemini response
        prompt = generate_gemini_prompt(report_data)
        response = model.generate_content(prompt)
        
        # Safely parse the response
        try:
            result = extract_json_from_response(response.text)
            return result
        except ValueError as e:
            # Fallback to simple summary if JSON parsing fails
            return {
                "progress_summary": {
                    "overview": "Analysis completed but format unexpected",
                    "raw_response": response.text[:500] + "..." if len(response.text) > 500 else response.text
                }
            }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case ID format")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )