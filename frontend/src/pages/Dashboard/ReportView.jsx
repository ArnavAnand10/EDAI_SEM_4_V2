// src/pages/Dashboard/ReportView.jsx
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import Sidebar from '../../components/Sidebar';
import url from '../../utils/url';

const ReportView = () => {
  const { reportId } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [caseDetails, setCaseDetails] = useState(null);
  const [patientDetails, setPatientDetails] = useState(null);

  useEffect(() => {
    fetchReportDetails();
  }, [reportId]);

  const fetchReportDetails = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${url}reports/report?report_id=${reportId}`);
      setReport(response.data);
      
      // After getting report details, fetch the case information
    //   if (response.data.case_id) {
    //     fetchCaseDetails(response.data.case_id);
    //   }
    } catch (error) {
      console.error("Error fetching report:", error);
      toast.error("Failed to load report details");
    } finally {
      setLoading(false);
    }
  };

//   const fetchCaseDetails = async (caseId) => {
//     try {
//       const response = await axios.get(`${url}case?case_id=${caseId}`);
//       setCaseDetails(response.data);
      
//       // Now fetch patient details if we have a patient ID
//       if (response.data.patient_id) {
//         fetchPatientDetails(response.data.patient_id);
//       }
//     } catch (error) {
//       console.error("Error fetching case details:", error);
//       toast.error("Failed to load case information");
//     }
//   };

//   const fetchPatientDetails = async (patientId) => {
//     try {
//       const response = await axios.get(`${url}patient?patient_id=${patientId}`);
//       setPatientDetails(response.data);
//     } catch (error) {
//       console.error("Error fetching patient details:", error);
//       toast.error("Failed to load patient information");
//     }
//   };

  if (loading) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 p-8 flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 p-8 flex items-center justify-center">
          <div className="text-center">
            <h2 className="text-2xl font-semibold text-gray-800">Report Not Found</h2>
            <p className="mt-2 text-gray-600">The report you're looking for doesn't exist or you don't have permission to view it.</p>
            <button
              onClick={() => navigate('/dashboard')}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 p-8 overflow-auto">
        {/* Breadcrumbs */}
        <div className="flex items-center text-sm text-gray-600 mb-6">
          <button onClick={() => navigate('/dashboard')} className="hover:text-blue-600">Dashboard</button>
          <span className="mx-2">/</span>
         
         
          <span className="text-gray-800">Report #{reportId.substring(0, 8)}</span>
        </div>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Column - Image */}
          <div className="lg:w-1/2">
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Medical Image</h2>
              <div className="bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
                {report.file_url ? (
                  <img 
                    src={report.file_url} 
                    alt="Medical report" 
                    className="max-w-full max-h-[600px] object-contain"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = "https://via.placeholder.com/600x800?text=Image+Not+Available";
                    }}
                  />
                ) : (
                  <div className="h-96 flex items-center justify-center text-gray-500">
                    No image available
                  </div>
                )}
              </div>
              <div className="mt-4 flex justify-between items-center">
                <a 
                  href={report.file_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 font-medium flex items-center"
                >
                  <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                  </svg>
                  View Full Image
                </a>
                <span className="text-xs text-gray-500">
                  {report.created_at ? new Date(report.created_at).toLocaleString() : 'No date available'}
                </span>
              </div>
            </div>
          </div>

          {/* Right Column - Analysis */}
          <div className="lg:w-1/2">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">AI Analysis Summary</h2>
              
              {report.analysis_json ? (
                <div className="space-y-6">
                  {/* Findings */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Findings</h3>
                    <p className="mt-2 text-gray-700">{report.analysis_json.findings}</p>
                  </div>
                  
                  {/* Abnormalities */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Abnormalities</h3>
                    <ul className="mt-2 list-disc list-inside text-gray-700">
                      {report.analysis_json.abnormalities.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Possible Diagnoses */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Possible Diagnoses</h3>
                    <ul className="mt-2 list-disc list-inside text-gray-700">
                      {report.analysis_json.possible_diagnoses.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Areas of Concern */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Areas of Concern</h3>
                    <ul className="mt-2 list-disc list-inside text-gray-700">
                      {report.analysis_json.areas_of_concern.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Recommendations */}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">Recommendations</h3>
                    <ul className="mt-2 list-disc list-inside text-gray-700">
                      {report.analysis_json.recommendations.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="p-4 text-center">
                  <p className="text-gray-600">Analysis not available for this report.</p>
                </div>
              )}
              
              {/* Patient and Case Information */}
              {(patientDetails || caseDetails) && (
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Report Information</h3>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {patientDetails && (
                      <div>
                        <h4 className="font-medium text-gray-700">Patient</h4>
                        <p className="text-gray-600">{patientDetails.name}</p>
                        <p className="text-gray-600">{patientDetails.age} years, {patientDetails.gender}</p>
                      </div>
                    )}
                    
                    {caseDetails && (
                      <div>
                        <h4 className="font-medium text-gray-700">Case</h4>
                        <p className="text-gray-600">{caseDetails.title}</p>
                        <p className="text-gray-600 text-sm">{caseDetails.description}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-8 flex justify-between">
                <button
                  onClick={() => navigate(-1)}
                  className="text-blue-600 hover:text-blue-800 font-medium flex items-center"
                >
                  <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
                  </svg>
                  Back
                </button>
                {patientDetails && (
                  <button
                    onClick={() => navigate(`/dashboard/patients/${patientDetails._id}`)}
                    className="bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    View Patient Record
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportView;