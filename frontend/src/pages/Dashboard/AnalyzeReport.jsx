// src/pages/Dashboard/AnalyzeReport.jsx
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import Sidebar from '../../components/Sidebar';
import url from '../../utils/url';

const AnalyzeReport = () => {
  const [recordType, setRecordType] = useState(null); // 'new' or 'existing'
  const [step, setStep] = useState(1);
  const [patientData, setPatientData] = useState({
    name: '',
    age: '',
    gender: 'male',
    address: ''
  });
  const [caseData, setCaseData] = useState({
    title: '',
    description: '',
  });
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reportAction, setReportAction] = useState(null); // 'view' or 'new'
  const navigate = useNavigate();
  
  // For existing records
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [cases, setCases] = useState([]);
  const [selectedCaseId, setSelectedCaseId] = useState('');
  const [reports, setReports] = useState([]);
  const [selectedReportId, setSelectedReportId] = useState('');
  const [loadingPatients, setLoadingPatients] = useState(false);
  const [loadingCases, setLoadingCases] = useState(false);
  const [loadingReports, setLoadingReports] = useState(false);

  // Fetch patients when component mounts
  useEffect(() => {
    if (recordType === 'existing') {
      fetchPatients();
    }
  }, [recordType]);

  // Fetch cases when a patient is selected
  useEffect(() => {
    if (selectedPatientId) {
      fetchCases(selectedPatientId);
    }
  }, [selectedPatientId]);

  // Fetch reports when a case is selected
  useEffect(() => {
    if (selectedCaseId) {
      fetchReports(selectedCaseId);
    }
  }, [selectedCaseId]);

  const fetchPatients = async () => {
    setLoadingPatients(true);
    try {
      const doctor_id = JSON.parse(localStorage.getItem('doctorData')).doctor_id;
      const response = await axios.get(`${url}patients?doctor_id=${doctor_id}`);
      console.log(response.data, "patients");
      
      setPatients(response.data);
    } catch (error) {
      toast.error('Failed to fetch patients');
      console.error(error);
    } finally {
      setLoadingPatients(false);
    }
  };

  const fetchCases = async (patientId) => {
    console.log(patientId, "fetching cases");
    
    setLoadingCases(true);
    try {
      const response = await axios.get(`${url}cases?patient_id=${patientId}`);
      setCases(response.data);
    } catch (error) {
      toast.error('Failed to fetch cases');
      console.error(error);
    } finally {
      setLoadingCases(false);
    }
  };

  const fetchReports = async (caseId) => {
    setLoadingReports(true);
    console.log(caseId, "caseId");
    
    try {
      const response = await axios.get(`${url}reports?case_id=${caseId}`);
      
      // The response now contains report_id and file_url directly
      // Example: [{ "report_id": "681df194d5ec235705d6e179", "file_url": "https://..." }, ...]
      console.log("Reports fetched:", response.data);
      setReports(response.data);
    } catch (error) {
      toast.error('Failed to fetch reports');
      console.error(error);
    } finally {
      setLoadingReports(false);
    }
  };

  const handlePatientSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    let doctor_id = JSON.parse(localStorage.getItem('doctorData')).doctor_id;
    
    try {
      const response = await axios.post(`${url}patients?doctor_id=${doctor_id}`, patientData);
      
      toast.success('Patient created successfully!');
      console.log(response.data.id);
      
      setPatientData(prev => ({ ...prev, id: response.data.id }));
      setStep(2);
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to create patient');
    } finally {
      setLoading(false);
    }
  };

  const handleCaseSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const pId = patientData.id;
    
    try {
      const response = await axios.post(`${url}cases?patient_id=${pId}`, 
      {title: caseData.title, description: caseData.description}, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      toast.success('Case created successfully!');
      setCaseData(prev => ({ ...prev, id: response.data._id }));
      setStep(3);
    } catch (error) {
      toast.error(error.response?.data?.message || 'Failed to create case');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const cloudinaryUpload = await uploadToCloudinary(file);
      
      // The case ID can come from either new case creation or existing case selection
      const cId = recordType === 'new' ? caseData.id : selectedCaseId;
      
      const response = await axios.post(`${url}reports?case_id=${cId}`, {
        file_url: cloudinaryUpload.secure_url,
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      console.log(response.data,"response from the image analysis")
      toast.success('Report uploaded and analysis started!');
      // Redirect to the patient page - either newly created patient or selected existing patient
      // const patientId = recordType === 'new' ? patientData.id : selectedPatientId;
      navigate(`/dashboard/reports/${response.data.report_id}`);
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.message || 'Failed to process report');
    } finally {
      setLoading(false);
    }
  };

  const uploadToCloudinary = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('upload_preset', 'newPreset'); 
    
    const response = await axios.post(
      `https://api.cloudinary.com/v1_1/duwjf9ify/upload`, 
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    
    return response.data;
  };

  const handlePatientSelect = (e) => {
    setSelectedPatientId(e.target.value);
    // Reset case and report selection when patient changes
    setSelectedCaseId('');
    setSelectedReportId('');
    setReportAction(null);
    if (e.target.value) {
      setStep(2);
    }
  };

  const handleCaseSelect = (e) => {
    setSelectedCaseId(e.target.value);
    setSelectedReportId('');
    setReportAction(null);
    if (e.target.value) {
      setStep(3);
    }
  };

  const handleReportSelect = (e) => {
    setSelectedReportId(e.target.value);
  };

  const viewReport = (reportId) => {
    // Navigate to report view page using the report_id from the backend response
    console.log("Viewing report:", reportId);
    navigate(`/dashboard/reports/${reportId}`);
  };
  

  const resetForm = () => {
    setRecordType(null);
    setStep(1);
    setPatientData({
      name: '',
      age: '',
      gender: 'male',
      address: ''
    });
    setCaseData({
      title: '',
      description: ''
    });
    setFile(null);
    setSelectedPatientId('');
    setSelectedCaseId('');
    setSelectedReportId('');
    setReportAction(null);
  };

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'No date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Render record type selection
  if (!recordType) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <div className="flex-1 p-8 overflow-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">Analyze Medical Report</h1>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div 
              className="bg-white rounded-lg shadow-md p-6 cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setRecordType('new')}
            >
              <div className="h-16 w-16 bg-blue-100 rounded-full flex items-center justify-center mb-4 mx-auto">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-center mb-2">Create New Record</h2>
              <p className="text-gray-600 text-center">Create a new patient, case, and upload a report for analysis</p>
            </div>
            
            <div 
              className="bg-white rounded-lg shadow-md p-6 cursor-pointer hover:shadow-lg transition-shadow"
              onClick={() => setRecordType('existing')}
            >
              <div className="h-16 w-16 bg-green-100 rounded-full flex items-center justify-center mb-4 mx-auto">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-center mb-2">Use Existing Record</h2>
              <p className="text-gray-600 text-center">Select an existing patient and case to add a new report for analysis</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 p-8 overflow-auto">
        <div className="flex items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">
            {recordType === 'new' ? 'Create New Report' : 'Add Report to Existing Record'}
          </h1>
          <button
            onClick={resetForm}
            className="ml-4 px-3 py-1 text-sm text-blue-600 border border-blue-600 rounded hover:bg-blue-50"
          >
            Change Record Type
          </button>
          <div className="ml-auto flex items-center space-x-4">
            <div className={`flex items-center ${step >= 1 ? 'text-blue-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-blue-100' : 'bg-gray-100'}`}>
                1
              </div>
              <span className="ml-2">Patient</span>
            </div>
            <div className="h-px w-8 bg-gray-300"></div>
            <div className={`flex items-center ${step >= 2 ? 'text-blue-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-blue-100' : 'bg-gray-100'}`}>
                2
              </div>
              <span className="ml-2">Case</span>
            </div>
            <div className="h-px w-8 bg-gray-300"></div>
            <div className={`flex items-center ${step >= 3 ? 'text-blue-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? 'bg-blue-100' : 'bg-gray-100'}`}>
                3
              </div>
              <span className="ml-2">Report</span>
            </div>
          </div>
        </div>

        {/* NEW RECORD FLOW */}
        {recordType === 'new' && step === 1 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">Create New Patient Record</h2>
            <form onSubmit={handlePatientSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-gray-700 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={patientData.name}
                    onChange={(e) => setPatientData({...patientData, name: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Age</label>
                  <input
                    type="number"
                    value={patientData.age}
                    onChange={(e) => setPatientData({...patientData, age: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Gender</label>
                  <select
                    value={patientData.gender}
                    onChange={(e) => setPatientData({...patientData, gender: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Address (Optional)</label>
                  <input
                    type="text"
                    value={patientData.address}
                    onChange={(e) => setPatientData({...patientData, address: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                  />
                </div>
              </div>
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
              >
                {loading ? 'Creating...' : 'Create Patient & Continue'}
              </button>
            </form>
          </div>
        )}

        {recordType === 'new' && step === 2 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">Create New Case for {patientData.name}</h2>
            <form onSubmit={handleCaseSubmit}>
              <div className="space-y-4 mb-6">
                <div>
                  <label className="block text-gray-700 mb-2">Case Title</label>
                  <input
                    type="text"
                    value={caseData.title}
                    onChange={(e) => setCaseData({...caseData, title: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                    required
                  />
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Description</label>
                  <textarea
                    value={caseData.description}
                    onChange={(e) => setCaseData({...caseData, description: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-lg"
                    rows="3"
                    required
                  />
                </div>
              </div>
              <div className="flex justify-between">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="text-blue-600 hover:text-blue-800 font-medium"
                >
                  ← Back to Patient Details
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
                >
                  {loading ? 'Creating...' : 'Create Case & Continue'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* EXISTING RECORD FLOW */}
        {recordType === 'existing' && step === 1 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">Select Existing Patient</h2>
            <div className="mb-6">
              <label className="block text-gray-700 mb-2">Patient</label>
              {loadingPatients ? (
                <div className="flex items-center justify-center p-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <select
                  value={selectedPatientId}
                  onChange={handlePatientSelect}
                  className="w-full p-3 border border-gray-300 rounded-lg"
                  required
                >
                  <option value="">Select a patient</option>
                  {patients.map(patient => (
                    <option key={patient.id} value={patient._id}>
                      {patient.name} - Age: {patient.age}, Gender: {patient.gender}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="flex justify-between">
              <button
                type="button"
                onClick={resetForm}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Change Record Type
              </button>
              <button
                type="button"
                disabled={!selectedPatientId}
                onClick={() => setStep(2)}
                className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
              >
                Continue to Case Selection
              </button>
            </div>
          </div>
        )}

        {recordType === 'existing' && step === 2 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">Select Existing Case</h2>
            <div className="mb-6">
              <label className="block text-gray-700 mb-2">Case</label>
              {loadingCases ? (
                <div className="flex items-center justify-center p-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
                </div>
              ) : cases.length === 0 ? (
                <div className="text-center p-4">
                  <p className="text-gray-600">No cases found for this patient.</p>
                  <button
                    type="button"
                    onClick={() => {
                      // Set up a new case for this existing patient
                      setPatientData(prev => ({ ...prev, id: selectedPatientId }));
                      setRecordType('new');
                      setStep(2);
                    }}
                    className="mt-3 text-blue-600 hover:text-blue-800 font-medium"
                  >
                    + Create new case for this patient
                  </button>
                </div>
              ) : (
                <select
                  value={selectedCaseId}
                  onChange={handleCaseSelect}
                  className="w-full p-3 border border-gray-300 rounded-lg"
                  required
                >
                  <option value="">Select a case</option>
                  {cases.map(case_item => (
                    <option key={case_item._id} value={case_item._id}>
                      {case_item.title} - {new Date(case_item.created_at).toLocaleDateString()}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="flex justify-between">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Back to Patient Selection
              </button>
              <button
                type="button"
                disabled={!selectedCaseId}
                onClick={() => setStep(3)}
                className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
              >
                Continue to Reports
              </button>
            </div>
          </div>
        )}

        {/* Reports Section - Modified for Existing Flow */}
        {recordType === 'existing' && step === 3 && !reportAction && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">
              Case Reports
            </h2>
            
            {/* Existing Reports */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4">Existing Reports</h3>
              {loadingReports ? (
                <div className="flex items-center justify-center p-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600"></div>
                </div>
              ) : reports.length === 0 ? (
                <p className="text-gray-600 text-center p-4">No reports found for this case.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {reports.map(report => (
                    <div 
                      key={report.report_id}
                      onClick={() => viewReport(report.report_id)}
                      className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer overflow-hidden"
                    >
                      <div className="relative">
                        {report.file_url && (
                          <img 
                            src={report.file_url} 
                            alt="Report preview" 
                            className="w-full h-48 object-cover"
                            onError={(e) => {
                              e.target.onerror = null;
                              e.target.src = "https://via.placeholder.com/400x300?text=No+Preview";
                            }}
                          />
                        )}
                        <div className={`absolute top-2 right-2 px-2 py-1 text-xs font-semibold rounded-full ${
                          report.status === 'completed' ? 'bg-green-100 text-green-800' : 
                          report.status === 'processing' ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {report.status || 'pending'}
                        </div>
                      </div>
                      
                      <div className="p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-medium text-gray-900">Report #{report.report_id.substring(0, 8)}</h3>
                          <span className="text-xs text-gray-500">{report.created_at ? formatDate(report.created_at) : 'No date'}</span>
                        </div>
                        
                        <button 
                          className="mt-2 w-full bg-blue-50 hover:bg-blue-100 text-blue-600 font-medium py-2 px-4 rounded-md transition-colors"
                        >
                          View Details
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Add New Report Button */}
            <div className="flex flex-col items-center justify-center p-6 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">Add a New Report to This Case</h3>
              <p className="text-gray-600 mb-4">Upload a new medical report for analysis</p>
              <button
                onClick={() => setReportAction('new')}
                className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Upload New Report
              </button>
            </div>
            
            <div className="flex justify-between mt-6">
              <button
                type="button"
                onClick={() => setStep(2)}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Back to Case Selection
              </button>
            </div>
          </div>
        )}

        {/* Upload Report - For New Reports in Both Flows */}
        {(step === 3 && recordType === 'new') || (step === 3 && recordType === 'existing' && reportAction === 'new') ? (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-6">
              Upload Medical Report
              {recordType === 'new' && ` for ${caseData.title}`}
            </h2>
            <form onSubmit={handleFileUpload}>
              <div className="mb-6">
                <label className="block text-gray-700 mb-2">Select Report File</label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <input
                    type="file"
                    accept=".pdf,.jpg,.png,.dicom"
                    className="hidden"
                    id="report-upload"
                    onChange={(e) => setFile(e.target.files[0])}
                  />
                  <label htmlFor="report-upload" className="cursor-pointer block">
                    <div className="mx-auto w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
                      <span className="text-2xl">📄</span>
                    </div>
                    {file ? (
                      <p className="text-gray-600 font-medium">{file.name}</p>
                    ) : (
                      <>
                        <p className="text-gray-600">Drag and drop files here or click to browse</p>
                        <p className="text-sm text-gray-500 mt-1">Supported formats: PDF, JPG, PNG, DICOM</p>
                      </>
                    )}
                  </label>
                </div>
              </div>
              <div className="flex justify-between">
                <button
                  type="button"
                  onClick={() => recordType === 'existing' ? setReportAction(null) : setStep(2)}
                  className="text-blue-600 hover:text-blue-800 font-medium"
                >
                  ← Back
                </button>
                <button
                  type="submit"
                  disabled={!file || loading}
                  className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-400"
                >
                  {loading ? 'Uploading...' : 'Upload & Analyze Report'}
                </button>
              </div>
            </form>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default AnalyzeReport;