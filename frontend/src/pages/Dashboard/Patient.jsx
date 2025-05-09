import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import Sidebar from '../../components/Sidebar';
import url from '../../utils/url';

const PatientProgressTracking = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [reports, setReports] = useState([]);
  const [progressSummary, setProgressSummary] = useState(null);
  const [loading, setLoading] = useState({
    patients: false,
    cases: false,
    reports: false,
    summary: false
  });

  // Check for authentication
  useEffect(() => {
    const doctorData = localStorage.getItem('doctorData');
    if (!doctorData) {
      toast.error('Please login first');
      navigate('/login');
      return;
    }
  }, [navigate]);

  // Fetch patients on component mount
  useEffect(() => {
    const fetchPatients = async () => {
      setLoading(prev => ({ ...prev, patients: true }));
      
      try {
        const doctorData = JSON.parse(localStorage.getItem('doctorData'));
        const doctor_id = doctorData.doctor_id;
        
        const response = await fetch(`${url}patients?doctor_id=${doctor_id}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch patients');
        }
        
        const data = await response.json();
        setPatients(data);
      } catch (error) {
        console.error('Error fetching patients:', error);
        toast.error('Failed to load patients');
      } finally {
        setLoading(prev => ({ ...prev, patients: false }));
      }
    };

    fetchPatients();
  }, []);

  // Fetch cases when a patient is selected
  useEffect(() => {
    if (selectedPatient) {
      const fetchCases = async () => {
        setLoading(prev => ({ ...prev, cases: true }));
        
        try {
          const response = await fetch(`${url}cases?patient_id=${selectedPatient._id}`);
          
          if (!response.ok) {
            throw new Error('Failed to fetch cases');
          }
          
          const data = await response.json();
          setCases(data);
        } catch (error) {
          console.error('Error fetching cases:', error);
          toast.error('Failed to load cases');
        } finally {
          setLoading(prev => ({ ...prev, cases: false }));
        }
      };

      fetchCases();
    } else {
      setCases([]);
      setSelectedCase(null);
    }
  }, [selectedPatient]);

  // Fetch reports when a case is selected
  useEffect(() => {
    if (selectedCase) {
      const fetchReports = async () => {
        setLoading(prev => ({ ...prev, reports: true }));
        
        try {
          const response = await fetch(`${url}reports?case_id=${selectedCase._id}`);
          
          if (!response.ok) {
            throw new Error('Failed to fetch reports');
          }
          
          const data = await response.json();
          setReports(data);
        } catch (error) {
          console.error('Error fetching reports:', error);
          toast.error('Failed to load reports');
        } finally {
          setLoading(prev => ({ ...prev, reports: false }));
        }
      };

      // fetchReports();
      
      // Fetch progress summary
      const fetchProgressSummary = async () => {
        setLoading(prev => ({ ...prev, summary: true }));
        
        try {
          const response = await fetch(`${url}progress?case_id=${selectedCase._id}`);
          
          if (!response.ok) {
            throw new Error('Failed to fetch progress summary');
          }
          
          const data = await response.json();
          setProgressSummary(data.progress_summary);
        } catch (error) {
          console.error('Error fetching progress summary:', error);
          toast.error('Failed to load progress summary');
        } finally {
          setLoading(prev => ({ ...prev, summary: false }));
        }
      };
      
      fetchProgressSummary();
    } else {
      setReports([]);
      setProgressSummary(null);
    }
  }, [selectedCase]);

  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
    setSelectedCase(null);
  };

  const handleCaseSelect = (caseItem) => {
    setSelectedCase(caseItem);
  };

  const viewReportDetails = (reportId) => {
    navigate(`/dashboard/report-details/${reportId}`);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      
      <div className="flex-1 p-8 overflow-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Patient Progress Tracking</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Patient List */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Your Patients</h2>
            
            {loading.patients ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
              </div>
            ) : patients.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No patients found</p>
            ) : (
              <ul className="divide-y divide-gray-200">
                {patients.map(patient => (
                  <li 
                    key={patient._id} 
                    className={`py-3 px-2 cursor-pointer hover:bg-gray-50 rounded ${selectedPatient?._id === patient._id ? 'bg-blue-50' : ''}`}
                    onClick={() => handlePatientSelect(patient)}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{patient.name}</p>
                        <p className="text-sm text-gray-500">Age: {patient.age}</p>
                      </div>
                      <span className="text-blue-500 text-sm">View</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          
          {/* Cases List */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">
              {selectedPatient 
                ? `Cases for ${selectedPatient.name}` 
                : 'Select a patient to view cases'}
            </h2>
            
            {selectedPatient ? (
              loading.cases ? (
                <div className="flex justify-center py-8">
                  <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                </div>
              ) : cases.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No cases found for this patient</p>
              ) : (
                <ul className="divide-y divide-gray-200">
                  {cases.map(caseItem => (
                    <li 
                      key={caseItem._id} 
                      className={`py-3 px-2 cursor-pointer hover:bg-gray-50 rounded ${selectedCase?._id === caseItem._id ? 'bg-blue-50' : ''}`}
                      onClick={() => handleCaseSelect(caseItem)}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="font-medium">{caseItem.title || 'Untitled Case'}</p>
                          <p className="text-sm text-gray-500">
                            {new Date(caseItem.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <span className="text-blue-500 text-sm">View</span>
                      </div>
                    </li>
                  ))}
                </ul>
              )
            ) : (
              <div className="py-8 text-center text-gray-500">
                <p>Please select a patient first</p>
              </div>
            )}
          </div>
          
          {/* Patient Progress Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">
              {selectedCase 
                ? `Progress Summary` 
                : 'Select a case to view progress'}
            </h2>
            
            {selectedCase ? (
              loading.summary ? (
                <div className="flex justify-center py-8">
                  <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                </div>
              ) : progressSummary ? (
                <div className="space-y-4">
                  {/* Overview */}
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h3 className="font-medium text-blue-800 mb-2">Overview</h3>
                    <p className="text-gray-700">{progressSummary.overview}</p>
                  </div>
                  
                  {/* Key Findings */}
                  <div>
                    <h3 className="font-medium text-gray-700 mb-2">Key Findings</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {progressSummary.key_findings.map((finding, index) => (
                        <li key={index} className="text-gray-600">{finding}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Next Steps */}
                  <div>
                    <h3 className="font-medium text-gray-700 mb-2">Recommended Next Steps</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {progressSummary.next_steps.map((step, index) => (
                        <li key={index} className="text-gray-600">{step}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Status */}
                  <div className="flex items-center">
                    <span className="font-medium mr-2">Status:</span>
                    <span className={`px-2 py-1 rounded text-sm ${
                      progressSummary.trend === 'improving' ? 'bg-green-100 text-green-800' :
                      progressSummary.trend === 'stable' ? 'bg-blue-100 text-blue-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {progressSummary.trend}
                    </span>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No progress data available</p>
              )
            ) : (
              <div className="py-8 text-center text-gray-500">
                <p>Please select a case first</p>
              </div>
            )}
          </div>
          
        
        </div>
      </div>
    </div>
  );
};

export default PatientProgressTracking;