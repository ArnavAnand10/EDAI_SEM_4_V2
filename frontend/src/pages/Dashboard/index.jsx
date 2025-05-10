// src/pages/Dashboard/index.jsx (Updated)
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../../components/Sidebar';
import DashboardCard from '../../components/DashboardCard';
import { toast } from 'react-hot-toast';

const Dashboard = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const doctorData = localStorage.getItem('doctorData');
    if (!doctorData) {
      toast.error('Please login first');
      navigate('/login');
    }
  }, [navigate]);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      
      <div className="flex-1 p-8 overflow-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Doctor Dashboard</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <DashboardCard 
            title="Patient Progress Tracking"
            description="Monitor and track your patients' treatment progress"
            icon="📈"
            link="/dashboard/patients"
          />
          
          <DashboardCard 
            title="Doctors Community Forum"
            description="Discuss cases and share knowledge with other doctors"
            icon="👥"
            link="/dashboard/forum"
          />
          
          <DashboardCard 
            title="Analyze Medical Report"
            description="Upload and analyze medical reports for new or existing patients"
            icon="🔍"
            link="/dashboard/analyze-report"
          />

          <DashboardCard 
            title="Generate Tamper-Proof Report"
            description="Create secure medical documents with embedded anti-tampering features"
            icon="📄🔒"
            link="/dashboard/secure-image"
          />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
