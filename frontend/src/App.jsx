// src/App.jsx

import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import Forum from "./pages/Dashboard/Forum";
import AnalyzeReport from "./pages/Dashboard/AnalyzeReport";
import ReportView from "./pages/Dashboard/ReportView";
import PatientProgressTracking from "./pages/Dashboard/Patient";


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/dashboard/patients" element={<PatientProgressTracking />} />
        <Route path="/dashboard/forum" element={<Forum  />} />
        <Route path="/dashboard/reports/:reportId" element={<ReportView  />} />

 
        <Route path="/dashboard/analyze-report" element={<AnalyzeReport />} />
        <Route path="/" element={<Login />} />
      </Routes>
    </Router>
  );
}

export default App;