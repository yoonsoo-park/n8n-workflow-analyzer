import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { 
  Upload, 
  BarChart3, 
  Network, 
  FileJson, 
  Workflow, 
  Home,
  Settings,
  FileAnalytics,
  Layers,
  LayoutDashboard
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from '@/components/ui/card'

import './App.css'

// Import pages
import Dashboard from './pages/Dashboard'
import UploadWorkflows from './pages/UploadWorkflows'
import WorkflowAnalysis from './pages/WorkflowAnalysis'
import PatternAnalysis from './pages/PatternAnalysis'
import NetworkAnalysis from './pages/NetworkAnalysis'
import SessionList from './pages/SessionList'

function App() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768)
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <Router>
      <div className="app-container">
        <aside className={`sidebar ${isMobile ? 'mobile' : ''}`}>
          <div className="sidebar-header">
            <h2 className="sidebar-title">n8n Analyzer</h2>
          </div>
          <nav className="sidebar-nav">
            <ul>
              <li>
                <Link to="/" className="nav-link">
                  <Home size={20} />
                  <span>Dashboard</span>
                </Link>
              </li>
              <li>
                <Link to="/upload" className="nav-link">
                  <Upload size={20} />
                  <span>Upload Workflows</span>
                </Link>
              </li>
              <li>
                <Link to="/sessions" className="nav-link">
                  <FileJson size={20} />
                  <span>Sessions</span>
                </Link>
              </li>
              <li className="nav-section">Analysis</li>
              <li>
                <Link to="/workflow-analysis" className="nav-link">
                  <Workflow size={20} />
                  <span>Workflow Analysis</span>
                </Link>
              </li>
              <li>
                <Link to="/pattern-analysis" className="nav-link">
                  <Layers size={20} />
                  <span>Pattern Analysis</span>
                </Link>
              </li>
              <li>
                <Link to="/network-analysis" className="nav-link">
                  <Network size={20} />
                  <span>Network Analysis</span>
                </Link>
              </li>
            </ul>
          </nav>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<UploadWorkflows />} />
            <Route path="/sessions" element={<SessionList />} />
            <Route path="/workflow-analysis" element={<WorkflowAnalysis />} />
            <Route path="/workflow-analysis/:sessionId" element={<WorkflowAnalysis />} />
            <Route path="/pattern-analysis" element={<PatternAnalysis />} />
            <Route path="/pattern-analysis/:sessionId" element={<PatternAnalysis />} />
            <Route path="/network-analysis" element={<NetworkAnalysis />} />
            <Route path="/network-analysis/:sessionId" element={<NetworkAnalysis />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App

