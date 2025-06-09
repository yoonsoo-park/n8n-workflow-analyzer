import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  BarChart3, 
  Network, 
  FileJson, 
  Upload, 
  ArrowRight,
  Loader2
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

const API_URL = 'http://localhost:5000/api'

const Dashboard = () => {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({
    totalSessions: 0,
    totalWorkflows: 0,
    averageNodesPerWorkflow: 0,
    averageConnectionsPerWorkflow: 0
  })

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await fetch(`${API_URL}/sessions`)
        const data = await response.json()
        
        // Sort by creation date (newest first)
        const sortedSessions = data.sort((a, b) => b.created_at - a.created_at)
        
        // Take only the 5 most recent sessions
        const recentSessions = sortedSessions.slice(0, 5)
        
        setSessions(recentSessions)
        
        // Calculate stats
        setStats({
          totalSessions: data.length,
          totalWorkflows: data.reduce((sum, session) => sum + session.file_count, 0),
          averageNodesPerWorkflow: 0, // Would need additional API call
          averageConnectionsPerWorkflow: 0 // Would need additional API call
        })
        
        setLoading(false)
      } catch (error) {
        console.error('Error fetching sessions:', error)
        setLoading(false)
      }
    }
    
    fetchSessions()
  }, [])

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-description">
          Overview of your n8n workflow analysis
        </p>
      </div>
      
      <div className="stats-grid">
        <Card>
          <CardContent className="pt-6">
            <div className="stat-value">{stats.totalSessions}</div>
            <div className="stat-label">Total Sessions</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="stat-value">{stats.totalWorkflows}</div>
            <div className="stat-label">Total Workflows</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="stat-value">{stats.averageNodesPerWorkflow}</div>
            <div className="stat-label">Avg. Nodes per Workflow</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <div className="stat-value">{stats.averageConnectionsPerWorkflow}</div>
            <div className="stat-label">Avg. Connections per Workflow</div>
          </CardContent>
        </Card>
      </div>
      
      <div className="card-grid">
        <Card>
          <CardHeader>
            <CardTitle>Upload Workflows</CardTitle>
            <CardDescription>
              Upload n8n workflow JSON files for analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-24">
              <Upload size={48} className="text-primary" />
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild className="w-full">
              <Link to="/upload">
                Upload Workflows
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Workflow Analysis</CardTitle>
            <CardDescription>
              Analyze workflow structure and complexity
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-24">
              <BarChart3 size={48} className="text-primary" />
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild variant="outline" className="w-full">
              <Link to="/workflow-analysis">
                View Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Pattern Analysis</CardTitle>
            <CardDescription>
              Discover frequent patterns and rules
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-24">
              <FileJson size={48} className="text-primary" />
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild variant="outline" className="w-full">
              <Link to="/pattern-analysis">
                View Patterns
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Network Analysis</CardTitle>
            <CardDescription>
              Analyze workflow network properties
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center h-24">
              <Network size={48} className="text-primary" />
            </div>
          </CardContent>
          <CardFooter>
            <Button asChild variant="outline" className="w-full">
              <Link to="/network-analysis">
                View Networks
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Recent Sessions</CardTitle>
          <CardDescription>
            Your most recent workflow analysis sessions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : sessions.length > 0 ? (
            <div className="space-y-4">
              {sessions.map((session) => (
                <Card key={session.session_id} className="session-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Session {session.session_id.substring(0, 8)}</CardTitle>
                    <CardDescription>
                      {new Date(session.created_at * 1000).toLocaleString()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pb-2">
                    <p className="text-sm">
                      {session.file_count} workflow{session.file_count !== 1 ? 's' : ''}
                    </p>
                  </CardContent>
                  <CardFooter>
                    {session.has_analysis ? (
                      <Button asChild variant="outline" size="sm" className="w-full">
                        <Link to={`/workflow-analysis/${session.session_id}`}>
                          View Analysis
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    ) : (
                      <Button asChild size="sm" className="w-full">
                        <Link to={`/analyze/${session.session_id}`}>
                          Analyze
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    )}
                  </CardFooter>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500">
              <p>No sessions found. Upload some workflows to get started.</p>
            </div>
          )}
        </CardContent>
        <CardFooter>
          <Button asChild variant="outline" className="w-full">
            <Link to="/sessions">
              View All Sessions
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}

export default Dashboard

