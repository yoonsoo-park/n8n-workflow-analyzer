import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  FileJson, 
  Trash2, 
  ArrowRight, 
  Loader2,
  AlertCircle,
  CheckCircle2
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
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

const API_URL = 'http://localhost:5000/api'

const SessionList = () => {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [deleteSessionId, setDeleteSessionId] = useState(null)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [deleteLoading, setDeleteLoading] = useState(false)
  
  const fetchSessions = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${API_URL}/sessions`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch sessions')
      }
      
      const data = await response.json()
      
      // Sort by creation date (newest first)
      const sortedSessions = data.sort((a, b) => b.created_at - a.created_at)
      
      setSessions(sortedSessions)
      setError(null)
    } catch (error) {
      console.error('Error fetching sessions:', error)
      setError('Failed to load sessions. Please try again later.')
    } finally {
      setLoading(false)
    }
  }
  
  useEffect(() => {
    fetchSessions()
  }, [])
  
  const handleDeleteClick = (sessionId) => {
    setDeleteSessionId(sessionId)
    setDeleteDialogOpen(true)
  }
  
  const handleDeleteConfirm = async () => {
    if (!deleteSessionId) return
    
    setDeleteLoading(true)
    
    try {
      const response = await fetch(`${API_URL}/sessions/${deleteSessionId}`, {
        method: 'DELETE'
      })
      
      if (!response.ok) {
        throw new Error('Failed to delete session')
      }
      
      // Remove the deleted session from the list
      setSessions(sessions.filter(session => session.session_id !== deleteSessionId))
      setDeleteDialogOpen(false)
    } catch (error) {
      console.error('Error deleting session:', error)
      setError('Failed to delete session. Please try again later.')
    } finally {
      setDeleteLoading(false)
      setDeleteSessionId(null)
    }
  }
  
  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString()
  }
  
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analysis Sessions</h1>
        <p className="page-description">
          Manage your workflow analysis sessions
        </p>
      </div>
      
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <div className="mb-6 flex justify-end">
        <Button asChild>
          <Link to="/upload">
            Upload New Workflows
          </Link>
        </Button>
      </div>
      
      {loading ? (
        <div className="loading-spinner">
          <Loader2 className="h-12 w-12 text-primary" />
        </div>
      ) : sessions.length > 0 ? (
        <div className="space-y-4">
          {sessions.map((session) => (
            <Card key={session.session_id} className="session-card">
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle>Session {session.session_id.substring(0, 8)}</CardTitle>
                    <CardDescription>
                      Created on {formatDate(session.created_at)}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDeleteClick(session.session_id)}
                  >
                    <Trash2 className="h-5 w-5 text-red-500" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center mb-4">
                  <FileJson className="h-5 w-5 text-primary mr-2" />
                  <span>
                    {session.file_count} workflow{session.file_count !== 1 ? 's' : ''}
                  </span>
                </div>
                
                <div className="flex items-center">
                  {session.has_analysis ? (
                    <div className="flex items-center text-green-600">
                      <CheckCircle2 className="h-5 w-5 mr-2" />
                      <span>Analysis complete</span>
                    </div>
                  ) : (
                    <div className="flex items-center text-amber-600">
                      <AlertCircle className="h-5 w-5 mr-2" />
                      <span>Not analyzed yet</span>
                    </div>
                  )}
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                {session.has_analysis ? (
                  <div className="grid grid-cols-3 gap-2 w-full">
                    <Button asChild variant="outline" size="sm">
                      <Link to={`/workflow-analysis/${session.session_id}`}>
                        Workflow Analysis
                      </Link>
                    </Button>
                    <Button asChild variant="outline" size="sm">
                      <Link to={`/pattern-analysis/${session.session_id}`}>
                        Pattern Analysis
                      </Link>
                    </Button>
                    <Button asChild variant="outline" size="sm">
                      <Link to={`/network-analysis/${session.session_id}`}>
                        Network Analysis
                      </Link>
                    </Button>
                  </div>
                ) : (
                  <Button asChild className="w-full">
                    <Link to={`/workflow-analysis/${session.session_id}`}>
                      Analyze Now
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                )}
              </CardFooter>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileJson className="h-16 w-16 text-gray-300 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Sessions Found</h3>
            <p className="text-gray-500 text-center mb-6">
              You haven't uploaded any workflow files yet.
              <br />
              Upload some workflows to get started.
            </p>
            <Button asChild>
              <Link to="/upload">
                Upload Workflows
              </Link>
            </Button>
          </CardContent>
        </Card>
      )}
      
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this session and all its associated workflows and analysis results.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleteLoading}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              disabled={deleteLoading}
              className="bg-red-500 hover:bg-red-600"
            >
              {deleteLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

export default SessionList

