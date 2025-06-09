import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { 
  BarChart3, 
  Workflow, 
  ArrowLeft, 
  Loader2,
  AlertCircle,
  Download
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
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const API_URL = 'http://localhost:5000/api'

const WorkflowAnalysis = () => {
  const { sessionId } = useParams()
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [selectedWorkflow, setSelectedWorkflow] = useState(null)
  
  useEffect(() => {
    const fetchResults = async () => {
      if (!sessionId) return
      
      try {
        setLoading(true)
        
        // First try to get existing results
        const resultsResponse = await fetch(`${API_URL}/results/${sessionId}`)
        
        if (resultsResponse.ok) {
          const data = await resultsResponse.json()
          setAnalysisResults(data)
          
          // Set the first workflow as selected by default
          if (data.workflow_analyses && Object.keys(data.workflow_analyses).length > 0) {
            setSelectedWorkflow(Object.keys(data.workflow_analyses)[0])
          }
          
          setLoading(false)
          return
        }
        
        // If no results exist, trigger analysis
        setAnalyzing(true)
        const analysisResponse = await fetch(`${API_URL}/analyze/${sessionId}`)
        
        if (!analysisResponse.ok) {
          const errorData = await analysisResponse.json()
          throw new Error(errorData.error || 'Analysis failed')
        }
        
        const data = await analysisResponse.json()
        setAnalysisResults(data)
        
        // Set the first workflow as selected by default
        if (data.workflow_analyses && Object.keys(data.workflow_analyses).length > 0) {
          setSelectedWorkflow(Object.keys(data.workflow_analyses)[0])
        }
        
      } catch (error) {
        console.error('Error:', error)
        setError(error.message || 'Failed to analyze workflows')
      } finally {
        setLoading(false)
        setAnalyzing(false)
      }
    }
    
    fetchResults()
  }, [sessionId])
  
  if (loading || analyzing) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Workflow Analysis</h1>
          <p className="page-description">
            {analyzing ? 'Analyzing workflows...' : 'Loading analysis results...'}
          </p>
        </div>
        
        <div className="loading-spinner">
          <Loader2 className="h-12 w-12 text-primary" />
          <p className="mt-4 text-center text-gray-500">
            {analyzing ? 
              'This may take a few moments depending on the number and complexity of workflows.' :
              'Loading analysis results...'}
          </p>
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Workflow Analysis</h1>
          <p className="page-description">
            Error loading analysis
          </p>
        </div>
        
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        
        <Button asChild>
          <Link to="/sessions">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Sessions
          </Link>
        </Button>
      </div>
    )
  }
  
  if (!analysisResults) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Workflow Analysis</h1>
          <p className="page-description">
            No analysis results found
          </p>
        </div>
        
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-16 w-16 text-gray-300 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Analysis Results</h3>
            <p className="text-gray-500 text-center mb-6">
              No analysis results were found for this session.
              <br />
              Try analyzing the workflows again.
            </p>
            <div className="flex space-x-4">
              <Button asChild variant="outline">
                <Link to="/sessions">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Sessions
                </Link>
              </Button>
              <Button onClick={() => window.location.reload()}>
                Analyze Again
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }
  
  const workflowIds = Object.keys(analysisResults.workflow_analyses)
  const currentWorkflow = analysisResults.workflow_analyses[selectedWorkflow]
  
  return (
    <div>
      <div className="page-header">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="page-title">Workflow Analysis</h1>
            <p className="page-description">
              Analysis results for session {sessionId.substring(0, 8)}
            </p>
          </div>
          <div className="flex space-x-2">
            <Button asChild variant="outline">
              <Link to="/sessions">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Sessions
              </Link>
            </Button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Workflows</CardTitle>
            <CardDescription>
              {workflowIds.length} workflow{workflowIds.length !== 1 ? 's' : ''} analyzed
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {workflowIds.map((id) => (
                <div
                  key={id}
                  className={`p-3 rounded-md cursor-pointer transition-colors ${
                    selectedWorkflow === id
                      ? 'bg-primary text-white'
                      : 'bg-gray-50 hover:bg-gray-100'
                  }`}
                  onClick={() => setSelectedWorkflow(id)}
                >
                  <div className="flex items-center">
                    <Workflow className={`h-5 w-5 ${selectedWorkflow === id ? 'text-white' : 'text-primary'} mr-2`} />
                    <div>
                      <div className="font-medium text-sm">
                        {analysisResults.workflow_analyses[id].name || `Workflow ${id.substring(0, 8)}`}
                      </div>
                      <div className={`text-xs ${selectedWorkflow === id ? 'text-white/80' : 'text-gray-500'}`}>
                        {analysisResults.workflow_analyses[id].node_count} nodes, {analysisResults.workflow_analyses[id].connection_count} connections
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        <Card className="md:col-span-3">
          <CardHeader>
            <CardTitle>
              {currentWorkflow.name || `Workflow ${selectedWorkflow.substring(0, 8)}`}
            </CardTitle>
            <CardDescription>
              Detailed analysis results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="metrics">
              <TabsList className="mb-4">
                <TabsTrigger value="metrics">Metrics</TabsTrigger>
                <TabsTrigger value="visualization">Visualization</TabsTrigger>
              </TabsList>
              
              <TabsContent value="metrics">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Complexity Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <div className="text-sm font-medium mb-1">Structural Complexity</div>
                          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${Math.min(currentWorkflow.metrics.complexity.structural_complexity * 10, 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Low</span>
                            <span>{currentWorkflow.metrics.complexity.structural_complexity.toFixed(2)}</span>
                            <span>High</span>
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-sm font-medium mb-1">Cognitive Complexity</div>
                          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${Math.min(currentWorkflow.metrics.complexity.cognitive_complexity * 10, 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Low</span>
                            <span>{currentWorkflow.metrics.complexity.cognitive_complexity.toFixed(2)}</span>
                            <span>High</span>
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-sm font-medium mb-1">Cyclomatic Complexity</div>
                          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${Math.min(currentWorkflow.metrics.complexity.cyclomatic_complexity * 10, 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Low</span>
                            <span>{currentWorkflow.metrics.complexity.cyclomatic_complexity.toFixed(2)}</span>
                            <span>High</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Structure Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <div className="text-sm font-medium mb-1">Error Handling Coverage</div>
                          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${Math.min(currentWorkflow.metrics.structure.error_handling_coverage * 100, 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>0%</span>
                            <span>{(currentWorkflow.metrics.structure.error_handling_coverage * 100).toFixed(1)}%</span>
                            <span>100%</span>
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-sm font-medium mb-1">Branching Factor</div>
                          <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${Math.min(currentWorkflow.metrics.structure.branching_factor * 33.3, 100)}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Low</span>
                            <span>{currentWorkflow.metrics.structure.branching_factor.toFixed(2)}</span>
                            <span>High</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Workflow Structure</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-gray-50 p-4 rounded-md">
                        <div className="text-2xl font-bold text-primary">
                          {currentWorkflow.node_count}
                        </div>
                        <div className="text-sm text-gray-500">Total Nodes</div>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-md">
                        <div className="text-2xl font-bold text-primary">
                          {currentWorkflow.connection_count}
                        </div>
                        <div className="text-sm text-gray-500">Connections</div>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-md">
                        <div className="text-2xl font-bold text-primary">
                          {(currentWorkflow.connection_count / (currentWorkflow.node_count || 1)).toFixed(2)}
                        </div>
                        <div className="text-sm text-gray-500">Connections per Node</div>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-md">
                        <div className="text-2xl font-bold text-primary">
                          {currentWorkflow.metrics.complexity.structural_complexity.toFixed(2)}
                        </div>
                        <div className="text-sm text-gray-500">Overall Complexity</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="visualization">
                {analysisResults.visualizations && 
                 analysisResults.visualizations.workflow_diagrams && 
                 analysisResults.visualizations.workflow_diagrams[selectedWorkflow] ? (
                  <div className="visualization-container">
                    <iframe 
                      src={`${API_URL}/visualizations/${sessionId}/${analysisResults.visualizations.workflow_diagrams[selectedWorkflow].workflow_diagram}`}
                      title="Workflow Visualization"
                    />
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <AlertCircle className="h-12 w-12 text-gray-300 mb-4" />
                    <h3 className="text-lg font-semibold mb-2">Visualization Not Available</h3>
                    <p className="text-gray-500 mb-4">
                      The visualization for this workflow is not available.
                    </p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Collection Overview</CardTitle>
          <CardDescription>
            Overview of all workflows in this session
          </CardDescription>
        </CardHeader>
        <CardContent>
          {analysisResults.visualizations && analysisResults.visualizations.collection_overview ? (
            <div className="visualization-container">
              <iframe 
                src={`${API_URL}/visualizations/${sessionId}/${analysisResults.visualizations.collection_overview}`}
                title="Collection Overview"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <AlertCircle className="h-12 w-12 text-gray-300 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Visualization Not Available</h3>
              <p className="text-gray-500 mb-4">
                The collection overview visualization is not available.
              </p>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button asChild variant="outline">
            <Link to={`/pattern-analysis/${sessionId}`}>
              View Pattern Analysis
            </Link>
          </Button>
          <Button asChild>
            <Link to={`/network-analysis/${sessionId}`}>
              View Network Analysis
            </Link>
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}

export default WorkflowAnalysis

