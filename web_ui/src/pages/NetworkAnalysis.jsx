import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { 
  Network, 
  ArrowLeft, 
  Loader2,
  AlertCircle,
  ArrowRight
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

const NetworkAnalysis = () => {
  const { sessionId } = useParams()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  
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
          setLoading(false)
          return
        }
        
        // If no results exist, trigger analysis
        const analysisResponse = await fetch(`${API_URL}/analyze/${sessionId}`)
        
        if (!analysisResponse.ok) {
          const errorData = await analysisResponse.json()
          throw new Error(errorData.error || 'Analysis failed')
        }
        
        const data = await analysisResponse.json()
        setAnalysisResults(data)
        
      } catch (error) {
        console.error('Error:', error)
        setError(error.message || 'Failed to analyze workflows')
      } finally {
        setLoading(false)
      }
    }
    
    fetchResults()
  }, [sessionId])
  
  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Network Analysis</h1>
          <p className="page-description">
            Loading network analysis results...
          </p>
        </div>
        
        <div className="loading-spinner">
          <Loader2 className="h-12 w-12 text-primary" />
        </div>
      </div>
    )
  }
  
  if (error) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Network Analysis</h1>
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
  
  if (!analysisResults || !analysisResults.network_analysis) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Network Analysis</h1>
          <p className="page-description">
            No network analysis results found
          </p>
        </div>
        
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-16 w-16 text-gray-300 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Network Analysis Results</h3>
            <p className="text-gray-500 text-center mb-6">
              No network analysis results were found for this session.
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
  
  const { network_analysis, visualizations } = analysisResults
  
  return (
    <div>
      <div className="page-header">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="page-title">Network Analysis</h1>
            <p className="page-description">
              Network analysis results for session {sessionId.substring(0, 8)}
            </p>
          </div>
          <div className="flex space-x-2">
            <Button asChild variant="outline">
              <Link to={`/pattern-analysis/${sessionId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Pattern Analysis
              </Link>
            </Button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Network Metrics</CardTitle>
            <CardDescription>
              Key metrics of the workflow network
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-2xl font-bold text-primary">
                    {network_analysis.node_count}
                  </div>
                  <div className="text-sm text-gray-500">Nodes</div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="text-2xl font-bold text-primary">
                    {network_analysis.edge_count}
                  </div>
                  <div className="text-sm text-gray-500">Edges</div>
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {network_analysis.density.toFixed(3)}
                </div>
                <div className="text-sm text-gray-500">Network Density</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {network_analysis.average_degree.toFixed(2)}
                </div>
                <div className="text-sm text-gray-500">Average Degree</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {network_analysis.diameter || 'N/A'}
                </div>
                <div className="text-sm text-gray-500">Network Diameter</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {network_analysis.average_path_length ? 
                    network_analysis.average_path_length.toFixed(2) : 'N/A'}
                </div>
                <div className="text-sm text-gray-500">Avg. Path Length</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {network_analysis.clustering_coefficient ? 
                    network_analysis.clustering_coefficient.toFixed(3) : 'N/A'}
                </div>
                <div className="text-sm text-gray-500">Clustering Coefficient</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Network Visualization</CardTitle>
            <CardDescription>
              Visual representation of the workflow network
            </CardDescription>
          </CardHeader>
          <CardContent>
            {visualizations && visualizations.network_analysis ? (
              <div className="visualization-container">
                <iframe 
                  src={`${API_URL}/visualizations/${sessionId}/${visualizations.network_analysis}`}
                  title="Network Analysis"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <AlertCircle className="h-12 w-12 text-gray-300 mb-4" />
                <h3 className="text-lg font-semibold mb-2">Visualization Not Available</h3>
                <p className="text-gray-500 mb-4">
                  The network visualization is not available.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Network Interpretation</CardTitle>
          <CardDescription>
            What the network metrics tell us about these workflows
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-medium mb-2">Density Analysis</h3>
              <p className="text-sm text-gray-600">
                The network density of {network_analysis.density.toFixed(3)} indicates 
                {network_analysis.density < 0.3 ? ' a sparse network with relatively few connections between nodes. This suggests these workflows have a linear or tree-like structure with limited branching.' : 
                 network_analysis.density < 0.6 ? ' a moderately connected network. These workflows have a balanced structure with some branching and parallel paths.' : 
                 ' a densely connected network. These workflows have many interconnections, suggesting complex logic with many decision points and parallel paths.'}
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium mb-2">Connectivity Analysis</h3>
              <p className="text-sm text-gray-600">
                With an average degree of {network_analysis.average_degree.toFixed(2)}, each node connects to approximately {network_analysis.average_degree.toFixed(1)} other nodes on average. 
                {network_analysis.average_degree < 2 ? ' This low connectivity suggests simple, linear workflows with minimal branching.' : 
                 network_analysis.average_degree < 4 ? ' This moderate connectivity indicates workflows with some branching and decision points.' : 
                 ' This high connectivity suggests complex workflows with significant branching and parallel processing.'}
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium mb-2">Path Analysis</h3>
              <p className="text-sm text-gray-600">
                {network_analysis.diameter ? 
                  `The network diameter of ${network_analysis.diameter} represents the longest shortest path between any two nodes. ` : ''}
                {network_analysis.average_path_length ? 
                  `The average path length of ${network_analysis.average_path_length.toFixed(2)} indicates ${
                    network_analysis.average_path_length < 3 ? 'short paths between nodes, suggesting efficient workflows with minimal steps between start and end points.' : 
                    network_analysis.average_path_length < 5 ? 'moderate path lengths, indicating balanced workflows with reasonable process flows.' : 
                    'long paths between nodes, suggesting complex workflows with many sequential steps.'
                  }` : ''}
              </p>
            </div>
            
            <div>
              <h3 className="text-base font-medium mb-2">Clustering Analysis</h3>
              <p className="text-sm text-gray-600">
                {network_analysis.clustering_coefficient ? 
                  `The clustering coefficient of ${network_analysis.clustering_coefficient.toFixed(3)} indicates ${
                    network_analysis.clustering_coefficient < 0.3 ? 'low clustering, suggesting workflows with few interconnected groups of nodes.' : 
                    network_analysis.clustering_coefficient < 0.6 ? 'moderate clustering, indicating some functional grouping of nodes in the workflows.' : 
                    'high clustering, suggesting workflows with well-defined functional modules or components.'
                  }` : ''}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <div className="mt-6 flex justify-between">
        <Button asChild variant="outline">
          <Link to={`/workflow-analysis/${sessionId}`}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Workflow Analysis
          </Link>
        </Button>
        
        <Button asChild>
          <Link to="/sessions">
            View All Sessions
          </Link>
        </Button>
      </div>
    </div>
  )
}

export default NetworkAnalysis

