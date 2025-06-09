import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { 
  Layers, 
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

const API_URL = 'http://localhost:5000/api'

const PatternAnalysis = () => {
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
          <h1 className="page-title">Pattern Analysis</h1>
          <p className="page-description">
            Loading pattern analysis results...
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
          <h1 className="page-title">Pattern Analysis</h1>
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
  
  if (!analysisResults || !analysisResults.pattern_analysis) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Pattern Analysis</h1>
          <p className="page-description">
            No pattern analysis results found
          </p>
        </div>
        
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-16 w-16 text-gray-300 mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Pattern Analysis Results</h3>
            <p className="text-gray-500 text-center mb-6">
              No pattern analysis results were found for this session.
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
  
  const { pattern_analysis, visualizations } = analysisResults
  
  return (
    <div>
      <div className="page-header">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="page-title">Pattern Analysis</h1>
            <p className="page-description">
              Pattern mining results for session {sessionId.substring(0, 8)}
            </p>
          </div>
          <div className="flex space-x-2">
            <Button asChild variant="outline">
              <Link to={`/workflow-analysis/${sessionId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Workflow Analysis
              </Link>
            </Button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle>Pattern Statistics</CardTitle>
            <CardDescription>
              Overview of discovered patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {pattern_analysis.pattern_count}
                </div>
                <div className="text-sm text-gray-500">Total Patterns</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {pattern_analysis.rule_count}
                </div>
                <div className="text-sm text-gray-500">Association Rules</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {pattern_analysis.top_patterns.length > 0 
                    ? pattern_analysis.top_patterns[0].support.toFixed(2) 
                    : '0.00'}
                </div>
                <div className="text-sm text-gray-500">Highest Support</div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="text-2xl font-bold text-primary">
                  {pattern_analysis.top_rules.length > 0 
                    ? pattern_analysis.top_rules[0].lift.toFixed(2) 
                    : '0.00'}
                </div>
                <div className="text-sm text-gray-500">Highest Lift</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Pattern Visualization</CardTitle>
            <CardDescription>
              Visual representation of discovered patterns
            </CardDescription>
          </CardHeader>
          <CardContent>
            {visualizations && visualizations.pattern_analysis ? (
              <div className="visualization-container">
                <iframe 
                  src={`${API_URL}/visualizations/${sessionId}/${visualizations.pattern_analysis}`}
                  title="Pattern Analysis"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <AlertCircle className="h-12 w-12 text-gray-300 mb-4" />
                <h3 className="text-lg font-semibold mb-2">Visualization Not Available</h3>
                <p className="text-gray-500 mb-4">
                  The pattern visualization is not available.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      
      <Tabs defaultValue="patterns">
        <TabsList className="mb-4">
          <TabsTrigger value="patterns">Frequent Patterns</TabsTrigger>
          <TabsTrigger value="rules">Association Rules</TabsTrigger>
        </TabsList>
        
        <TabsContent value="patterns">
          <Card>
            <CardHeader>
              <CardTitle>Top Frequent Patterns</CardTitle>
              <CardDescription>
                Most common patterns discovered in the workflows
              </CardDescription>
            </CardHeader>
            <CardContent>
              {pattern_analysis.top_patterns.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Pattern Items</TableHead>
                      <TableHead>Support</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pattern_analysis.top_patterns.map((pattern, index) => (
                      <TableRow key={index}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {pattern.items.map((item, i) => (
                              <span 
                                key={i} 
                                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary"
                              >
                                {item}
                              </span>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>{pattern.support.toFixed(3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <p>No patterns found.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="rules">
          <Card>
            <CardHeader>
              <CardTitle>Top Association Rules</CardTitle>
              <CardDescription>
                Strongest relationships between workflow elements
              </CardDescription>
            </CardHeader>
            <CardContent>
              {pattern_analysis.top_rules.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>#</TableHead>
                      <TableHead>Antecedent</TableHead>
                      <TableHead>Consequent</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Lift</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pattern_analysis.top_rules.map((rule, index) => (
                      <TableRow key={index}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {rule.antecedent.map((item, i) => (
                              <span 
                                key={i} 
                                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                              >
                                {item}
                              </span>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {rule.consequent.map((item, i) => (
                              <span 
                                key={i} 
                                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800"
                              >
                                {item}
                              </span>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>{rule.confidence.toFixed(3)}</TableCell>
                        <TableCell>{rule.lift.toFixed(3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <p>No association rules found.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      <div className="mt-6 flex justify-end">
        <Button asChild>
          <Link to={`/network-analysis/${sessionId}`}>
            View Network Analysis
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  )
}

export default PatternAnalysis

