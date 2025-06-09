import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, X, FileJson, Check, Loader2 } from 'lucide-react'

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

const API_URL = 'http://localhost:5000/api'

const UploadWorkflows = () => {
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  
  const fileInputRef = useRef(null)
  const navigate = useNavigate()
  
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }
  
  const handleDragLeave = () => {
    setIsDragging(false)
  }
  
  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      file => file.name.endsWith('.json')
    )
    
    if (droppedFiles.length === 0) {
      setUploadError('Please drop valid JSON files')
      return
    }
    
    setFiles(prevFiles => [...prevFiles, ...droppedFiles])
    setUploadError(null)
  }
  
  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files).filter(
      file => file.name.endsWith('.json')
    )
    
    if (selectedFiles.length === 0) {
      setUploadError('Please select valid JSON files')
      return
    }
    
    setFiles(prevFiles => [...prevFiles, ...selectedFiles])
    setUploadError(null)
  }
  
  const handleRemoveFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index))
  }
  
  const handleUpload = async () => {
    if (files.length === 0) {
      setUploadError('Please select at least one file to upload')
      return
    }
    
    setUploading(true)
    setUploadError(null)
    
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    
    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Upload failed')
      }
      
      const data = await response.json()
      setSessionId(data.session_id)
      setUploadSuccess(true)
      setFiles([])
    } catch (error) {
      console.error('Upload error:', error)
      setUploadError(error.message || 'Failed to upload files')
    } finally {
      setUploading(false)
    }
  }
  
  const handleAnalyzeNow = () => {
    navigate(`/workflow-analysis/${sessionId}`)
  }
  
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Upload Workflows</h1>
        <p className="page-description">
          Upload n8n workflow JSON files for analysis
        </p>
      </div>
      
      {uploadSuccess ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-green-600 flex items-center">
              <Check className="mr-2" /> Upload Successful
            </CardTitle>
            <CardDescription>
              Your workflows have been uploaded successfully
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              Session ID: <span className="font-mono">{sessionId}</span>
            </p>
            <p>
              You can now analyze these workflows to gain insights into their structure,
              patterns, and network properties.
            </p>
          </CardContent>
          <CardFooter className="flex justify-between">
            <Button variant="outline" onClick={() => {
              setUploadSuccess(false)
              setSessionId(null)
            }}>
              Upload More
            </Button>
            <Button onClick={handleAnalyzeNow}>
              Analyze Now
            </Button>
          </CardFooter>
        </Card>
      ) : (
        <>
          {uploadError && (
            <Alert variant="destructive" className="mb-6">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{uploadError}</AlertDescription>
            </Alert>
          )}
          
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Upload n8n Workflow Files</CardTitle>
              <CardDescription>
                Drag and drop your workflow JSON files or click to browse
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={`upload-container ${isDragging ? 'border-primary bg-primary/10' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  multiple
                  accept=".json"
                  className="hidden"
                />
                <Upload size={48} className="upload-icon mx-auto" />
                <p className="text-lg font-medium mb-2">Drop your files here</p>
                <p className="text-sm text-gray-500">
                  or click to browse (only .json files)
                </p>
              </div>
            </CardContent>
          </Card>
          
          {files.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Selected Files</CardTitle>
                <CardDescription>
                  {files.length} file{files.length !== 1 ? 's' : ''} selected
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {files.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-md"
                    >
                      <div className="flex items-center">
                        <FileJson className="h-5 w-5 text-primary mr-2" />
                        <span className="text-sm font-medium">{file.name}</span>
                        <span className="text-xs text-gray-500 ml-2">
                          ({(file.size / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleRemoveFile(index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
              <CardFooter>
                <Button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="w-full"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    'Upload Files'
                  )}
                </Button>
              </CardFooter>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

export default UploadWorkflows

