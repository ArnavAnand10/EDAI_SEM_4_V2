import { useState } from 'react';
import { toast } from 'react-hot-toast';

const SecureReportPage = () => {
  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [verificationResult, setVerificationResult] = useState(null);
  const [activeTab, setActiveTab] = useState('generate'); // 'generate' or 'verify'

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.match('image.*')) {
        toast.error('Please select an image file');
        return;
      }
      setImage(file);
      setVerificationResult(null); // Reset verification result when new image is selected
    }
  };

  const handleGenerate = async (event) => {
    event.preventDefault();
    
    if (!image) {
      toast.error("Please select an image first");
      return;
    }

    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8080/embed', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Processing failed');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = `secured_${image.name.replace(/\.[^/.]+$/, '')}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      toast.success("Secure report generated successfully!");
      
    } catch (error) {
      console.error('Error:', error);
      toast.error(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerify = async (event) => {
    event.preventDefault();
    
    if (!image) {
      toast.error("Please select an image first");
      return;
    }

    setIsLoading(true);
    setVerificationResult(null);
    
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8080/verify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Verification failed');
      }

      const result = await response.json();
      setVerificationResult(result);
      
      if (result.is_intact) {
        toast.success("Verification successful - Image is authentic!");
      } else {
        toast.error("Verification failed - Image may have been tampered with");
      }
      
    } catch (error) {
      console.error('Error:', error);
      toast.error(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100 p-4">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        <h2 className="text-2xl font-bold mb-4 text-center">Document Security</h2>
        
        {/* Tab Navigation */}
        <div className="flex mb-6 border-b">
          <button
            className={`py-2 px-4 font-medium ${activeTab === 'generate' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('generate')}
          >
            Generate Secure Report
          </button>
          <button
            className={`py-2 px-4 font-medium ${activeTab === 'verify' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('verify')}
          >
            Verify Scan
          </button>
        </div>
        
        {/* File Input */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Upload Document Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
          {image && (
            <p className="mt-1 text-sm text-gray-600">
              Selected: {image.name} ({Math.round(image.size / 1024)} KB)
            </p>
          )}
        </div>
        
        {/* Action Button */}
        {activeTab === 'generate' ? (
          <button
            type="button"
            onClick={handleGenerate}
            disabled={isLoading || !image}
            className={`w-full py-2 px-4 rounded-md text-white font-medium
              ${isLoading || !image
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
              } transition-colors`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </span>
            ) : 'Generate Secure Report'}
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={handleVerify}
              disabled={isLoading || !image}
              className={`w-full py-2 px-4 rounded-md text-white font-medium
                ${isLoading || !image
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
                } transition-colors mb-4`}
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Verifying...
                </span>
              ) : 'Verify Document'}
            </button>
            
            {verificationResult && (
              <div className="mt-4 p-4 bg-gray-50 rounded-md">
                <h3 className="font-medium mb-2">Verification Results:</h3>
                <div className={`p-3 rounded ${verificationResult.is_intact ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                  <p>{verificationResult.message}</p>
                  {verificationResult.original_metadata && (
                    <div className="mt-2 text-sm">
                      <p>Original Metadata:</p>
                      <pre className="bg-white p-2 rounded mt-1 overflow-x-auto">
                        {JSON.stringify(verificationResult.original_metadata, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default SecureReportPage;