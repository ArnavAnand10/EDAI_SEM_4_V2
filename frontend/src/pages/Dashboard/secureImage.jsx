// src/pages/SecureReportPage/index.jsx
import { useState } from 'react';
import { toast } from 'react-hot-toast';

const SecureReportPage = () => {
  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!image) {
      toast.error("Please select an image to upload");
      return;
    }
    
    // Show loading state
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('image', image);
    
    try {
      // Use the correct API URL - replace with your actual server URL
      // If running locally on the same port, use relative path
      const response = await fetch('http://0.0.0.0/embed', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        // Get metadata from headers if present
        const metadataHeader = response.headers.get('X-Metadata');
        
        // Get and process the image
        const blob = await response.blob();
        const downloadUrl = URL.createObjectURL(blob);
        
        toast.success("Report generated successfully!");
        
        // Create a download link and trigger it
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = 'secured-report.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        // Show metadata if available
        if (metadataHeader) {
          toast.success(`Report metadata: ${metadataHeader}`);
        }
      } else {
        // Try to parse the error response
        const errorData = await response.text();
        toast.error(`Failed to generate report: ${errorData}`);
      }
    } catch (error) {
      toast.error("An error occurred while connecting to the server");
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
        <h2 className="text-2xl font-bold mb-6">Generate Tamper-Proof Report</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">Upload Report Image</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="mt-2 p-2 border border-gray-300 rounded w-full"
            />
            {image && (
              <div className="mt-2">
                <p className="text-sm text-gray-600">Selected: {image.name}</p>
              </div>
            )}
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-2 rounded-lg ${
              isLoading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-500 hover:bg-blue-600 text-white'
            }`}
          >
            {isLoading ? 'Generating...' : 'Generate Report'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default SecureReportPage;