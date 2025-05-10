// src/pages/SecureReportPage/index.jsx
import { useState } from 'react';
import { toast } from 'react-hot-toast';

const SecureReportPage = () => {
  const [image, setImage] = useState(null);

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

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('/embed', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const metadataUUID = response.headers.get('X-Metadata-UUID');
        const metadataTimestamp = response.headers.get('X-Metadata-Timestamp');
        const blob = await response.blob();
        const downloadUrl = URL.createObjectURL(blob);
        
        toast.success("Report generated successfully!");
        window.location.href = downloadUrl; // Automatically download the report

        // Optionally show metadata details
        alert(`Metadata UUID: ${metadataUUID}\nTimestamp: ${metadataTimestamp}`);
      } else {
        toast.error("Failed to generate the report.");
      }
    } catch (error) {
      toast.error("An error occurred while generating the report.");
      console.error(error);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-lg w-1/2">
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
          </div>
          <button
            type="submit"
            className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600"
          >
            Generate Report
          </button>
        </form>
      </div>
    </div>
  );
};

export default SecureReportPage;
