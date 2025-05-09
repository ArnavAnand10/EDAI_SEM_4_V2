// src/utils/toastConfig.js
import toast from 'react-hot-toast';

export const showSuccessToast = (message) => {
  toast.success(message, {
    position: 'top-center',
    style: {
      background: '#4BB543',
      color: '#fff',
      padding: '12px 20px',
      borderRadius: '8px',
    },
  });
};

export const showErrorToast = (message) => {
  toast.error(message, {
    position: 'top-center',
    style: {
      background: '#FF3333',
      color: '#fff',
      padding: '12px 20px',
      borderRadius: '8px',
    },
  });
};