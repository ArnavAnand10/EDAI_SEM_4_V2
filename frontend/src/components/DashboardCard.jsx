// src/components/DashboardCard.jsx
import { Link } from 'react-router-dom';

const DashboardCard = ({ title, description, icon, link }) => {
  return (
    <Link
      to={link}
      className="block bg-white rounded-xl shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300"
    >
      <div className="p-6">
        <div className="text-4xl mb-4">{icon}</div>
        <h3 className="text-xl font-semibold text-gray-800 mb-2">{title}</h3>
        <p className="text-gray-600">{description}</p>
      </div>
    </Link>
  );
};

export default DashboardCard;