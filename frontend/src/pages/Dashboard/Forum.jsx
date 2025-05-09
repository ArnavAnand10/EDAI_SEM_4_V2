import { useState, useEffect } from 'react';
import Sidebar from '../../components/Sidebar';
import { toast } from 'react-hot-toast';

const Forum = () => {
  // State for chat functionality
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [username, setUsername] = useState('');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [roomId, setRoomId] = useState('general');
  const [loading, setLoading] = useState(false);
  
  // Replace with your actual Rocket Chat server URL
  const serverUrl = 'https://doctorforum.rocket.chat';
  
  // Simple login function
  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username.trim()) {
      toast.error('Please enter a username');
      return;
    }
    
    setLoading(true);
    try {
      // In a real app, you would make an actual API call to authenticate
      // For this example, we're just simulating a login
      setTimeout(() => {
        setIsLoggedIn(true);
        toast.success(`Welcome, ${username}!`);
        fetchMessages();
        setLoading(false);
      }, 1000);
    } catch (error) {
      toast.error('Failed to login');
      setLoading(false);
    }
  };
  
  // Fetch messages from the room
  const fetchMessages = async () => {
    try {
      // This is where you would make an actual API call to get messages
      // For this example, we're just setting mock data
      const mockMessages = [
        { id: 1, user: 'John', text: 'Hello everyone!', timestamp: new Date().toISOString() },
        { id: 2, user: 'Sarah', text: 'Hi John, how are you?', timestamp: new Date().toISOString() },
        { id: 3, user: 'Mike', text: 'Welcome to our chat room', timestamp: new Date().toISOString() }
      ];
      setMessages(mockMessages);
    } catch (error) {
      toast.error('Failed to fetch messages');
    }
  };
  
  // Send a new message
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim()) return;
    
    try {
      // In a real app, you would send this to the Rocket Chat API
      const newMsg = {
        id: messages.length + 1,
        user: username,
        text: newMessage,
        timestamp: new Date().toISOString()
      };
      
      setMessages([...messages, newMsg]);
      setNewMessage('');
      toast.success('Message sent');
    } catch (error) {
      toast.error('Failed to send message');
    }
  };
  
  // Change room
  const changeRoom = (newRoomId) => {
    setRoomId(newRoomId);
    setMessages([]);
    // In a real app, you would fetch messages for the new room
    setTimeout(fetchMessages, 500);
    toast.success(`Switched to ${newRoomId} room`);
  };
  
  // Format timestamp
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Login form
  const renderLoginForm = () => (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-md w-96">
        <h2 className="text-2xl font-bold mb-6 text-center">Rocket Chat Login</h2>
        <form onSubmit={handleLogin}>
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              placeholder="Enter your username"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
  
  // Chat interface
  const renderChatInterface = () => (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 text-white p-4">
        <h2 className="text-xl font-bold mb-4">Rooms</h2>
        <ul>
          {['general', 'random', 'support'].map((room) => (
            <li 
              key={room} 
              className={`p-2 cursor-pointer hover:bg-gray-700 rounded ${roomId === room ? 'bg-gray-700' : ''}`}
              onClick={() => changeRoom(room)}
            >
              # {room}
            </li>
          ))}
        </ul>
        <div className="mt-auto pt-4">
          <p className="text-sm">Logged in as:</p>
          <p className="font-bold">{username}</p>
          <button 
            onClick={() => setIsLoggedIn(false)} 
            className="mt-2 bg-red-500 text-white px-3 py-1 rounded text-sm"
          >
            Logout
          </button>
        </div>
      </div>
      
      {/* Chat area */}
      <div className="flex flex-col flex-1">
        {/* Header */}
        <div className="bg-white p-4 shadow-sm">
          <h2 className="text-xl font-bold">#{roomId}</h2>
        </div>
        
        {/* Messages */}
        <div className="flex-1 p-4 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="text-center text-gray-500">No messages yet</p>
          ) : (
            messages.map((msg) => (
              <div 
                key={msg.id} 
                className={`mb-4 ${msg.user === username ? 'text-right' : ''}`}
              >
                <div className={`inline-block p-3 rounded-lg ${
                  msg.user === username ? 'bg-blue-500 text-white' : 'bg-gray-200'
                }`}>
                  {msg.user !== username && (
                    <p className="font-bold text-sm">{msg.user}</p>
                  )}
                  <p>{msg.text}</p>
                  <p className="text-xs mt-1 opacity-75">{formatTime(msg.timestamp)}</p>
                </div>
              </div>
            ))
          )}
        </div>
        
        {/* Message input */}
        <form onSubmit={sendMessage} className="bg-white p-4 border-t">
          <div className="flex">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              className="flex-1 border rounded-l py-2 px-3 focus:outline-none"
              placeholder={`Message #${roomId}`}
            />
            <button
              type="submit"
              className="bg-blue-500 text-white px-4 rounded-r hover:bg-blue-600"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
  
  return isLoggedIn ? renderChatInterface() : renderLoginForm();
};

export default Forum;