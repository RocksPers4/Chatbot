import React from 'react';

const HistoryModal = ({ isOpen, onClose, history }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg p-6 max-w-lg w-full max-h-[80vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Historial de Chat</h2>
        <div className="space-y-4">
          {history.map((message, index) => (
            <div key={index} className={`${message.isUser ? 'text-right' : 'text-left'}`}>
              <span className={`inline-block p-2 rounded-lg ${
                message.isUser ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
              }`}>
                {message.text}
              </span>
              <div className="text-xs text-gray-500 mt-1">
                {new Date(message.timestamp).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
        <button
          onClick={onClose}
          className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Cerrar
        </button>
      </div>
    </div>
  );
};

export default HistoryModal;

