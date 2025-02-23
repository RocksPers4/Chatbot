import React from 'react';

const HistoryModal = ({ isOpen, onClose, history }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-lg max-h-[80vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Historial de Chat</h2>
        <ul className="space-y-2">
          {history.map((item, index) => (
            <li key={index} className={`p-2 rounded ${item.isUser ? 'bg-blue-100' : 'bg-gray-100'}`}>
              {item.text}
            </li>
          ))}
        </ul>
        <button
          onClick={onClose}
          className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors duration-200"
        >
          Cerrar
        </button>
      </div>
    </div>
  )
}

export default HistoryModal;

