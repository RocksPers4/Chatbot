import React from 'react';

const HelpModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-lg max-h-[80vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Ayuda</h2>
        <p className="mb-4">
          Este chatbot está diseñado para ayudarte con información sobre becas y ayudas económicas en la ESPOCH.
          Puedes hacer preguntas sobre tipos de becas, requisitos, fechas de aplicación, y más.
        </p>
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

export default HelpModal;

