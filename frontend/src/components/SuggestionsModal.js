import CategorizedSuggestions from "./CategorizedSuggestions"

const SuggestionsModal = ({ isOpen, onClose, onSuggestionClick }) => {
  if (!isOpen) return null

  const suggestions = [
    "¿Qué tipos de becas ofrece la ESPOCH?",
    "¿Cuáles son los requisitos para una beca de excelencia académica?",
    "¿Cuándo son las fechas de aplicación para becas?",
    "¿Cómo puedo aplicar para una ayuda económica?",
    "¿Qué documentos necesito para solicitar una beca?"
  ]

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-lg max-h-[80vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Sugerencias de preguntas</h2>
        <ul className="space-y-2">
          {suggestions.map((suggestion, index) => (
            <li
              key={index}
              className="cursor-pointer hover:bg-gray-100 p-2 rounded transition-colors duration-200"
              onClick={() => {
                onSuggestionClick(suggestion)
                onClose()
              }}
            >
              {suggestion}
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

export default SuggestionsModal

