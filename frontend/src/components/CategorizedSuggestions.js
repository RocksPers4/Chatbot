import { useState } from "react"
import sugerencias from "./sugerencias"

const CategorizedSuggestions = ({ onSuggestionClick, onClose }) => {
  const [activeCategory, setActiveCategory] = useState("general")

  const categories = {
    general: "General",
    becas: "Becas",
    ayudasEconomicas: "Ayudas Económicas",
  }

  return (
    <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
      <h2 className="text-2xl font-bold mb-4">Sugerencias de preguntas</h2>
      <div className="flex mb-4 space-x-2">
        {Object.entries(categories).map(([key, value]) => (
          <button
            key={key}
            onClick={() => setActiveCategory(key)}
            className={`px-4 py-2 rounded-md transition-colors duration-200 ${
              activeCategory === key ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800 hover:bg-gray-300"
            }`}
          >
            {value}
          </button>
        ))}
      </div>
      <ul className="space-y-2">
        {sugerencias[activeCategory].map((suggestion, index) => (
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
        className="mt-4 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors duration-200"
      >
        Cerrar
      </button>
    </div>
  )
}

export default CategorizedSuggestions

