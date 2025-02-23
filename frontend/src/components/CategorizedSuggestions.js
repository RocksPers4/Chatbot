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
    <div className="p-4 sm:p-6">
      <h2 className="text-xl sm:text-2xl font-bold mb-4">Sugerencias de preguntas</h2>
      <div className="flex flex-wrap mb-4 gap-2">
        {Object.entries(categories).map(([key, value]) => (
          <button
            key={key}
            onClick={() => setActiveCategory(key)}
            className={`px-3 py-1 sm:px-4 sm:py-2 rounded-md text-sm sm:text-base transition-colors duration-200 ${
              activeCategory === key
                ? "bg-blue-500 text-white"
                : "bg-gray-200 text-gray-800 hover:bg-gray-300"
            }`}
          >
            {value}
          </button>
        ))}
      </div>
      <ul className="space-y-2 mb-4 max-h-[50vh] overflow-y-auto">
        {sugerencias[activeCategory].map((suggestion, index) => (
          <li
            key={index}
            className="cursor-pointer hover:bg-gray-100 p-2 rounded transition-colors duration-200 text-sm sm:text-base"
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
        className="w-full bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors duration-200"
      >
        Cerrar
      </button>
    </div>
  )
}

export default CategorizedSuggestions