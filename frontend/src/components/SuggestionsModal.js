import CategorizedSuggestions from "./CategorizedSuggestions"

const SuggestionsModal = ({ isOpen, onClose, onSuggestionClick }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50">
      <div className="bg-white rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <CategorizedSuggestions onSuggestionClick={onSuggestionClick} onClose={onClose} />
      </div>
    </div>
  )
}

export default SuggestionsModal