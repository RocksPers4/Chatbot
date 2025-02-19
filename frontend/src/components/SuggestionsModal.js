import CategorizedSuggestions from "./CategorizedSuggestions"

const SuggestionsModal = ({ isOpen, onClose, onSuggestionClick }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <CategorizedSuggestions onSuggestionClick={onSuggestionClick} onClose={onClose} />
    </div>
  )
}

export default SuggestionsModal

