import AnimatedLogo from './AnimatedLogo';

const Header = ({ onHistoryClick, onHelpClick, onSuggestionsClick }) => {
  return (
    <header className="bg-blue-600 text-white p-4">
      <div className="container mx-auto flex flex-col sm:flex-row justify-between items-center">
        <div className="flex items-center mb-4 sm:mb-0">
          <AnimatedLogo className="w-12 h-12 mr-4" />
        </div>
        <nav className="flex flex-wrap justify-center sm:justify-end space-y-2 sm:space-y-0 sm:space-x-2">
          <button
            onClick={onSuggestionsClick}
            className="bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded w-full sm:w-auto"
          >
            Sugerencias
          </button>
          <button
            onClick={onHistoryClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full sm:w-auto"
          >
            Historial
          </button>
          <button
            onClick={onHelpClick}
            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded w-full sm:w-auto"
          >
            Ayuda
          </button>
        </nav>
      </div>
    </header>
  )
}

export default Header