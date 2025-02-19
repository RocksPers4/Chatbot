import React from 'react';
import AnimatedLogo from './AnimatedLogo';

const Header = ({ onHistoryClick, onHelpClick, onSuggestionsClick }) => {
  return (
    <header className="bg-blue-600 text-white p-4 shadow-md">
      <div className="container mx-auto flex justify-between items-center">
        <AnimatedLogo />
        <nav>
          <button
            onClick={onSuggestionsClick}
            className="bg-yellow-500 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded mr-2"
          >
            Sugerencias
          </button>
          <button 
            onClick={onHistoryClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mr-2"
          >
            Historial
          </button>
          <button 
            onClick={onHelpClick}
            className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
          >
            Ayuda
          </button>
        </nav>
      </div>
    </header>
  );
};

export default Header;
