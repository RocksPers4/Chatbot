import React from 'react';

const AnimatedLogo = () => {
  return (
    <div className="flex items-center">
      {/* Reemplaza 'ruta_a_tu_logo_animado' con la ruta real a tu archivo */}
      <img 
        src="./frontend/src/pochi.png" 
        alt="Logo animado" 
        className="w-10 h-10 mr-2"
      />

      <span className="text-2xl font-bold">PochiBot</span>
    </div>
  );
};

export default AnimatedLogo;