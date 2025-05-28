const HelpModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg p-6 max-w-lg w-full">
        <h2 className="text-2xl font-bold mb-4">Ayuda para Becas y Ayudas Económicas</h2>
        <p>Para más información: <a href='https://www.espoch.edu.ec/becas-prueba/' className="text-blue-500 hover:underline">Becas y Ayudas Economicas</a></p>
        {/* https://www.espoch.edu.ec/becas-prueba/ */}
        <div>
        <a h2 href="https://becas.espoch.edu.ec/becario" className="text-blue-500 hover:underline">Llenar mi Ficha</a>
        </div>
        <p>¿Por qué debo actualizar mi ficha?: <a href='https://www.facebook.com/share/v/15QhCzPBHn/' className="text-blue-500 hover:underline">Click aquí</a></p>
        <div>
        <a h2 href="https://forms.office.com/r/FrghAuRVbC" className="text-blue-500 hover:underline">Cuestionario</a>
        </div>
        <button
          onClick={onClose}
          className="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Cerrar
        </button>
      </div>
    </div>
  );
};

export default HelpModal;

