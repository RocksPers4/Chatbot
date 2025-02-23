import { useState, useEffect } from "react"
import ChatBot from "./components/ChatBot"
import Header from "./components/Header"
import HistoryModal from "./components/HistoryModal"
import HelpModal from "./components/HelpModal"
import SuggestionsModal from "./components/SuggestionsModal"

const App = () => {
  const [showHistory, setShowHistory] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [chatHistory, setChatHistory] = useState([])
  const [currentMessage, setCurrentMessage] = useState("")
  const [autoSendTrigger, setAutoSendTrigger] = useState(0)

  useEffect(() => {
    const savedHistory = localStorage.getItem("chatHistory")
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory))
    }
  }, [])

  const updateHistory = () => {
    const savedHistory = localStorage.getItem("chatHistory")
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory))
    }
  }

  const handleSuggestionClick = (suggestion) => {
    setCurrentMessage(suggestion)
    setAutoSendTrigger((prev) => prev + 1)
    setShowSuggestions(false)
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <Header
        onHistoryClick={() => {
          updateHistory()
          setShowHistory(true)
        }}
        onHelpClick={() => setShowHelp(true)}
        onSuggestionsClick={() => setShowSuggestions(true)}
      />
      <main className="flex-grow flex items-center justify-center p-4">
        <ChatBot
          currentMessage={currentMessage}
          setCurrentMessage={setCurrentMessage}
          autoSendTrigger={autoSendTrigger}
        />
      </main>
      <HistoryModal isOpen={showHistory} onClose={() => setShowHistory(false)} history={chatHistory} />
      <HelpModal isOpen={showHelp} onClose={() => setShowHelp(false)} />
      <SuggestionsModal
        isOpen={showSuggestions}
        onClose={() => setShowSuggestions(false)}
        onSuggestionClick={handleSuggestionClick}
      />
    </div>
  )
}

export default App