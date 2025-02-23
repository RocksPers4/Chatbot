"use client"

import { useState, useRef, useEffect } from "react"
import axios from "axios"

const ChatBot = ({ currentMessage, setCurrentMessage, autoSendTrigger }) => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    // Cargar mensajes del localStorage al iniciar
    const savedMessages = localStorage.getItem("chatHistory")
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages))
    }
  }, [])

  useEffect(() => {
    // Guardar mensajes en localStorage cada vez que se actualicen
    localStorage.setItem("chatHistory", JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    if (autoSendTrigger) {
      sendMessage()
    }
  }, [autoSendTrigger])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, []) // Update: Removed unnecessary dependency 'messages'

  const sendMessage = async (e) => {
    if (e) e.preventDefault()
    if (currentMessage.trim() === "") return

    const userMessage = { text: currentMessage, isUser: true, timestamp: new Date().toISOString() }
    setMessages((prevMessages) => [...prevMessages, userMessage])
    setIsLoading(true)
    setError(null)

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 segundos de timeout

      const response = await axios.post(
        "/api/chat",
        {
          user_id: "user123", // Implementar sistema de autenticación real en el futuro
          message: currentMessage,
        },
        { signal: controller.signal },
      )

      clearTimeout(timeoutId)

      const botMessage = { text: response.data.response, isUser: false, timestamp: new Date().toISOString() }
      setMessages((prevMessages) => [...prevMessages, botMessage])
    } catch (error) {
      console.error("Error al enviar mensaje:", error)
      const errorMessage = {
        text:
          error.name === "AbortError"
            ? "La respuesta está tardando más de lo esperado. Por favor, intenta de nuevo."
            : "Lo siento, ha ocurrido un error. Por favor, intenta de nuevo.",
        isUser: false,
        timestamp: new Date().toISOString(),
      }
      setMessages((prevMessages) => [...prevMessages, errorMessage])
      setError("No se pudo conectar con el servidor. Por favor, intenta de nuevo más tarde.")
    } finally {
      setIsLoading(false)
      setCurrentMessage("")
    }
  }

  return (
    <div className="bg-white shadow-md rounded-lg w-full max-w-lg mx-auto h-[80vh] flex flex-col">
      <div className="flex-grow p-4 overflow-y-auto">
        {messages.map((message, index) => (
          <div key={index} className={`mb-4 ${message.isUser ? "text-right" : "text-left"}`}>
            <span
              className={`inline-block p-2 rounded-lg ${
                message.isUser ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800"
              } max-w-[80%] break-words`}
            >
              {message.text}
            </span>
            <div className="text-xs text-gray-500 mt-1">{new Date(message.timestamp).toLocaleString()}</div>
          </div>
        ))}
        {isLoading && (
          <div className="text-center">
            <span className="inline-block animate-pulse">Escribiendo...</span>
          </div>
        )}
        {error && <div className="text-center text-red-500 mb-4">{error}</div>}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t p-4">
        <form onSubmit={sendMessage} className="flex">
          <input
            type="text"
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            className="flex-grow px-3 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Escribe tu pregunta sobre becas..."
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded-r-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          >
            Enviar
          </button>
        </form>
      </div>
    </div>
  )
}

export default ChatBot

