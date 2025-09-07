import React, { useState, useRef, useEffect } from "react";
import { marked } from "marked";
import "./App.css";

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const BOT_AVATAR = "https://img.icons8.com/color/48/000000/chatbot.png";
const USER_AVATAR = "https://img.icons8.com/fluency/48/000000/user-male-circle.png";

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span />
      <span />
      <span />
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`chat-message ${msg.sender}`}>
      <div className="avatar">
        <img
          src={msg.sender === "user" ? USER_AVATAR : BOT_AVATAR}
          alt={msg.sender === "user" ? "You" : "Bot"}
        />
      </div>
      <div className="bubble">
        <span
          dangerouslySetInnerHTML={{
            __html: marked.parseInline(msg.text)
          }}
        />
        <div className="timestamp">{msg.time}</div>
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "👋 **Hello!** How can I help you today?",
      time: formatTime(new Date())
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState("dark");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    document.body.className = theme === "dark" ? "dark-theme" : "light-theme";
  }, [theme]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const now = new Date();
    const userMessage = {
      sender: "user",
      text: input,
      time: formatTime(now)
    };
    setMessages((msgs) => [...msgs, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input })
      });
      const data = await res.json();
      setMessages((msgs) => [
        ...msgs,
        {
          sender: "bot",
          text: data.response,
          time: formatTime(new Date())
        }
      ]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        {
          sender: "bot",
          text: "❌ Sorry, I couldn't reach the server.",
          time: formatTime(new Date())
        }
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="gpt-outer">
      <div className="gpt-header">
        <img src={BOT_AVATAR} alt="Bot" />
        <span>Smart Chatbot</span>
        <button
          className="theme-toggle"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Toggle dark/light mode"
        >
          {theme === "dark" ? "🌞" : "🌙"}
        </button>
      </div>
      <div className="gpt-container">
        <div className="gpt-box">
          {messages.map((msg, idx) => (
            <Message msg={msg} key={idx} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>
        <form className="gpt-input" onSubmit={sendMessage}>
          <textarea
            rows={1}
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            disabled={loading}
            autoFocus
            style={{ minHeight: 44, maxHeight: 120, overflow: "auto" }}
            onInput={e => {
              e.target.style.height = "44px";
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey && !loading) {
                e.preventDefault();
                sendMessage(e);
              }
            }}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? <TypingIndicator /> : "Send"}
          </button>
        </form>
      </div>
      <div className="gpt-footer">
        <span>
          Powered by <b>SmartChatbot</b> &middot; {new Date().getFullYear()}
        </span>
      </div>
      <div className="system-message">
        <span>SmartChatbot can answer questions, help brainstorm, and more. Your conversation is private.</span>
      </div>
      {messages.length > 1 && !loading && (
        <button
          className="regenerate-btn"
          onClick={() => {
            // Find the last user message
            const lastUserMsg = [...messages].reverse().find(m => m.sender === "user");
            if (lastUserMsg) setInput(lastUserMsg.text);
          }}
          type="button"
        >
          🔄 Regenerate
        </button>
      )}
    </div>
  );
}

export default App;