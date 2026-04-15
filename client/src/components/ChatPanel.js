"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Bot, User, FileText, ChevronDown, ChevronRight, Loader2, PlusCircle, MessageSquare, Menu, X, UserCircle } from "lucide-react";

export default function ChatPanel({ session }) {
  const defaultMessage = {
    role: "ai",
    content: "Hello! I am your Research Assistant. Ask me anything about your uploaded documents.",
    sources: []
  };

  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([defaultMessage]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  // UI State
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/chats/sessions", {
        headers: { "x-user-email": session?.user?.email || "" }
      });
      if (res.ok) {
        const data = await res.json();
        setChatSessions(data);
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  }, [session?.user?.email]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const loadSession = async (sessionId) => {
    setCurrentSessionId(sessionId);
    setMessages([defaultMessage]);
    setIsLoading(true);
    setIsSidebarOpen(false); // Close sidebar automatically on mobile
    try {
      const res = await fetch(`/api/chats/sessions/${sessionId}`, {
        headers: { "x-user-email": session?.user?.email || "" }
      });
      if (res.ok) {
        const data = await res.json();
        if (data.length > 0) {
          setMessages(data);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = async () => {
    setIsSidebarOpen(false); // Close overlay
    try {
      const res = await fetch("/api/chats/sessions", {
        method: "POST",
        headers: { "x-user-email": session?.user?.email || "" }
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentSessionId(data.session_id);
        setMessages([defaultMessage]);
        fetchSessions();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    let targetSessionId = currentSessionId;
    if (!targetSessionId) {
      try {
        const res = await fetch("/api/chats/sessions", {
          method: "POST",
          headers: { "x-user-email": session?.user?.email || "" }
        });
        if (res.ok) {
          const data = await res.json();
          targetSessionId = data.session_id;
          setCurrentSessionId(targetSessionId);
          fetchSessions();
        }
      } catch (e) {
        console.error(e);
      }
    }

    const userQuery = input.trim();
    setInput("");
    
    setMessages(prev => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("question", userQuery);
      if (targetSessionId) formData.append("session_id", targetSessionId);

      const response = await fetch("/api/ask/", {
        method: "POST",
        headers: {
          "x-user-email": session?.user?.email || ""
        },
        body: formData
      });

      if (!response.ok) throw new Error("Failed to fetch answer");

      const data = await response.json();
      
      setMessages(prev => [
        ...prev, 
        { 
          role: "ai", 
          content: data.response, 
          sources: data.sources || [] 
        }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        {
          role: "ai",
          content: "Sorry, I encountered an error. Make sure the backend is running and you've uploaded some documents."
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100%', position: 'relative', width: '100%', overflow: 'hidden' }}>
      
      {/* OVERLAY BACKDROP */}
      {isSidebarOpen && (
        <div 
          className="animate-fade-in"
          onClick={() => setIsSidebarOpen(false)}
          style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', zIndex: 40, cursor: 'pointer'
          }}
        />
      )}

      {/* SESSIONS SIDEBAR OVERLAY */}
      <div 
        className="glass-panel" 
        style={{ 
          position: 'absolute', top: 0, left: 0, height: '100%', width: '320px', zIndex: 50,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
          borderRadius: '0 20px 20px 0', borderLeft: 'none',
          transform: isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition: 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
          boxShadow: isSidebarOpen ? '20px 0 50px rgba(0,0,0,0.5)' : 'none'
        }}
      >
        {/* Profile Header */}
        <div style={{ padding: '24px 20px', borderBottom: '1px solid var(--glass-border)', background: 'rgba(139, 92, 246, 0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2 style={{ margin: 0, color: 'var(--accent-color)' }}>Navigation</h2>
            <button onClick={() => setIsSidebarOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
              <X size={20} />
            </button>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <UserCircle size={32} color="var(--text-primary)" />
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontSize: '15px', fontWeight: 600, color: '#fff' }}>{session?.user?.name || "User"}</div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {session?.user?.email || "No email"}
              </div>
            </div>
          </div>
        </div>

        <div style={{ padding: '20px', borderBottom: '1px solid var(--glass-border)' }}>
          <button 
            onClick={startNewChat}
            className="glass-button primary" 
            style={{ width: '100%', padding: '12px' }}
          >
            <PlusCircle size={18} /> New Chat
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <h3 style={{ fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px', paddingLeft: '4px' }}>Recent Chats</h3>
          {chatSessions.map(sess => (
            <button
              key={sess.id}
              onClick={() => loadSession(sess.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: '12px', padding: '14px',
                background: currentSessionId === sess.id ? 'rgba(139, 92, 246, 0.2)' : 'rgba(255,255,255,0.02)',
                border: currentSessionId === sess.id ? '1px solid var(--accent-light)' : '1px solid transparent',
                borderRadius: '12px', cursor: 'pointer', textAlign: 'left', color: 'var(--text-primary)',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => { if(currentSessionId !== sess.id) e.currentTarget.style.background = 'rgba(255,255,255,0.06)' }}
              onMouseLeave={(e) => { if(currentSessionId !== sess.id) e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
            >
              <MessageSquare size={16} color="var(--accent-color)" shrink={0} />
              <div style={{ overflow: 'hidden' }}>
                <div style={{ fontSize: '14px', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {sess.title}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {new Date(sess.created_at).toLocaleDateString()}
                </div>
              </div>
            </button>
          ))}
          {chatSessions.length === 0 && (
            <div style={{ fontSize: '14px', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '20px' }}>
              No chats found.
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area (100% Width naturally, but content centered) */}
      <div className="main-content glass-panel animate-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', background: 'rgba(15, 17, 21, 0.85)' }}>
        
        {/* Header Ribbon */}
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '16px', background: 'linear-gradient(90deg, rgba(139, 92, 246, 0.1), rgba(236, 72, 153, 0.05))' }}>
          <button 
            onClick={() => setIsSidebarOpen(true)}
            style={{ background: 'none', border: 'none', color: 'var(--text-primary)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', padding: '8px', borderRadius: '8px', transition: 'background 0.2s' }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <Menu size={24} />
          </button>
          <div>
            <h2 style={{ margin: 0 }}>InsightsPop Analytics</h2>
            <p style={{ margin: 0, fontSize: '13px' }}>Powered by Llama 3 & RAG Vector Inference</p>
          </div>
        </div>

        {/* Scrollable Chat History (Centered 50%-ish Width via max-width) */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ width: '100%', maxWidth: '800px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {messages.map((msg, index) => (
              <div key={index} className="animate-slide-in" style={{
                display: 'flex',
                gap: '16px',
                width: '100%',
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
              }}>
                <div style={{ 
                  width: '40px', height: '40px', borderRadius: '50%', flexShrink: 0, 
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: msg.role === 'user' ? 'var(--accent-gradient)' : 'rgba(255,255,255,0.05)',
                  border: msg.role === 'ai' ? '1px solid rgba(255,255,255,0.1)' : 'none',
                  boxShadow: '0 4px 10px rgba(0,0,0,0.2)'
                }}>
                  {msg.role === 'user' ? <User size={20} color="white" /> : <Bot size={20} color="var(--accent-color)" />}
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxWidth: '85%' }}>
                  <div style={{ 
                    padding: '20px', 
                    borderRadius: msg.role === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px',
                    background: msg.role === 'user' ? 'rgba(139, 92, 246, 0.15)' : 'rgba(0,0,0,0.3)',
                    border: msg.role === 'user' ? '1px solid rgba(139, 92, 246, 0.3)' : '1px solid var(--glass-border)',
                  }}>
                    {msg.role === 'ai' ? (
                      <div className="markdown-body">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <span style={{ color: 'var(--text-primary)', fontSize: '15.5px', lineHeight: 1.6 }}>{msg.content}</span>
                    )}
                  </div>

                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: '4px' }}>
                      <SourceChips sources={msg.sources} />
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div style={{ display: 'flex', gap: '16px', width: '100%' }}>
                <div style={{ 
                  width: '40px', height: '40px', borderRadius: '50%', flexShrink: 0, 
                  display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.05)'
                }}>
                  <Bot size={20} color="var(--accent-color)" />
                </div>
                <div className="glass-panel" style={{ padding: '20px', borderRadius: '20px 20px 20px 4px' }}>
                  <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                    <div style={{ width: '8px', height: '8px', background: 'var(--accent-color)', borderRadius: '50%', animation: 'pulse 1s infinite' }} />
                    <div style={{ width: '8px', height: '8px', background: 'var(--accent-color)', borderRadius: '50%', animation: 'pulse 1s infinite 0.2s' }} />
                    <div style={{ width: '8px', height: '8px', background: 'var(--accent-color)', borderRadius: '50%', animation: 'pulse 1s infinite 0.4s' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Footer (Centered) */}
        <div style={{ padding: '24px', borderTop: '1px solid var(--glass-border)', background: 'linear-gradient(0deg, rgba(0,0,0,0.4) 0%, transparent 100%)', display: 'flex', justifyContent: 'center' }}>
          <form onSubmit={handleSubmit} style={{ width: '100%', maxWidth: '800px', display: 'flex', gap: '12px' }}>
            <input
              type="text"
              className="glass-input"
              placeholder="Message your research assistant..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              style={{ background: 'rgba(255,255,255,0.03)', fontSize: '15px', padding: '16px 24px', borderRadius: '40px' }}
            />
            <button type="submit" className="glass-button primary" disabled={isLoading || !input.trim()} style={{ borderRadius: '50%', width: '54px', height: '54px', padding: 0 }}>
              {isLoading ? <Loader2 size={20} className="animate-pulse" style={{ animation: 'spin 1s linear infinite' }} /> : <Send size={20} style={{ marginLeft: '-2px', marginTop: '2px' }} />} 
            </button>
          </form>
        </div>
        
      </div>
    </div>
  );
}

function SourceChips({ sources }) {
  const [expanded, setExpanded] = useState(false);
  
  if(!sources || sources.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <button 
        onClick={() => setExpanded(!expanded)}
        style={{
          background: 'none', border: 'none', color: 'var(--text-secondary)',
          fontSize: '13px', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer',
          padding: 0, transition: 'color 0.2s'
        }}
        onMouseEnter={e => e.currentTarget.style.color = '#fff'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-secondary)'}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        View {sources.length} Linked Sources
      </button>

      {expanded && (
        <div className="animate-slide-in" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {sources.map((src, i) => (
             <div key={i} style={{
               background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)',
               borderRadius: '10px', padding: '12px 14px', fontSize: '13px', color: 'var(--text-secondary)',
               display: 'flex', flexWrap: 'wrap'
             }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-color)' }}>
                 <FileText size={14} /> {src}
               </div>
             </div>
          ))}
        </div>
      )}
    </div>
  );
}

