"use client";

import { createContext, useContext, useState, useEffect } from "react";

const SessionContext = createContext();

export function SessionProvider({ children }) {
  const [sessions, setSessions] = useState([]); // Array of {user, token}
  const [activeSessionId, setActiveSessionId] = useState(null); // Index of active session
  const [loading, setLoading] = useState(true);

  // Load sessions from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("sessions");
    const activeId = localStorage.getItem("activeSessionId");
    
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSessions(parsed);
        
        // Set active session (default to first)
        const active = activeId ? parseInt(activeId) : (parsed.length > 0 ? 0 : null);
        if (active !== null && active < parsed.length) {
          setActiveSessionId(active);
        }
      } catch (e) {
        console.error("Failed to load sessions:", e);
      }
    }
    setLoading(false);
  }, []);

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    if (!loading) {
      localStorage.setItem("sessions", JSON.stringify(sessions));
      localStorage.setItem("activeSessionId", activeSessionId?.toString() || "0");
    }
  }, [sessions, activeSessionId, loading]);

  const addSession = (user) => {
    const newSessions = [...sessions, user];
    setSessions(newSessions);
    setActiveSessionId(newSessions.length - 1);
  };

  const removeSession = (index) => {
    const newSessions = sessions.filter((_, i) => i !== index);
    setSessions(newSessions);
    
    // Reset active session if removed
    if (activeSessionId === index) {
      setActiveSessionId(newSessions.length > 0 ? 0 : null);
    } else if (activeSessionId > index) {
      setActiveSessionId(activeSessionId - 1);
    }
  };

  const switchSession = (index) => {
    if (index >= 0 && index < sessions.length) {
      setActiveSessionId(index);
    }
  };

  const activeUser = activeSessionId !== null ? sessions[activeSessionId] : null;

  return (
    <SessionContext.Provider
      value={{
        sessions,
        activeUser,
        activeSessionId,
        addSession,
        removeSession,
        switchSession,
        loading,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useSession must be used within SessionProvider");
  }
  return context;
}
