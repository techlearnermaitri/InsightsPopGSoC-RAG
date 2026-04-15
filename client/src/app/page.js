"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import UploadPanel from "../components/UploadPanel";
import ChatPanel from "../components/ChatPanel";
import { useSession } from "../context/SessionContext";
import { LogOut, Plus } from "lucide-react";

export default function Page() {
  const router = useRouter();
  const { activeUser, sessions, switchSession, removeSession, loading } = useSession();
  const [validating, setValidating] = useState(true);

  // Validate active user session with backend
  useEffect(() => {
    const validateSession = async () => {
      if (loading) return;
      
      if (!activeUser) {
        router.push("/auth");
        return;
      }

      try {
        const res = await fetch("/api/auth/me", {
          headers: { "x-user-email": activeUser?.email || "" },
        });

        if (!res.ok) {
          router.push("/auth");
          return;
        }

        setValidating(false);
      } catch {
        router.push("/auth");
      }
    };

    validateSession();
  }, [activeUser, loading, router]);

  const handleLogoutSession = (index) => {
    removeSession(index);
    if (sessions.length <= 1) {
      router.push("/auth");
    }
  };

  const handleAddAccount = () => {
    router.push("/auth");
  };

  if (loading || validating) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Loading...</p>
      </div>
    );
  }

  if (!activeUser) {
    return null; // Redirecting to /auth
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header className="glass-panel" style={{
        padding: '12px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        borderRadius: 0, borderTop: 'none', borderLeft: 'none', borderRight: 'none'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h2 style={{ margin: 0, color: 'var(--accent-color)' }}>InsightsPop</h2>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
            Welcome, {activeUser.name}
          </span>
          
          {sessions.length > 1 && (
            <select 
              value={sessions.findIndex(u => u.email === activeUser.email)}
              onChange={(e) => switchSession(parseInt(e.target.value))}
              className="glass-button"
              style={{ 
                padding: '6px 10px', 
                fontSize: '13px',
                background: 'rgba(139, 92, 246, 0.2)',
                border: '1px solid rgba(139, 92, 246, 0.3)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                cursor: 'pointer'
              }}
            >
              {sessions.map((user, idx) => (
                <option key={idx} value={idx}>
                  {user.name} ({user.email})
                </option>
              ))}
            </select>
          )}

          <button 
            onClick={handleAddAccount}
            className="glass-button" 
            style={{ padding: '6px 12px', fontSize: '13px', background: 'rgba(139, 92, 246, 0.2)' }}
          >
            <Plus size={14} /> Add Account
          </button>

          <button 
            onClick={() => handleLogoutSession(sessions.findIndex(u => u.email === activeUser.email))}
            className="glass-button" 
            style={{ padding: '6px 12px', fontSize: '13px' }}
          >
            <LogOut size={14} /> Sign out
          </button>
        </div>
      </header>
      <main className="app-container" style={{ paddingTop: '16px' }}>
        <UploadPanel session={{ user: activeUser }} />
        <ChatPanel session={{ user: activeUser }} />
      </main>
    </div>
  );
}
