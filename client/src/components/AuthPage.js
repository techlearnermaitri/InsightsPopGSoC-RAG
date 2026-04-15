"use client";

import { useState, useEffect, useCallback } from "react";
import { Mail, Lock, User, ArrowRight, Loader2, KeyRound, ArrowLeft } from "lucide-react";

export default function AuthPage({ onLogin, initialError = "" }) {
  // "login" | "register" | "verify-otp"
  const [view, setView] = useState("login");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  // Form fields
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [otp, setOtp] = useState("");

  useEffect(() => {
    if (initialError) {
      setError(initialError);
    }
  }, [initialError]);

  // ---------- Google Identity Services ----------
  const handleGoogleResponse = useCallback(async (response) => {
    // Decode the JWT credential to get user info
    const payload = JSON.parse(atob(response.credential.split('.')[1]));
    
    try {
      const res = await fetch("/api/auth/google", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: payload.name, email: payload.email })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Google auth failed");
      onLogin(data.user);
    } catch (err) {
      setError(err.message);
    }
  }, [onLogin]);

  useEffect(() => {
    const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
    if (!clientId || clientId === "PASTE_YOUR_GOOGLE_CLIENT_ID_HERE") return;

    // Load Google Identity Services script
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => {
      window.google?.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleResponse,
      });
      window.google?.accounts.id.renderButton(
        document.getElementById("google-signin-btn"),
        { 
          theme: "filled_black",
          size: "large",
          width: 340,
          text: "continue_with",
          shape: "pill"
        }
      );
    };
    document.head.appendChild(script);

    return () => {
      // Cleanup - remove script if component unmounts
      const existingScript = document.querySelector('script[src="https://accounts.google.com/gsi/client"]');
      if (existingScript) existingScript.remove();
    };
  }, [handleGoogleResponse]);

  // ---------- Email / Password Auth ----------
  const handleLogin = async (e) => {
    e.preventDefault();
    setError(""); setIsLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed");
      onLogin(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError(""); setSuccess(""); setIsLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Registration failed");
      setSuccess(data.message);
      setView("verify-otp");
    } catch (err) {
      setError(err.message);
      if (String(err.message || "").toLowerCase().includes("google")) {
        setView("login");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyOTP = async (e) => {
    e.preventDefault();
    setError(""); setIsLoading(true);
    try {
      const res = await fetch("/api/auth/verify-otp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, code: otp })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Verification failed");
      onLogin(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  const hasGoogleSetup = googleClientId && googleClientId !== "PASTE_YOUR_GOOGLE_CLIENT_ID_HERE";

  return (
    <div style={{
      height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg-dark)',
      backgroundImage: 'radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.15), transparent 25%), radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.15), transparent 25%)'
    }}>
      <div className="glass-panel animate-fade-in" style={{ padding: '40px', width: '100%', maxWidth: '420px' }}>
        <h1 style={{ fontSize: '32px', marginBottom: '8px', textAlign: 'center' }}>InsightsPop</h1>
        <p style={{ textAlign: 'center', marginBottom: '32px' }}>
          {view === "login" && "Welcome back! Sign in to continue."}
          {view === "register" && "Create your research account."}
          {view === "verify-otp" && "Enter the code sent to your email."}
        </p>

        {error && (
          <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px', padding: '10px 14px', marginBottom: '16px', fontSize: '13px', color: '#ef4444' }}>
            {error}
          </div>
        )}
        {success && (
          <div style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', borderRadius: '8px', padding: '10px 14px', marginBottom: '16px', fontSize: '13px', color: '#10b981' }}>
            {success}
          </div>
        )}

        {/* =================== LOGIN VIEW =================== */}
        {view === "login" && (
          <>
            <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ position: 'relative' }}>
                <Mail size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Email" type="email" required value={email} onChange={e => setEmail(e.target.value)} style={{ paddingLeft: '38px' }} />
              </div>
              <div style={{ position: 'relative' }}>
                <Lock size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Password" type="password" required value={password} onChange={e => setPassword(e.target.value)} style={{ paddingLeft: '38px' }} />
              </div>
              <button type="submit" className="glass-button primary" style={{ width: '100%', padding: '14px' }} disabled={isLoading}>
                {isLoading ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> : <><ArrowRight size={18} /> Sign In</>}
              </button>
            </form>

            {hasGoogleSetup && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '24px 0' }}>
                  <div style={{ flex: 1, height: '1px', background: 'var(--glass-border)' }} />
                  <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>OR</span>
                  <div style={{ flex: 1, height: '1px', background: 'var(--glass-border)' }} />
                </div>
                <div id="google-signin-btn" style={{ display: 'flex', justifyContent: 'center' }} />
              </>
            )}

            <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px' }}>
              Don&apos;t have an account?{" "}
              <button onClick={() => { setView("register"); setError(""); setSuccess(""); }} style={{ background: 'none', border: 'none', color: 'var(--accent-color)', cursor: 'pointer', fontWeight: 600, fontSize: '14px' }}>
                Create Account
              </button>
            </p>
          </>
        )}

        {/* =================== REGISTER VIEW =================== */}
        {view === "register" && (
          <>
            <form onSubmit={handleRegister} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ position: 'relative' }}>
                <User size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Full Name" type="text" required value={name} onChange={e => setName(e.target.value)} style={{ paddingLeft: '38px' }} />
              </div>
              <div style={{ position: 'relative' }}>
                <Mail size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Email" type="email" required value={email} onChange={e => setEmail(e.target.value)} style={{ paddingLeft: '38px' }} />
              </div>
              <div style={{ position: 'relative' }}>
                <Lock size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Password (min 6 chars)" type="password" required minLength={6} value={password} onChange={e => setPassword(e.target.value)} style={{ paddingLeft: '38px' }} />
              </div>
              <button type="submit" className="glass-button primary" style={{ width: '100%', padding: '14px' }} disabled={isLoading}>
                {isLoading ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> : <><Mail size={18} /> Send Verification Code</>}
              </button>
            </form>

            {hasGoogleSetup && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', margin: '24px 0' }}>
                  <div style={{ flex: 1, height: '1px', background: 'var(--glass-border)' }} />
                  <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>OR</span>
                  <div style={{ flex: 1, height: '1px', background: 'var(--glass-border)' }} />
                </div>
                <div id="google-signin-btn" style={{ display: 'flex', justifyContent: 'center' }} />
              </>
            )}

            <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px' }}>
              <button onClick={() => { setView("login"); setError(""); setSuccess(""); }} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '4px', margin: '0 auto' }}>
                <ArrowLeft size={14} /> Back to Sign In
              </button>
            </p>
          </>
        )}

        {/* =================== OTP VERIFICATION VIEW =================== */}
        {view === "verify-otp" && (
          <>
            <form onSubmit={handleVerifyOTP} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <p style={{ textAlign: 'center', fontSize: '13px', color: 'var(--text-secondary)' }}>
                We sent a 6-digit code to <strong style={{ color: 'var(--accent-color)' }}>{email}</strong>
              </p>
              <div style={{ position: 'relative' }}>
                <KeyRound size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
                <input className="glass-input" placeholder="Enter 6-digit code" type="text" required maxLength={6} value={otp} onChange={e => setOtp(e.target.value.replace(/[^0-9]/g, ""))} style={{ paddingLeft: '38px', letterSpacing: '4px', fontSize: '20px', textAlign: 'center' }} />
              </div>
              <button type="submit" className="glass-button primary" style={{ width: '100%', padding: '14px' }} disabled={isLoading || otp.length !== 6}>
                {isLoading ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> : <><KeyRound size={18} /> Verify &amp; Sign In</>}
              </button>
            </form>

            <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px' }}>
              <button onClick={() => { setView("register"); setError(""); setSuccess(""); }} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '4px', margin: '0 auto' }}>
                <ArrowLeft size={14} /> Back to register
              </button>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
