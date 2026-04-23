"use client";

import { useState, useEffect, useCallback } from "react";
import { Mail, Lock, User, ArrowRight, Loader2, KeyRound, ArrowLeft } from "lucide-react";

// ─── Custom Google button SVG ────────────────────────────────────────────────
function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" style={{ flexShrink: 0 }}>
      <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
      <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
      <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
      <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.35-8.16 2.35-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
    </svg>
  );
}

export default function AuthPage({ onLogin, initialError = "" }) {
  // "login" | "register" | "verify-otp"
  const [view, setView] = useState("login");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [showGoogleHint, setShowGoogleHint] = useState(false);

  // Form fields
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [otp, setOtp] = useState("");

  // Remember Me
  const [rememberMe, setRememberMe] = useState(false);

  // GIS script loaded flag
  const [gisLoaded, setGisLoaded] = useState(false);

  const googleClientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  const hasClientId = !!(googleClientId && googleClientId !== "PASTE_YOUR_GOOGLE_CLIENT_ID_HERE");

  // Load remembered credentials on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("insightspop_remembered");
      if (saved) {
        const { email: savedEmail, password: savedPassword } = JSON.parse(saved);
        if (savedEmail) setEmail(savedEmail);
        if (savedPassword) setPassword(savedPassword);
        setRememberMe(true);
      }
    } catch (_) {}
  }, []);

  useEffect(() => {
    if (initialError) {
      setError(initialError);
      if (initialError.toLowerCase().includes("google")) setShowGoogleHint(true);
    }
  }, [initialError]);

  // ---------- Google Identity Services ----------
  const handleGoogleResponse = useCallback(async (response) => {
    try {
      const payload = JSON.parse(atob(response.credential.split(".")[1]));
      const res = await fetch("/api/auth/google", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: payload.name, email: payload.email }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Google auth failed");
      onLogin(data.user);
    } catch (err) {
      setError(err.message);
    }
  }, [onLogin]);

  // Load GIS script once on mount
  useEffect(() => {
    if (!hasClientId) return;
    if (window.google?.accounts?.id) { setGisLoaded(true); return; }

    const existing = document.querySelector('script[src="https://accounts.google.com/gsi/client"]');
    if (existing) {
      existing.addEventListener("load", () => setGisLoaded(true));
      return;
    }
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => setGisLoaded(true);
    document.head.appendChild(script);
  }, [hasClientId]);

  // Re-render the Google button into the correct container whenever
  // the view changes or GIS finishes loading.
  useEffect(() => {
    if (!hasClientId || !gisLoaded) return;

    // Wait one frame so React finishes painting the new view's DOM
    const raf = requestAnimationFrame(() => {
      const btnId =
        view === "login" ? "google-btn-login" : "google-btn-register";
      const container = document.getElementById(btnId);
      if (!container) return;

      window.google.accounts.id.initialize({
        client_id: googleClientId,
        callback: handleGoogleResponse,
      });

      container.innerHTML = ""; // clear any stale render
      window.google.accounts.id.renderButton(container, {
        theme: "filled_black",
        size: "large",
        width: 340,
        text: view === "login" ? "continue_with" : "signup_with",
        shape: "pill",
      });
    });

    return () => cancelAnimationFrame(raf);
  }, [gisLoaded, view, hasClientId, googleClientId, handleGoogleResponse]);

  // ---------- Email / Password Auth ----------
  const handleLogin = async (e) => {
    e.preventDefault();
    setError(""); setShowGoogleHint(false); setIsLoading(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed");

      if (rememberMe) {
        localStorage.setItem("insightspop_remembered", JSON.stringify({ email, password }));
      } else {
        localStorage.removeItem("insightspop_remembered");
      }
      onLogin(data.user);
    } catch (err) {
      setError(err.message);
      if (String(err.message || "").toLowerCase().includes("google")) {
        setShowGoogleHint(true);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError(""); setSuccess(""); setShowGoogleHint(false); setIsLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Registration failed");
      setSuccess(data.message);
      setView("verify-otp");
    } catch (err) {
      setError(err.message);
      if (String(err.message || "").toLowerCase().includes("google")) {
        setShowGoogleHint(true);
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
        body: JSON.stringify({ email, code: otp }),
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

  const switchView = (nextView) => {
    setView(nextView);
    setError("");
    setSuccess("");
    setShowGoogleHint(false);
  };

  // ── Shared divider ──────────────────────────────────────────────────────────
  const Divider = () => (
    <div style={{ display: "flex", alignItems: "center", gap: "12px", margin: "20px 0" }}>
      <div style={{ flex: 1, height: "1px", background: "var(--glass-border)" }} />
      <span style={{ color: "var(--text-secondary)", fontSize: "11px", letterSpacing: "0.05em" }}>OR</span>
      <div style={{ flex: 1, height: "1px", background: "var(--glass-border)" }} />
    </div>
  );

  // ── Google button slot — rendered by GIS SDK into this div ─────────────────
  // When client ID is not set, we show a disabled placeholder instead.
  const GoogleSlot = ({ id, label }) => (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "6px" }}>
      {hasClientId ? (
        // GIS SDK will inject the real button here
        <div id={id} style={{ display: "flex", justifyContent: "center", minHeight: "44px" }} />
      ) : (
        // Fallback when no client ID is configured
        <button
          type="button"
          disabled
          style={{
            width: "100%", display: "flex", alignItems: "center", justifyContent: "center",
            gap: "10px", padding: "13px 20px",
            background: "rgba(255,255,255,0.04)", border: "1.5px solid rgba(255,255,255,0.1)",
            borderRadius: "10px", color: "rgba(255,255,255,0.4)", fontSize: "14px",
            fontWeight: 500, cursor: "not-allowed",
          }}
        >
          <GoogleIcon />
          <span>{label}</span>
        </button>
      )}
      {showGoogleHint && (
        <p style={{ fontSize: "12px", color: "#f87171", textAlign: "center", margin: 0 }}>
          ↑ This email uses Google Sign-In — please use the button above
        </p>
      )}
    </div>
  );

  return (
    <div style={{
      height: "100vh", display: "flex", alignItems: "center", justifyContent: "center",
      background: "var(--bg-dark)",
      backgroundImage: "radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.15), transparent 25%), radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.15), transparent 25%)",
    }}>
      <div className="glass-panel animate-fade-in" style={{ padding: "40px", width: "100%", maxWidth: "420px" }}>
        <h1 style={{ fontSize: "32px", marginBottom: "8px", textAlign: "center" }}>InsightsPop</h1>
        <p style={{ textAlign: "center", marginBottom: "32px", color: "var(--text-secondary)", fontSize: "14px" }}>
          {view === "login" && "Welcome back! Sign in to continue."}
          {view === "register" && "Create your research account."}
          {view === "verify-otp" && "Enter the code sent to your email."}
        </p>

        {/* Error banner */}
        {error && (
          <div style={{
            background: showGoogleHint ? "rgba(234,67,53,0.12)" : "rgba(239,68,68,0.1)",
            border: `1px solid ${showGoogleHint ? "rgba(234,67,53,0.5)" : "rgba(239,68,68,0.3)"}`,
            borderRadius: "8px", padding: "10px 14px", marginBottom: "16px",
            fontSize: "13px", color: showGoogleHint ? "#f87171" : "#ef4444",
            display: "flex", alignItems: "flex-start", gap: "8px",
          }}>
            {showGoogleHint && <span style={{ fontSize: "16px", lineHeight: 1 }}>⚠️</span>}
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div style={{ background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.3)", borderRadius: "8px", padding: "10px 14px", marginBottom: "16px", fontSize: "13px", color: "#10b981" }}>
            {success}
          </div>
        )}

        {/* =================== LOGIN VIEW =================== */}
        {view === "login" && (
          <>
            {/* Google button highlighted at top when hint is active */}
            {showGoogleHint && (
              <>
                <GoogleSlot id="google-btn-login" label="Sign in with Google" />
                <Divider />
              </>
            )}

            <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <div style={{ position: "relative" }}>
                <Mail size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Email" type="email" required value={email} onChange={e => setEmail(e.target.value)} style={{ paddingLeft: "38px" }} />
              </div>
              <div style={{ position: "relative" }}>
                <Lock size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Password" type="password" required value={password} onChange={e => setPassword(e.target.value)} style={{ paddingLeft: "38px" }} />
              </div>

              {/* Remember Me */}
              <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer", userSelect: "none", fontSize: "14px", color: "var(--text-secondary)" }}>
                <span style={{ position: "relative", display: "inline-block", width: "18px", height: "18px", flexShrink: 0 }}>
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={e => {
                      setRememberMe(e.target.checked);
                      if (!e.target.checked) localStorage.removeItem("insightspop_remembered");
                    }}
                    style={{ opacity: 0, position: "absolute", width: "100%", height: "100%", margin: 0, cursor: "pointer" }}
                  />
                  <span style={{
                    display: "block", width: "18px", height: "18px",
                    border: `2px solid ${rememberMe ? "var(--accent-color)" : "var(--glass-border)"}`,
                    borderRadius: "5px",
                    background: rememberMe ? "var(--accent-color)" : "transparent",
                    transition: "all 0.2s ease",
                    boxShadow: rememberMe ? "0 0 8px rgba(139,92,246,0.4)" : "none",
                  }}>
                    {rememberMe && (
                      <svg viewBox="0 0 12 10" width="12" height="10" style={{ display: "block", margin: "1px auto", fill: "none", stroke: "#fff", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" }}>
                        <polyline points="1,5 4.5,8.5 11,1" />
                      </svg>
                    )}
                  </span>
                </span>
                Remember me
              </label>

              <button type="submit" className="glass-button primary" style={{ width: "100%", padding: "14px" }} disabled={isLoading}>
                {isLoading ? <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> : <><ArrowRight size={18} /> Sign In</>}
              </button>
            </form>

            {/* Google button below form (normal state) */}
            {!showGoogleHint && (
              <>
                <Divider />
                <GoogleSlot id="google-btn-login" label="Continue with Google" />
              </>
            )}

            <p style={{ textAlign: "center", marginTop: "24px", fontSize: "14px" }}>
              Don&apos;t have an account?{" "}
              <button onClick={() => switchView("register")} style={{ background: "none", border: "none", color: "var(--accent-color)", cursor: "pointer", fontWeight: 600, fontSize: "14px" }}>
                Create Account
              </button>
            </p>
          </>
        )}

        {/* =================== REGISTER VIEW =================== */}
        {view === "register" && (
          <>
            {/* Google Sign-Up always at top */}
            <GoogleSlot id="google-btn-register" label="Sign up with Google" />
            <Divider />

            <form onSubmit={handleRegister} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <div style={{ position: "relative" }}>
                <User size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Full Name" type="text" required value={name} onChange={e => setName(e.target.value)} style={{ paddingLeft: "38px" }} />
              </div>
              <div style={{ position: "relative" }}>
                <Mail size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Email" type="email" required value={email} onChange={e => setEmail(e.target.value)} style={{ paddingLeft: "38px" }} />
              </div>
              <div style={{ position: "relative" }}>
                <Lock size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Password (min 6 chars)" type="password" required minLength={6} value={password} onChange={e => setPassword(e.target.value)} style={{ paddingLeft: "38px" }} />
              </div>
              <button type="submit" className="glass-button primary" style={{ width: "100%", padding: "14px" }} disabled={isLoading}>
                {isLoading ? <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> : <><Mail size={18} /> Send Verification Code</>}
              </button>
            </form>

            <p style={{ textAlign: "center", marginTop: "24px", fontSize: "14px" }}>
              <button onClick={() => switchView("login")} style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer", fontSize: "14px", display: "flex", alignItems: "center", gap: "4px", margin: "0 auto" }}>
                <ArrowLeft size={14} /> Back to Sign In
              </button>
            </p>
          </>
        )}

        {/* =================== OTP VERIFICATION VIEW =================== */}
        {view === "verify-otp" && (
          <>
            <form onSubmit={handleVerifyOTP} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              <p style={{ textAlign: "center", fontSize: "13px", color: "var(--text-secondary)" }}>
                We sent a 6-digit code to <strong style={{ color: "var(--accent-color)" }}>{email}</strong>
              </p>
              <div style={{ position: "relative" }}>
                <KeyRound size={16} style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)", color: "var(--text-secondary)" }} />
                <input className="glass-input" placeholder="Enter 6-digit code" type="text" required maxLength={6} value={otp} onChange={e => setOtp(e.target.value.replace(/[^0-9]/g, ""))} style={{ paddingLeft: "38px", letterSpacing: "4px", fontSize: "20px", textAlign: "center" }} />
              </div>
              <button type="submit" className="glass-button primary" style={{ width: "100%", padding: "14px" }} disabled={isLoading || otp.length !== 6}>
                {isLoading ? <Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /> : <><KeyRound size={18} /> Verify &amp; Sign In</>}
              </button>
            </form>

            <p style={{ textAlign: "center", marginTop: "24px", fontSize: "14px" }}>
              <button onClick={() => switchView("register")} style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer", fontSize: "14px", display: "flex", alignItems: "center", gap: "4px", margin: "0 auto" }}>
                <ArrowLeft size={14} /> Back to register
              </button>
            </p>
          </>
        )}
      </div>
    </div>
  );
}
