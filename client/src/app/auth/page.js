"use client";

import { useRouter } from "next/navigation";
import { useSession } from "../../context/SessionContext";
import AuthPage from "../../components/AuthPage";

export default function AuthRoute() {
  const router = useRouter();
  const { addSession, sessions } = useSession();

  const handleLogin = (userData) => {
    // Add session to context (stores in localStorage automatically)
    addSession(userData);
    
    // Redirect to chat
    router.push("/");
  };

  return (
    <div>
      <AuthPage onLogin={handleLogin} initialError="" />
    </div>
  );
}
