import "./globals.css";
import { SessionProvider } from "../context/SessionContext";

export const metadata = {
  title: "InsightsPop - Research Assistant",
  description: "AI-powered Retrieval-Augmented Generation (RAG) system for insights extraction and analysis.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <SessionProvider>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
