"use client";

import { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, File, CheckCircle, AlertCircle, Loader2, Edit2, Check, X } from "lucide-react";

export default function UploadPanel({ session }) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [message, setMessage] = useState("");
  
  const [editingFile, setEditingFile] = useState(null);
  const [editName, setEditName] = useState("");

  const fetchFiles = useCallback(async () => {
    if (!session?.user?.email) return;
    try {
      const res = await fetch("/api/files", {
        headers: { "x-user-email": session.user.email }
      });
      if (res.ok) {
        const data = await res.json();
        setUploadedFiles(data);
      }
    } catch (e) {
      console.error("Failed to fetch files", e);
    }
  }, [session]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0 || !session?.user?.email) return;
    
    setIsUploading(true);
    setUploadStatus(null);
    setMessage("");
    
    const formData = new FormData();
    acceptedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await fetch("/api/upload_pdf/", {
        method: "POST",
        headers: { "x-user-email": session.user.email },
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to upload");

      const data = await response.json();
      await fetchFiles(); // Refresh DB list
      setUploadStatus("success");
      setMessage(data.message || "Files processed and vector store updated!");
      
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus("error");
      setMessage(error.message || "Failed to process files");
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadStatus(null), 5000);
    }
  }, [session, fetchFiles]);

  const handleRename = async (oldName, newName) => {
    if (!newName.trim() || oldName === newName) {
      setEditingFile(null);
      return;
    }
    try {
      const response = await fetch("/api/files/rename", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "x-user-email": session.user.email
        },
        body: JSON.stringify({ old_name: oldName, new_name: newName })
      });
      if (response.ok) {
        await fetchFiles();
      }
    } catch (e) {
      console.error("Failed to rename:", e);
    } finally {
      setEditingFile(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] }
  });

  return (
    <div className="sidebar glass-panel animate-fade-in" style={{ borderRadius: '16px 0 0 16px', borderRight: 'none' }}>
      <div 
        {...getRootProps()} 
        style={{
          border: '2px dashed var(--glass-border)',
          borderRadius: '12px',
          padding: '30px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          background: isDragActive ? 'rgba(139, 92, 246, 0.1)' : 'rgba(0,0,0,0.2)',
          transition: 'all 0.2s ease',
          borderColor: isDragActive ? 'var(--accent-color)' : uploadStatus === 'error' ? 'var(--danger-color)' : 'var(--glass-border)'
        }}
      >
        <input {...getInputProps()} />
        {isUploading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <Loader2 className="animate-pulse" size={32} color="var(--accent-color)" style={{ animation: 'spin 1s linear infinite' }} />
            <p style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Processing Document...</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <UploadCloud size={32} color={isDragActive ? "var(--accent-color)" : "var(--text-secondary)"} />
            {isDragActive ? (
              <p style={{ color: 'var(--accent-color)', fontWeight: 500 }}>Drop PDFs here...</p>
            ) : (
              <p>Drag & drop PDFs here, or click to select files</p>
            )}
          </div>
        )}
      </div>

      {uploadStatus === "success" && (
        <div style={{ display: 'flex', gap: '8px', color: 'var(--success-color)', fontSize: '13px', alignItems: 'center', background: 'rgba(16, 185, 129, 0.1)', padding: '10px', borderRadius: '8px' }}>
          <CheckCircle size={16} /> <span>{message}</span>
        </div>
      )}

      {uploadStatus === "error" && (
        <div style={{ display: 'flex', gap: '8px', color: 'var(--danger-color)', fontSize: '13px', alignItems: 'center', background: 'rgba(239, 68, 68, 0.1)', padding: '10px', borderRadius: '8px' }}>
          <AlertCircle size={16} /> <span>{message}</span>
        </div>
      )}

      <div style={{ flex: 1, overflowY: 'auto', marginTop: '10px' }}>
        <h2><File size={18} /> My Library</h2>
        {uploadedFiles.length === 0 ? (
          <p style={{ fontStyle: 'italic', opacity: 0.6 }}>No documents available.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {uploadedFiles.map((file, i) => (
              <div key={i} style={{ 
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', 
                background: 'rgba(0,0,0,0.2)', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--glass-border)' 
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                  <File size={16} color="var(--accent-color)" style={{ flexShrink: 0 }} />
                  {editingFile === file.custom_name ? (
                     <input 
                       autoFocus
                       className="glass-input"
                       style={{ padding: '4px 8px', fontSize: '13px' }}
                       value={editName}
                       onChange={(e) => setEditName(e.target.value)}
                       onKeyDown={(e) => e.key === 'Enter' && handleRename(file.custom_name, editName)}
                     />
                  ) : (
                    <span style={{ fontSize: '13px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {file.custom_name}
                    </span>
                  )}
                </div>
                
                {editingFile === file.custom_name ? (
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button onClick={() => handleRename(file.custom_name, editName)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--success-color)' }}><Check size={14}/></button>
                    <button onClick={() => setEditingFile(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--danger-color)' }}><X size={14}/></button>
                  </div>
                ) : (
                  <button onClick={() => { setEditingFile(file.custom_name); setEditName(file.custom_name); }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                    <Edit2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
