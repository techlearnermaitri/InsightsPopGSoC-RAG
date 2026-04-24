/**
 * /api/upload_pdf — Next.js App Router passthrough route.
 *
 * WHY THIS FILE EXISTS:
 * Next.js rewrites (next.config.js) consume and drop the raw multipart/form-data
 * body before proxying, so FastAPI receives an empty body and returns 422/500.
 * This route opts out of body parsing entirely and pipes the raw stream directly
 * to the FastAPI backend, preserving all file bytes and headers.
 */

export const runtime = "nodejs";

// Tell Next.js NOT to parse the body — we forward it raw.
export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

async function handler(request) {
  const backendTarget = `${BACKEND_URL}/upload_pdf/`;

  // Clone the headers, stripping host to avoid "invalid host" errors from uvicorn.
  const forwardHeaders = new Headers();
  request.headers.forEach((value, key) => {
    if (key.toLowerCase() !== "host") {
      forwardHeaders.set(key, value);
    }
  });

  try {
    // Buffer the entire body into memory first.
    // Streaming via request.body + duplex:"half" is unreliable in Next.js App Router
    // and causes TypeError: fetch failed with no useful cause.
    const bodyBuffer = await request.arrayBuffer();

    const response = await fetch(backendTarget, {
      method: "POST",
      headers: forwardHeaders,
      body: bodyBuffer,
    });

    const data = await response.json();
    return new Response(JSON.stringify(data), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("[upload_pdf proxy] Error forwarding to backend:", err);
    return new Response(JSON.stringify({ error: "Proxy error: " + err.message }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export { handler as POST };
