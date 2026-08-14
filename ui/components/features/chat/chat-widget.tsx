"use client";

/**
 * Floating chat widget: LLM conversational layer that can price parlays,
 * route users to pages, explain the model, and handle prediction requests.
 *
 * Sits in the bottom-right of every page via the layout.  Opens/closes with
 * a toggle button.  Messages are local state (not persisted across page
 * navigations -- intentionally lightweight).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { ChatResponse } from "@/lib/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  action?: { type: "navigate"; route: string };
}

const WELCOME: Message = {
  role: "assistant",
  content:
    "Hi! I can help with match predictions, correlated parlay pricing, " +
    "league standings, and model explanations. Try:\n\n" +
    '- "Price a parlay with home win + over 2.5"\n' +
    '- "Predict Arsenal vs Man City"\n' +
    '- "Show me La Liga standings"\n' +
    '- "How does the model work?"',
};

export function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([WELCOME]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // focus input when opened
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-10).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "Sorry, something went wrong. Try again." },
        ]);
      } else {
        const data = (await res.json()) as ChatResponse;
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: data.reply, action: data.action },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Network error. Is the backend running?" },
      ]);
    } finally {
      setBusy(false);
    }
  }, [input, busy, messages]);

  function handleNavigate(route: string) {
    router.push(route);
    setOpen(false);
  }

  return (
    <>
      {/* toggle button */}
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close chat" : "Open chat"}
        className="fixed bottom-4 right-4 z-50 flex h-12 w-12 items-center
          justify-center rounded-full bg-brand text-surface-950 shadow-lg
          transition-transform hover:scale-105 active:scale-95 sm:bottom-6 sm:right-6"
      >
        {open ? (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
            <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
            <path fillRule="evenodd" d="M10 2c-2.236 0-4.43.18-6.57.524C1.993 2.755 1 3.993 1 5.396v3.208c0 1.403.993 2.64 2.43 2.872 1.202.194 2.424.336 3.662.424.083.044.167.085.25.126L10 13.75l2.658-1.724c.083-.04.167-.082.25-.126 1.238-.088 2.46-.23 3.662-.424C18.007 11.244 19 10.007 19 8.604V5.396c0-1.403-.993-2.64-2.43-2.872A49.2 49.2 0 0010 2zM6.5 6a1 1 0 000 2h7a1 1 0 100-2h-7zM6 9.5a1 1 0 011-1h2a1 1 0 110 2H7a1 1 0 01-1-1z" clipRule="evenodd" />
            <path d="M2.002 12.298a.75.75 0 00-.752 1.298A18.455 18.455 0 005 15.623v1.627a.75.75 0 001.244.563l1.942-1.713a18.6 18.6 0 001.814-.12c-1.86-.232-3.588-.7-5.13-1.37a17.956 17.956 0 01-2.868-2.312z" />
          </svg>
        )}
      </button>

      {/* chat panel */}
      {open && (
        <div
          className="fixed bottom-20 right-4 z-50 flex w-80 flex-col
            overflow-hidden rounded-lg border border-line bg-surface-900
            shadow-2xl sm:bottom-20 sm:right-6 sm:w-96"
          style={{ maxHeight: "min(32rem, calc(100vh - 8rem))" }}
        >
          {/* header */}
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <span className="text-sm font-semibold text-ink-100">
              Match<span className="text-brand">Intel</span> Chat
            </span>
            <span className="text-2xs text-ink-600">parlays + routing + Q&A</span>
          </div>

          {/* messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3">
            <div className="flex flex-col gap-3">
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed
                      ${msg.role === "user"
                        ? "bg-brand/15 text-ink-100"
                        : "bg-surface-800 text-ink-300"}`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    {msg.action?.type === "navigate" && (
                      <button
                        onClick={() => handleNavigate(msg.action!.route)}
                        className="mt-2 inline-block rounded border border-brand/50 px-2 py-1
                          text-2xs font-semibold text-brand hover:bg-brand/10"
                      >
                        Go to {msg.action.route}
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {busy && (
                <div className="flex justify-start">
                  <div className="rounded-lg bg-surface-800 px-3 py-2 text-xs text-ink-600">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* input */}
          <div className="border-t border-line px-3 py-2">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                placeholder="Ask about parlays, predictions, or navigation..."
                disabled={busy}
                className="w-full rounded border border-line bg-surface-800 px-2 py-1.5
                  text-xs text-ink-100 placeholder:text-ink-600
                  focus:border-brand focus:outline-none"
              />
              <button
                onClick={send}
                disabled={!input.trim() || busy}
                className="shrink-0 rounded bg-brand px-3 py-1.5 text-xs
                  font-semibold text-surface-950 disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
