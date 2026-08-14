import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import type { ReactNode } from "react";
import { ChatWidget } from "@/components/features/chat/chat-widget";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Match Intelligence — Agentic Soccer Prediction",
  description:
    "Calibrated match predictions with conformal uncertainty, market edges, and an evidence-grounded agent.",
};

// explicit so phones render at device width and honour the dark surface in
// the browser chrome; user scaling is left enabled (never disable zoom)
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#06080C",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="font-sans">
        <div className="mx-auto flex min-h-screen w-full min-w-0 max-w-7xl
          flex-col px-3 sm:px-4">
          {children}
        </div>
        <ChatWidget />
      </body>
    </html>
  );
}
