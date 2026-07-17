import type { Metadata } from "next";
import { Inter } from "next/font/google";
import type { ReactNode } from "react";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Match Intelligence — Agentic Soccer Prediction",
  description:
    "Calibrated match predictions with conformal uncertainty, market edges, and an evidence-grounded agent.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`dark ${inter.variable}`}>
      <body className="font-sans">
        <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4">
          {children}
        </div>
      </body>
    </html>
  );
}
