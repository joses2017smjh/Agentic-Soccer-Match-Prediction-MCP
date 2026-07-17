import type { Config } from "tailwindcss";

/**
 * Design tokens — strict dark mode, sportsbook-density.
 *
 * Three surface depths, two border weights, three ink tiers; a single
 * green/red pair reserved exclusively for edge/EV semantics (positive =
 * value, negative = no value) so color always means the same thing, and an
 * amber brand accent for interactive emphasis. High contrast throughout:
 * ink-100 on surface-950 is ~15:1.
 */
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          950: "#06080C", // page
          900: "#0B0F16", // card
          800: "#121826", // elevated / hover
        },
        line: {
          DEFAULT: "#1C2433", // subtle borders everywhere
          strong: "#2A3550",
        },
        ink: {
          100: "#EDF1F7", // primary text
          400: "#94A0B8", // secondary
          600: "#5C6880", // muted / labels
        },
        edge: {
          pos: "#4ADE80", // positive EV only
          neg: "#F87171", // negative EV only
        },
        brand: {
          DEFAULT: "#F5B638", // interactive accent (buttons, focus)
          dim: "#8A6A24",
        },
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }], // dense table text
      },
    },
  },
  plugins: [],
};

export default config;
