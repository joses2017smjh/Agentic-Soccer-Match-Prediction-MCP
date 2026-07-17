/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone", // small Docker runtime image
  // The FastAPI gateway is reached ONLY from Route Handlers (server side);
  // its URL and API key never ship to the browser.
};

export default nextConfig;
