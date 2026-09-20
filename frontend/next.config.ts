import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // The dev overlay badge sits exactly where the first live metric renders.
  devIndicators: false,
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
