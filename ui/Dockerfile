# Two-stage build → small standalone runtime.
FROM node:20-alpine AS builder
WORKDIR /app
# `npm ci` installs exactly what package-lock.json pins and fails if the lock
# and package.json disagree, so a dependency bump cannot be half-applied
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
