# Stage 1: build the static bundle.
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json ./
RUN npm install --no-audit --no-fund

COPY vite.config.js index.html ./
COPY src ./src
RUN npm run build

# Stage 2: serve it. No node runtime in the final image, and no build tooling.
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
# 127.0.0.1, not localhost. BusyBox wget resolves localhost to the IPv6 [::1]
# and nginx here listens on IPv4 only, so the hostname form reports the
# container unhealthy while it is in fact serving correctly.
HEALTHCHECK --interval=15s --timeout=4s --start-period=8s --retries=5 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
CMD ["nginx", "-g", "daemon off;"]
