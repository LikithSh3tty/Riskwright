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
HEALTHCHECK --interval=15s --timeout=4s --start-period=8s --retries=5 \
  CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
CMD ["nginx", "-g", "daemon off;"]
