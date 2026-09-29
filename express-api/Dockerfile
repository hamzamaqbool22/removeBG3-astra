FROM node:22-bookworm-slim
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8002 DATA_DIR=/app/data
COPY package*.json ./
RUN npm ci --omit=dev
COPY src ./src
COPY assets ./assets
COPY scripts ./scripts
COPY licenses ./licenses
RUN mkdir -p /app/data && chown -R node:node /app
USER node
EXPOSE 8002
CMD ["node", "src/server.js"]
