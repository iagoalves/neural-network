FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first so the layer is reusable when only source files change.
COPY package*.json ./

RUN npm ci --no-audit --no-fund

# Public Vite variables must be baked into the build output.
ARG VITE_API_BASE_URL=http://localhost:8787
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

COPY . .

RUN npm run build

FROM nginx:alpine AS runtime

COPY --from=build /app/dist/ /usr/share/nginx/html/
RUN rm -f /usr/share/nginx/html/50x.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
