# claudegate needs both runtimes: Python for the server, Node for the CLI it drives.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_MAJOR=22 \
    CLAUDEGATE_HOST=0.0.0.0 \
    CLAUDEGATE_PORT=8080

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y gnupg && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# The CLI refuses permission bypass as root and exits silently without this.
# Running as a non-root user is the better answer where you can.
ENV IS_SANDBOX=1

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5).status==200 else 1)"

ENTRYPOINT ["claudegate"]
CMD ["serve"]
