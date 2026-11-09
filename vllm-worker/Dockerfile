
FROM vllm/vllm-openai:v0.27.1

USER root

# Install dependencies for AWS CLI.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (Linux AMD64).
RUN curl -fsSL \
    "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o /tmp/awscliv2.zip && \
    cd /tmp && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip

# Copy the generic FlashLLM startup script.
COPY --chmod=755 start.sh /app/start.sh

# Start the worker using Bash.
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
