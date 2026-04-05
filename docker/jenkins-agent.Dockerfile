FROM jenkins/inbound-agent:latest

USER root

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Install Python packages
RUN pip3 install --break-system-packages \
    langchain \
    langchain-ollama \
    langgraph \
    requests

USER jenkins
