pipeline {
    agent any

    environment {
        IMAGE = "alpine:latest"
        OLLAMA_HOST = "http://host-gateway:11434"
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📥 Cloning repository...'
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                echo '🔧 Setting up Python environment...'
                sh '''
                    python3 --version
                    pip3 install --break-system-packages -q \
                        langchain \
                        langchain-ollama \
                        langgraph \
                        requests
                    echo "✅ Dependencies installed"
                '''
            }
        }

        stage('IntelliOps Agent Pipeline') {
            steps {
                echo '🤖 Running all 5 AI agents...'
                sh '''
                    python3 agents/orchestrator.py
                '''
            }
        }

        stage('Cost Optimization Check') {
            steps {
                echo '💰 Running Cost Optimizer Agent...'
                sh '''
                    python3 agents/cost_optimizer_agent.py
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline APPROVED - Deployment successful!'
        }
        failure {
            echo '🚫 Pipeline REJECTED - Deployment blocked!'
        }
        always {
            echo '📊 Pipeline complete. Check Grafana for metrics.'
        }
    }
}
