pipeline {
    agent any

    environment {
        IMAGE = "alpine:latest"
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
                    python3 --version || apt-get install -y python3 python3-pip
                    pip3 install -q langchain langchain-ollama langgraph requests 2>/dev/null || true
                '''
            }
        }

        stage('IntelliOps Agent Pipeline') {
            steps {
                echo '🤖 Running all 5 AI agents...'
                sh '''
                    pip3 install -q langchain langchain-ollama langgraph requests
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
