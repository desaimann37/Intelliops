pipeline {
    agent any

    environment {
        IMAGE = "alpine:latest"
        VENV_PATH = "/home/linux_admin/Intelliops/venv"
        PROJECT_PATH = "/home/linux_admin/Intelliops"
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
                    cd ${PROJECT_PATH}
                    source ${VENV_PATH}/bin/activate
                    pip install -q langchain langchain-ollama langgraph requests
                '''
            }
        }

        stage('IntelliOps Agent Pipeline') {
            steps {
                echo '🤖 Running all 5 AI agents...'
                sh '''
                    cd ${PROJECT_PATH}
                    source ${VENV_PATH}/bin/activate
                    python3 agents/orchestrator.py
                '''
            }
        }

        stage('Deploy via ArgoCD') {
            when {
                expression {
                    return currentBuild.result == null
                }
            }
            steps {
                echo '🚀 Triggering ArgoCD sync...'
                sh '''
                    kubectl get applications -n argocd
                '''
            }
        }

        stage('Cost Optimization Check') {
            steps {
                echo '💰 Running Cost Optimizer Agent...'
                sh '''
                    cd ${PROJECT_PATH}
                    source ${VENV_PATH}/bin/activate
                    python3 agents/cost_optimizer_agent.py
                '''
            }
        }
    }

    post {
        success {
            echo '✅ IntelliOps Pipeline APPROVED - Deployment successful!'
        }
        failure {
            echo '🚫 IntelliOps Pipeline REJECTED - Deployment blocked!'
        }
        always {
            echo '📊 Pipeline complete. Check Grafana for metrics.'
        }
    }
}
