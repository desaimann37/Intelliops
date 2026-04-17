pipeline {
    agent {
    kubernetes {
        yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.11-slim
    command:
    - sleep
    args:
    - infinity
"""
        defaultContainer 'python'
    }
}

    environment {
        IMAGE = "alpine:latest"
        ARGOCD_SERVER = "argocd-server.argocd.svc.cluster.local"
        ARGOCD_APP = "nginx-app"
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
            python3 agents/orchestrator.py 2>&1 | tee /tmp/agent_result.txt
            if grep -q "INTELLIOPS PIPELINE DECISION: APPROVED" /tmp/agent_result.txt; then
                echo "APPROVED" > /tmp/decision.txt
            else
                echo "REJECTED" > /tmp/decision.txt
            fi
        '''
    }
}
        

        stage('Deploy via ArgoCD') {
            when {
                expression {
                    return sh(
                        script: 'cat /tmp/decision.txt',
                        returnStdout: true
                    ).trim() == 'APPROVED'
                }
            }
            steps {
                echo '🚀 Triggering ArgoCD sync...'
                sh '''
                    # Login to ArgoCD
                    argocd login ${ARGOCD_SERVER} \
                        --username admin \
                        --password $(kubectl get secret argocd-initial-admin-secret \
                            -n argocd \
                            -o jsonpath="{.data.password}" | base64 -d) \
                        --insecure \
                        --grpc-web

                    # Sync the application
                    argocd app sync ${ARGOCD_APP} --grpc-web

                    # Wait for healthy status
                    argocd app wait ${ARGOCD_APP} \
                        --health \
                        --timeout 120 \
                        --grpc-web

                    echo "✅ ArgoCD sync complete - ${ARGOCD_APP} is healthy"
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
