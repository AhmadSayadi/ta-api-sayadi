#!/usr/bin/env python3
"""
Deployment script for TA API Sayadi
Supports deployment to Render and Azure
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, check=True):
    """Run shell command and return result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")
    
    return result

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print("Checking prerequisites...")
    
    # Check if git is installed
    try:
        run_command("git --version")
    except:
        sys.exit("Git is not installed or not in PATH")
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        sys.exit("Not in a git repository. Please initialize git first.")
    
    # Check if required files exist
    required_files = ["main.py", "requirements.txt", "model/svm_haz_pmk.pkl", "model/label_encoder.pkl"]
    for file in required_files:
        if not Path(file).exists():
            sys.exit(f"Required file not found: {file}")
    
    print("✅ All prerequisites met")

def deploy_to_render():
    """Deploy to Render"""
    print("\n🚀 Deploying to Render...")
    
    # Check if render.yaml exists
    if not Path("render.yaml").exists():
        sys.exit("render.yaml not found")
    
    # Push to git (Render auto-deploys from git)
    print("Pushing to git repository...")
    run_command("git add .")
    run_command('git commit -m "Deploy to Render"', check=False)
    run_command("git push origin main", check=False)
    run_command("git push origin master", check=False)
    
    print("✅ Code pushed to repository")
    print("📝 Render will automatically deploy from your git repository")
    print("🔗 Check your Render dashboard for deployment status")

def deploy_to_azure_webapp():
    """Deploy to Azure Web App"""
    print("\n☁️ Deploying to Azure Web App...")
    
    # Check if Azure CLI is installed
    try:
        run_command("az --version")
    except:
        sys.exit("Azure CLI is not installed. Please install it first.")
    
    # Check if logged in to Azure
    result = run_command("az account show", check=False)
    if result.returncode != 0:
        print("Please login to Azure first: az login")
        return
    
    # Get configuration
    app_name = input("Enter Azure Web App name (default: ta-api-sayadi): ") or "ta-api-sayadi"
    resource_group = input("Enter resource group name (default: rg-ta-api-sayadi): ") or "rg-ta-api-sayadi"
    location = input("Enter location (default: Southeast Asia): ") or "Southeast Asia"
    
    # Create resource group if it doesn't exist
    print(f"Creating resource group {resource_group}...")
    run_command(f'az group create --name {resource_group} --location "{location}"', check=False)
    
    # Create App Service Plan
    plan_name = f"plan-{app_name}"
    print(f"Creating App Service Plan {plan_name}...")
    run_command(f'az appservice plan create --name {plan_name} --resource-group {resource_group} --sku B1 --is-linux', check=False)
    
    # Create Web App
    print(f"Creating Web App {app_name}...")
    run_command(f'az webapp create --resource-group {resource_group} --plan {plan_name} --name {app_name} --runtime "PYTHON:3.13" --deployment-local-git')
    
    # Set startup command
    print("Setting startup command...")
    run_command(f'az webapp config set --resource-group {resource_group} --name {app_name} --startup-file "uvicorn main:app --host 0.0.0.0 --port 8000"')
    
    # Set environment variables
    print("Setting environment variables...")
    api_key = input("Enter API key: ")
    run_command(f'az webapp config appsettings set --resource-group {resource_group} --name {app_name} --settings MODEL_DIR=model PYTHONPATH=. API_KEY={api_key}')
    
    # Get git URL and deploy
    print("Getting deployment URL...")
    result = run_command(f'az webapp deployment source config-local-git --name {app_name} --resource-group {resource_group}')
    
    print("\n📝 To complete deployment, run:")
    print(f"git remote add azure <git-url-from-above>")
    print(f"git push azure main")
    
    print(f"\n✅ Azure Web App created: https://{app_name}.azurewebsites.net")

def deploy_to_azure_container():
    """Deploy to Azure Container Instances"""
    print("\n🐳 Deploying to Azure Container Instances...")
    
    # Check if Docker is installed
    try:
        run_command("docker --version")
    except:
        sys.exit("Docker is not installed")
    
    # Build Docker image
    print("Building Docker image...")
    run_command("docker build -t ta-api-sayadi .")
    
    # Get configuration
    registry_name = input("Enter Azure Container Registry name: ")
    resource_group = input("Enter resource group name (default: rg-ta-api-sayadi): ") or "rg-ta-api-sayadi"
    container_name = input("Enter container name (default: ta-api-sayadi): ") or "ta-api-sayadi"
    
    # Tag and push to ACR
    image_name = f"{registry_name}.azurecr.io/ta-api-sayadi:latest"
    print(f"Tagging image as {image_name}...")
    run_command(f"docker tag ta-api-sayadi {image_name}")
    
    print("Pushing to Azure Container Registry...")
    run_command(f"az acr login --name {registry_name}")
    run_command(f"docker push {image_name}")
    
    # Deploy to ACI
    print("Deploying to Azure Container Instances...")
    api_key = input("Enter API key: ")
    run_command(f'az container create --resource-group {resource_group} --name {container_name} --image {image_name} --cpu 1 --memory 1 --ports 8000 --environment-variables MODEL_DIR=model PYTHONPATH=. API_KEY={api_key} --ip-address public')
    
    print(f"✅ Container deployed successfully")

def main():
    parser = argparse.ArgumentParser(description="Deploy TA API Sayadi")
    parser.add_argument("platform", choices=["render", "azure-webapp", "azure-container"], 
                       help="Deployment platform")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip prerequisite checks")
    
    args = parser.parse_args()
    
    if not args.skip_checks:
        check_prerequisites()
    
    if args.platform == "render":
        deploy_to_render()
    elif args.platform == "azure-webapp":
        deploy_to_azure_webapp()
    elif args.platform == "azure-container":
        deploy_to_azure_container()

if __name__ == "__main__":
    main()