# Document Q&A Application

This is a Streamlit-based web application that allows users to upload documents (PDF, DOCX, XLSX) and ask questions about their content. The app uses natural language processing to understand and respond to user queries based on the uploaded documents.

## Features

- Document upload support for PDF, DOCX, and XLSX files
- Natural language question answering based on document content
- Conversation history tracking
- Source attribution for answers

## Prerequisites

- Python 3.12.4
- Miniconda 

## Environment Setup

This project uses a Conda environment. To set up the environment:

1. Ensure you have Miniconda, VS code and Git installed.
2. Create the environment using the provided `environment.yml` file:
    # Navigate to your project directory:
        cd C:\document_chat\Test 1
    # Create the Environment 
        conda env create -f environment.yml
        conda activate document_chat
        conda list
    # Update the Environment
        conda env update --file environment.yml --prune
3. Connect your local files to your GitHub repository
    # Navigate to your project directory
        cd C:\document_chat\Test 1
    # Initialize a Git repository in your local project folder
        git init
    # Add your files to the Git repository
        git add . 
    # Commit the files
        git commit -m "Initial commit"
    # Link your local repository to the GitHub repository
        git remote add origin https://github.com/kodwouj/document_chat.git
    # Push your code to GitHub
        git push -u origin main