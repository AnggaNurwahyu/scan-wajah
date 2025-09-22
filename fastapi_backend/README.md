# Face Recognition FastAPI Backend

This directory contains the Python FastAPI backend for the face recognition feature of the Flutter application.

## Prerequisites

**This project requires Python 3.10.**

Due to a dependency on the `mediapipe` library, this project is not compatible with newer versions of Python (3.11+) on Windows at this time. It is strongly recommended to use a dedicated virtual environment with Python 3.10 to run this server.

## Setup and Running the Server (for Windows)

These instructions will guide you through setting up a clean, isolated Python 3.10 environment for this project.

### Step 1: Install Python 3.10

If you don't have Python 3.10, you'll need to install it.

1.  Go to the official Python website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2.  Download a Python 3.10 installer (e.g., the "Windows installer (64-bit)").
3.  Run the installer. **Important:** On the first screen of the installer, make sure to check the box that says **"Add Python 3.10 to PATH"**.

### Step 2: Create a New Project Environment

Once Python 3.10 is installed, create a virtual environment.

1.  Open a Command Prompt or PowerShell window.
2.  Navigate to the root of the `scan-wajah-main` project folder.
3.  Create a virtual environment using Python 3.10 by running this command:
    ```bash
    py -3.10 -m venv .venv
    ```
    This will create a new folder named `.venv` in your project directory.

### Step 3: Activate the Environment and Install Packages

1.  Activate the new environment. In the same terminal, run:
    ```bash
    .venv\Scripts\activate
    ```
    You should see `(.venv)` appear at the beginning of your command prompt line. This tells you the environment is active.
2.  Now, navigate into this backend folder:
    ```bash
    cd fastapi_backend
    ```
3.  Install all the required packages into this new, clean environment:
    ```bash
    pip install -r requirements.txt
    ```

### Step 4: Run the Server

1.  With the virtual environment still active (you see `(.venv)` in the prompt) and while you are inside the `fastapi_backend` directory, run the server:
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 5000
    ```

The server is now running and accessible at `http://localhost:5000`. Remember to update the URL in the Flutter application to point to this address.
