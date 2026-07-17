# Dynamic Pricing ML Engine

## Overview
This project is an end-to-end Machine Learning web application that predicts the optimal price for a product to maximize total revenue. It uses a custom-trained machine learning model wrapped in a high-speed REST API, complete with a clean, interactive frontend dashboard.

## Key Engineering Achievements
* **Engineered** an end-to-end dynamic pricing machine learning system using Scikit-Learn to predict optimal product pricing and maximize projected revenue.
* **Developed** a high-performance REST API backend using Python and FastAPI to serve the pre-trained model, enabling instant, real-time data inference.
* **Designed and Deployed** a responsive, interactive frontend dashboard using HTML, CSS (Glassmorphism), and JavaScript to collect user inputs and visualize calculations asynchronously.
* **Optimized** the system architecture by decoupling model training from the web server, serializing the model to drastically reduce API load times and computational overhead.

## Tech Stack
* **Machine Learning:** Python, Scikit-Learn, Pandas, NumPy
* **Backend API:** FastAPI, Uvicorn
* **Frontend:** HTML, CSS, JavaScript

## How to Run Locally
1. Clone this repository.
2. Open your terminal and start the backend server: `python -m uvicorn api:app --reload`
3. Open the `index.html` file in any standard web browser to view the dashboard.
