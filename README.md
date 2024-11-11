
# Quiz App

A simple and interactive quiz application built with Python. This project includes a backend API for managing questions and a Streamlit frontend for user interaction.

![ui](https://raw.githubusercontent.com/PeterLOVANAS/MCQ-quiz-app/refs/heads/main/images/backend_4.png)
*Figure 1: The main UI of search section the Quiz App.*

![ui2](https://raw.githubusercontent.com/PeterLOVANAS/MCQ-quiz-app/refs/heads/main/images/backend_3.png)
*Figure 2: The main UI of quiz section the Quiz App. This screen happened after the users submit their answer.*


---

## Directory Structure

Organize your project directory as follows:

```plaintext
quiz_app/
├── main.py         # Main streamlit script (Frontend)
├── backend.py		# Main API script (Backend)
├── agent.py		# LLM Agent script
├── your_db.db		# The database contains set of questions
├── template.txt	# Prompt template file
├── topics.txt 		# Default topics list to generated file
├── config.yaml		# app config
├── .streamlit/           # Streamlit theme config
│   ├── config.toml       
├── requirements.txt    # Required Python libraries
├── README.md           # Project documentation
```

---

## Libraries Needed

Ensure you have the following libraries installed. Use the provided `requirements.txt` to install them quickly.

### Required Libraries:
- **FastAPI**: For building the backend API.
- **Uvicorn**: To serve the FastAPI app.
- **Streamlit**: For building the frontend app.
- **Requests**: To handle HTTP requests between the frontend and backend.
- **Pydantic**: For data validation in FastAPI.
- **OpenAI**: For OpenAI API requests (LLM + Embedding model)
- **pymilvus**: For database manipulation (add, search, query, upsert, delete)
	- **pymilvus-model**: To use Embedding model and Reranker.

### Install Libraries:
```python
pip install -r requirements.txt
```

---

## Running the API Script

The backend API is responsible for handling questions-related operations such as adding and searching questions.

### Steps:
1. Run the API script using `uvicorn` (*`app` is the FastAPI instance defined in `backend.py`*):
    ```python
    uvicorn backend:app --reload
    ```

2. The API will be available at `http://127.0.0.1:8000`.

3. Test the API using tools like:
   - [Postman](https://www.postman.com/)
   - cURL
   - Directly via the auto-generated API docs and UI at `http://127.0.0.1:8000/docs`.

4. For UI users, access the interactive API documentation at:
   - `http://127.0.0.1:8000/docs` to try the following endpoints:
     - `/question/add`
     - `/question/update`
     - `/question/delete`

5. If you want to use `requests` in Python to interact with the API, here is an example:

```python
import requests
import logging

# Endpoint URL
endpoint = 'http://127.0.0.1:8000/questions/search'

# Parameters
params = {
    'num': '2'  # Number of questions to fetch
}

# Define the headers
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Define the data to be sent in the body (as JSON)
select_topics = ["Python", "DataFrame"]
select_levels = ["easy", "medium"]
data = {
    "query_topics": {f"{i}": t for i, t in enumerate(select_topics)},
    "query_level": {f"{i}": l for i, l in enumerate(select_levels)}
}

# Send a POST request
logging.debug(f"Sending POST request to {endpoint}")
res = requests.post(url=endpoint, headers=headers, params=params, json=data)

# Log response status
logging.debug(f"Status: {res.status_code}")

# Retrieve questions from the response
questions = res.json().get('questions', [])
print(questions)
```

---

## Running the Streamlit App

The frontend application, built with Streamlit, interacts with the backend to display questions and allow user interaction.

### Steps:
1. Run the Streamlit script:
    ```python
    streamlit run main.py
    ```

2. The Streamlit app will open in your default web browser. Alternatively, you can access it at `http://localhost:8501`.

---

## Example Usage

### Starting the Application:
1. Start the backend API:
    ```python
    uvicorn backend:app --reload
    ```
2. Start the Streamlit app:
    ```python
    streamlit run main.py
    ```

### Interacting with the App:
- Use the frontend to answer quizzes, add new questions, or browse through existing ones.
- The backend handles all API requests, such as searching and retrieving questions or adding and generating new questions.

---

## Notes

- Ensure the backend API is running before starting the Streamlit app.
- Customize the `template.txt`, `topics.txt`, and `your_db.db` files for specific requirements, such as default questions, topics, and database initialization (if wanted).
