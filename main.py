import streamlit as st
import random
import time
from streamlit_tags import st_tags
from typing import List
import requests

# Simulated quiz data

def main():
    # Session State Initialization
    ss = st.session_state
    if 'counter' not in ss:
        ss['counter'] = 0
    if 'start' not in ss:
        ss['start'] = False
    if 'stop' not in ss:
        ss['stop'] = False
    if 'refresh' not in ss:
        ss['refresh'] = False
    if 'current_quiz' not in ss:
        ss['current_quiz'] = []
    if 'user_answers' not in ss:
        ss['user_answers'] = []
    if 'grade' not in ss:
        ss['grade'] = 0
    if 'button_label' not in ss:
        ss['button_label'] = ['START', 'SUBMIT', 'RELOAD']

    # Function to handle button clicks and page refreshes
    def btn_click():
        ss.counter += 1
        if ss.counter > 2:
            ss.counter = 0
            ss.clear()  # Reset session state
        else:
            update_session_state()
            with st.spinner("Loading quiz..."):
                time.sleep(2)
    

    def search_questions(num: int, select_topics: List[str], select_levels: List[str] ):
        # Search for questions based on selected topics and levels
        
        endpoint = 'http://127.0.0.1:8000/questions/search'
        params = {
            'num': str(num)
        }

        # Define the headers
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        # Define the data to be sent in the body (as JSON)
        data = {
            "query_topics": {f"{i}": t for i,t in enumerate(select_topics)},
            "query_level": {f"{i}": l for i,l in enumerate(select_levels)}
        }

        # Send a POST request
        logging.debug(f"Sending POST request to {endpoint}")
        res = requests.post(url = endpoint, 
                            headers = headers, 
                            params = params, 
                            json = data)
        logging.debug(f"Status: {res.status_code}")

        # Adding questions retrieved from the API to the quiz_data
        questions = res.json()['questions']
        quiz_data = []
        for q in questions:
          if 'entity' in q:
              
              q_args= q['entity']
              option_list = [q_args['opt_1'], q_args['opt_2'], q_args['opt_3'], q_args['opt_4']]
              correct_option = q_args['correct_opt']
              # Check correct options
              if correct_option in ['opt_1', 'opt_2', 'opt_3', 'opt_4']:
                  correct_option = q_args[q_args['correct_opt']]
              elif correct_option in option_list:
                  correct_option = correct_option
              question = {
                  "question": q_args['question'],
                  "code" : q_args['code'],
                  "options": option_list,
                  "explanation": q_args['explanation'],
                  "correct_answer": correct_option
              }
              quiz_data.append(question)
          elif 'entity' not in q:
              option_list_enc = [q['opt_1'], q['opt_2'], q['opt_3'], q['opt_4']]
              correct_option = q['correct_opt']
              if correct_option in ['opt_1', 'opt_2', 'opt_3', 'opt_4']:
                  correct_option = q[q['correct_opt']]
              elif correct_option in option_list_enc:
                  correct_option = correct_option

              question = {
                  "question": q['question'],
                  "code" : q['code'],
                  "options": option_list_enc,
                  "explanation": q['explanation'],
                  "correct_answer": correct_option
              }
              
              quiz_data.append(question)
        return quiz_data

    # Function to update session state
    def update_session_state():
        if ss.counter == 1:
            ss['start'] = True
            ss['current_quiz'] = search_questions(num_questions, topics, levels)  # Adjust number of questions
        elif ss.counter == 2:
            ss['start'] = True
            ss['stop'] = True
        elif ss.counter == 3:
            ss['start'] = ss['stop'] = False
            ss['refresh'] = True
            ss.clear()  # Trigger a fresh start by clearing the session state

    # Function to display the quiz app
    def quiz_app():
        scorecard_placeholder = st.empty()
        if ss['start']:
            for i, question in enumerate(ss['current_quiz']):
                # Add spacing and labels for each question
                st.markdown(f"### **Question {i + 1}:** {question['question']}")
                
                # Display the Python code block if the question contains code
                if question['code'] != "":
                    st.code(question['code'], language='python')  # Display code only if it exists

                # Create two columns: One for options and one for explanation
                question_col, options_col = st.columns([3, 1])

                with question_col:
                    # Display options as radio buttons (no pre-selection)
                    selected_option = st.radio("", question['options'], index=None, key=f"Q{i + 1}")

                with options_col:
                    pass  

                # After the quiz is stopped, show results
                if ss['stop']:
                    if selected_option == question['correct_answer']:
                        ss['user_answers'].append(True)
                    else:
                        ss['user_answers'].append(False)

                    # Results feedback (display correct answer)
                    if ss['user_answers'][i]:
                        st.success(f"Correct Answer: {question['correct_answer']}")
                    else:
                        st.error(f"Incorrect! The correct answer was: {question['correct_answer']}")

                    # Display explanation
                    st.write(f"**Explanation:** {question['explanation']}")

            # Display final score after submission
            if ss['stop']:
                ss['grade'] = ss['user_answers'].count(True)
                scorecard_placeholder.write(f"### **Your Final Score: {ss['grade']} / {len(ss['current_quiz'])}**")

    # Search Page (First Page) - Allows user to input number of questions
    st.write("# Quiz Search")
    num_questions = st.number_input("Enter the number of questions you want:", min_value=1, value=3, step=1)  # Number of questions wanted to quiz

    # Tag inputs for topics
    topics = st_tags(
        label="Select at least one Topics for the Quiz:",
        value=['Python', 'Pandas'],
        suggestions=['Python', 'Pandas DataFrame', 'Python Variable', 'Numpy Array', 'File Handling', 'Exceptions', 'Python Dictionary'],
        maxtags=15,
        key="topics_input"
    )

    # Tag inputs for levels
    levels = st_tags(
        label="Select at least one Difficulty Level:",
        value=['easy'],
        suggestions=['easy', 'medium', 'hard'],
        maxtags=3,
        key="levels_input"
    )

    # Display the number of questions, topics, and levels selected by the user
    st.write("### Number of Questions:")
    st.write(f"You selected: {num_questions} questions")

    st.write("### Selected Topics:")
    st.write(topics)

    st.write("### Selected Difficulty Levels:")
    st.write(levels)

    # Logging (this would display in the terminal if you're running Streamlit locally)
    import logging

    logging.basicConfig(level=logging.DEBUG)

    logging.debug(f"Number of Questions: {num_questions}")
    logging.debug(f"Selected Topics: {topics}")
    logging.debug(f"Selected Levels: {levels}")

    # Optional: Add a code block showing how the user input might be used
    st.code('''# Example Code Snippet

def get_questions(num_questions, topics, levels):
    # Here you would call your backend API to fetch questions
    print(f"Fetching {num_questions} questions on {topics} at {levels} level")

# Example function call
get_questions(5, ['Python', 'AI'], ['Easy'])
''', language="python")

    # Start button logic
    st.button(label=ss['button_label'][ss['counter']], key='button_press', on_click=btn_click)

    # Display quiz questions
    quiz_app()

# Running the app
if __name__ == "__main__":
    main()
