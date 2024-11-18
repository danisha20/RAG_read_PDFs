# RAG Chatbot with Streamlit

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** to provide intelligent and context-aware responses by leveraging a combination of document retrieval and generative AI.

---

## Features

- **User-Friendly Interface**: Clean and responsive chat interface with user and bot avatars.
- **Real-Time Interactions**: Instant responses powered by the RAG architecture.
- **Customizable**: Easily update styles and functionality to fit your needs.
- **Streamlit-Powered**: Fully integrated into a lightweight Streamlit app.
- **Expander for Detailed Logs**: Scrollable log section for tracking conversations.

---

## How It Works

1. **Input Query**: Users can enter their queries through a chat-like interface.
2. **RAG Workflow**:
   - Retrieves relevant documents based on the query.
   - Uses a generative model to formulate a contextual response.
3. **Display Response**: The chatbot replies with well-formed and contextually relevant answers.
4. **Conversation Logs**: Scrollable logs are available in an expandable section for review.

---

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   pip install -r requirements.txt
   streamlit run app.py
