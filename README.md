Note: This is not a cloned web application this web application contains some inspiration from other creators so as a sign to thank them added thier name as collaborators 

# Local Multimodal AI Chat
## Getting Started

To get started with Local Multimodal AI Chat, follow these simple steps:

### Prerequisites

1. **Python**: Python 3.8 or higher is required
2. **Gemini API**
### Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Leon-Sander/Local-Multimodal-AI-Chat.git
   cd Local-Multimodal-AI-Chat
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```
   This will:
   - Install all required dependencies
   - Check if Ollama is running
   - Initialize the database

3. **Start the application**:
   ```bash
   streamlit run app.py
   ```

4. **Pull Required Models**: 
   - Open [localhost:8501](http://localhost:8501) in your browser
   - Go to https://ollama.com/library and choose the models you want to use
   - Enter `/pull MODEL_NAME` in the chat bar
   - You need:
     - One embedding model (e.g. [nomic-embed-text](https://ollama.com/library/nomic-embed-text)) for PDF files
     - One model that understands images (e.g. [llava](https://ollama.com/library/llava))

5. **Optional Configuration**: 
   - Check the `config.yaml` file and adjust settings to your needs
   - Place your custom `user_image.png` and/or `bot_image.png` inside the `chat_icons` fold
