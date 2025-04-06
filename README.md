# UniBot - University Information Chatbot

UniBot is an intelligent chatbot designed to provide information about university facilities, admission processes, and student services. It uses advanced natural language processing techniques to understand and respond to user queries about university-related topics.

## Features

- Natural Language Understanding for university-related queries
- Semantic search for relevant information
- Sentiment analysis of user queries and responses
- Comprehensive evaluation metrics
- Real-time response generation
- Support for multiple types of queries (facilities, admissions, student life, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/monkeybio5414/unibot.git
cd unibot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python src/main.py
```

2. The server will be available at `http://127.0.0.1:8001`

3. Send queries using the chat endpoint:
```bash
curl -X POST http://127.0.0.1:8001/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "What facilities are available at the university?"}'
```

## Project Structure

```
unibot/
├── src/
│   ├── models/
│   │   ├── nlg.py           # Natural Language Generation model
│   │   ├── evaluate.py      # Model evaluation metrics
│   │   └── evaluate_model.py # Model evaluation script
│   ├── utils/
│   │   └── logger.py        # Logging utilities
│   └── main.py              # Main server application
├── logs/                    # Application logs
├── models/                  # Pretrained models
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Evaluation

The project includes comprehensive evaluation tools to assess the model's performance:

- Response time metrics
- Confidence scores
- BERT similarity scores
- Context relevance scores
- Sentiment analysis scores

Run the evaluation script:
```bash
python src/models/evaluate_model.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 