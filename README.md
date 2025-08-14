# 🍳 Recipe Recommendation Engine

A sophisticated content-based recipe recommendation system built with FastAPI and machine learning. This RESTful API provides intelligent recipe suggestions based on user preferences, cuisine types, and dietary restrictions, achieving high precision scores through advanced content filtering algorithms.

## ✨ Features

- **Content-Based Filtering**: Advanced recommendation algorithm using TF-IDF vectorization and cosine similarity
- **RESTful API**: Clean, documented API endpoints for easy integration
- **Smart Filtering**: Filter by cuisine, prep time, cook time, and difficulty level
- **High Performance**: Optimized for speed with 10,000+ recipe database
- **Docker Support**: Fully containerized application for easy deployment
- **Model Evaluation**: Built-in performance metrics and evaluation tools
- **Scalable Architecture**: Designed for production use and easy scaling

## 🛠️ Technologies Used

- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **API Documentation**: Automatic OpenAPI/Swagger docs
- **Containerization**: Docker
- **Server**: Uvicorn (ASGI)

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/recipe-recommendation-engine.git
   cd recipe-recommendation-engine
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t recipe-recommendation-engine .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 recipe-recommendation-engine
   ```

3. **Access the API**
   Navigate to http://localhost:8000

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and available endpoints |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/recipes` | Get all recipes (with optional limit) |
| `GET` | `/recipes/{id}` | Get specific recipe by ID |
| `POST` | `/recommend` | Get personalized recipe recommendations |
| `GET` | `/evaluate` | Evaluate model performance metrics |
| `GET` | `/cuisines` | Get available cuisine types |
| `GET` | `/difficulties` | Get available difficulty levels |

### Recommendation Request Format

```json
{
  "user_preferences": ["chicken", "spicy", "quick"],
  "cuisine_preference": "Indian",
  "max_prep_time": 30,
  "max_cook_time": 45,
  "num_recommendations": 5
}
```

### Response Format

```json
[
  {
    "recipe": {
      "id": 1,
      "title": "Chicken Tikka Masala",
      "ingredients": ["chicken breast", "yogurt", "spices"],
      "instructions": ["Marinate chicken...", "Grill chicken..."],
      "cuisine": "Indian",
      "prep_time": 20,
      "cook_time": 30,
      "servings": 6,
      "difficulty": "Medium",
      "tags": ["chicken", "curry", "spicy", "creamy"]
    },
    "similarity_score": 0.85
  }
]
```

## 🔬 Machine Learning Model

### Content-Based Filtering Algorithm

The recommendation system uses a sophisticated content-based filtering approach:

1. **Feature Extraction**: TF-IDF vectorization of recipe text features
2. **Text Processing**: Combines title, cuisine, difficulty, tags, and ingredients
3. **Similarity Calculation**: Cosine similarity between user preferences and recipes
4. **Ranking**: Top-N recommendations based on similarity scores

### Model Performance

The system achieves high precision through:
- **TF-IDF Vectorization**: Captures ingredient and flavor importance
- **Multi-dimensional Features**: Considers cuisine, difficulty, and preparation time
- **Cosine Similarity**: Measures semantic similarity between preferences and recipes
- **Filtering Constraints**: Allows users to specify dietary and time constraints

### Performance Metrics

- **Target Precision**: ≥ 0.85
- **Evaluation Method**: Synthetic user preference testing
- **Dataset Size**: 10,000+ recipes with variations
- **Response Time**: < 100ms for recommendations

## 🗄️ Data Structure

### Recipe Schema

```python
class Recipe:
    id: int                    # Unique identifier
    title: str                 # Recipe name
    ingredients: List[str]     # List of ingredients
    instructions: List[str]    # Step-by-step instructions
    cuisine: str               # Cuisine type
    prep_time: int             # Preparation time in minutes
    cook_time: int             # Cooking time in minutes
    servings: int              # Number of servings
    difficulty: str            # Easy/Medium/Hard
    tags: List[str]           # Descriptive tags
```

### Sample Cuisines

- Italian, Indian, American, Mexican, Thai
- Greek, Japanese, French, Chinese, Mediterranean

### Difficulty Levels

- **Easy**: Quick recipes with simple techniques
- **Medium**: Moderate complexity and time investment
- **Hard**: Advanced techniques and longer preparation

## 🚀 Deployment Options

### Local Development
```bash
python main.py
```

### Docker
```bash
docker build -t recipe-engine .
docker run -p 8000:8000 recipe-engine
```

### Production Deployment

1. **Cloud Platforms**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances

2. **Kubernetes**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: recipe-recommendation-engine
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: recipe-engine
   ```

3. **Load Balancing**
   - Use nginx or HAProxy for load distribution
   - Implement health checks and auto-scaling

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Customize server settings
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

### Performance Tuning

- **TF-IDF Features**: Adjust `max_features` in vectorizer
- **Similarity Matrix**: Pre-compute for faster responses
- **Caching**: Implement Redis for session storage
- **Database**: Replace in-memory storage with PostgreSQL

## 📊 Monitoring and Health

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "message": "Recipe Recommendation Engine is running"
}
```

### Model Evaluation

```bash
curl http://localhost:8000/evaluate
```

Response:
```json
{
  "model_performance": {
    "precision": 0.87,
    "recall": 0.82,
    "f1_score": 0.84,
    "test_users": 5
  },
  "target_precision": 0.85,
  "achieved_target": true
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public methods
- Write tests for new features
- Update API documentation

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### API Testing
```bash
# Test recommendation endpoint
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_preferences": ["chicken", "spicy"], "num_recommendations": 3}'
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Scikit-learn for machine learning capabilities
- The open-source community for inspiration and tools

## 📞 Support

- **Documentation**: Check the interactive API docs at `/docs`
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions for help and ideas

---

**Built with ❤️ for food lovers and developers**
# Update API documentation
