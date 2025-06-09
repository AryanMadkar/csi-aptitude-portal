from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import json
import logging
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise ValueError("GROQ_API_KEY environment variable is required")

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

# Enhanced prompt templates
PROMPT_TEMPLATES = {
    "academic": """
Generate {num_questions} academically rigorous multiple-choice questions about "{topic}" at {difficulty} level.

Requirements:
- Questions must test deep understanding, not just memorization
- Include application, analysis, and synthesis level questions
- Options should be plausible and challenging
- Avoid obvious incorrect answers
- Include explanations with references to key concepts
- Specify Bloom's taxonomy level for each question

Difficulty Guidelines:
- Easy: Basic concepts and definitions
- Medium: Application and analysis
- Hard: Synthesis, evaluation, and complex problem-solving
- Expert: Advanced theoretical concepts and real-world applications

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "metadata": {{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "total_questions": {num_questions},
    "generation_time": "{timestamp}",
    "bloom_taxonomy_levels": ["remember", "understand", "apply", "analyze", "evaluate", "create"]
  }},
  "questions": [
    {{
      "id": 1,
      "question": "Clear, specific question text",
      "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_answer": "A",
      "explanation": "Detailed explanation of why this answer is correct and why other options are incorrect",
      "bloom_level": "analyze",
      "estimated_time_seconds": 45,
      "tags": ["concept1", "concept2"]
    }}
  ]
}}
""",
    
    "practical": """
Generate {num_questions} practical, scenario-based multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Real-world applications and case studies
- Problem-solving scenarios with realistic constraints
- Industry best practices and common pitfalls
- Hands-on implementation considerations

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "metadata": {{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "total_questions": {num_questions},
    "generation_time": "{timestamp}",
    "question_type": "practical"
  }},
  "questions": [
    {{
      "id": 1,
      "question": "Scenario-based question text",
      "scenario_context": "Brief description of the real-world scenario",
      "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_answer": "A",
      "explanation": "Detailed explanation with practical reasoning",
      "estimated_time_seconds": 60,
      "tags": ["practical", "scenario"]
    }}
  ]
}}
""",
    
    "conceptual": """
Generate {num_questions} conceptual multiple-choice questions about "{topic}" at {difficulty} level.

Focus on:
- Theoretical frameworks and models
- Relationships between core concepts
- Comparative analysis of different approaches
- Fundamental principles and their implications

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "metadata": {{
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "total_questions": {num_questions},
    "generation_time": "{timestamp}",
    "question_type": "conceptual"
  }},
  "questions": [
    {{
      "id": 1,
      "question": "Conceptual question text",
      "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_answer": "A",
      "explanation": "Detailed conceptual explanation",
      "estimated_time_seconds": 50,
      "tags": ["concept", "theory"]
    }}
  ]
}}
"""
}

# Pydantic model for request validation
class QuestionRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200, description="Subject matter")
    difficulty: str = Field("medium", description="Difficulty level")
    num_questions: int = Field(5, ge=1, le=50, description="Number of questions")
    question_type: str = Field("academic", description="Question type")
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v not in ["easy", "medium", "hard", "expert"]:
            raise ValueError("Difficulty must be one of: easy, medium, hard, expert")
        return v
    
    @validator('question_type')
    def validate_question_type(cls, v):
        if v not in ["academic", "practical", "conceptual"]:
            raise ValueError("Question type must be one of: academic, practical, conceptual")
        return v

# Enhanced JSON extraction function
def extract_json_from_response(content: str) -> Optional[Dict]:
    """Extract JSON from response with multiple fallback strategies"""
    try:
        # Try direct parsing first
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    
    # Try finding JSON within markdown code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'\{.*\}',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.groups() else match.group()
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    logger.error(f"Failed to extract JSON from response: {content[:200]}...")
    return None

# Core question generation function
def generate_questions(data: Dict) -> Dict:
    """Generate questions with enhanced error handling and validation"""
    try:
        # Validate input using Pydantic model
        request_data = QuestionRequest(**data)
        
        # Build prompt
        prompt = PROMPT_TEMPLATES[request_data.question_type].format(
            topic=request_data.topic,
            difficulty=request_data.difficulty,
            num_questions=request_data.num_questions,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Generating {request_data.num_questions} {request_data.difficulty} {request_data.question_type} questions about '{request_data.topic}'")
        
        # Generate questions with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert educational content creator. Generate high-quality multiple-choice questions. Always respond with valid JSON only, no markdown formatting or extra text."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = extract_json_from_response(content)
                
                if result and "questions" in result:
                    # Validate the structure
                    if len(result["questions"]) == request_data.num_questions:
                        logger.info(f"Successfully generated {len(result['questions'])} questions")
                        return result
                    else:
                        logger.warning(f"Generated {len(result['questions'])} questions instead of {request_data.num_questions}")
                
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    continue
                    
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                
        return {"error": "Failed to generate questions after multiple attempts"}
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"error": f"Invalid input: {str(e)}"}

# Simple conversation handler without LangChain
class ConversationHandler:
    def __init__(self):
        self.conversation_history = []
        logger.info("Conversation Handler initialized successfully")
    
    def process_message(self, user_input: str) -> str:
        """Process user message and generate appropriate response"""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "message": user_input, "timestamp": datetime.now().isoformat()})
            
            # Extract parameters from user input
            params = self._extract_parameters(user_input)
            
            if params.get("needs_clarification"):
                response = params["message"]
            else:
                # Generate questions
                result = generate_questions(params)
                
                if "error" in result:
                    response = f"I encountered an issue: {result['error']}. Could you please try again with more specific details?"
                else:
                    response = self._format_questions_response(result)
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "message": response, "timestamp": datetime.now().isoformat()})
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return "I'm having trouble processing your request. Could you please specify: topic, difficulty (easy/medium/hard/expert), number of questions, and question type (academic/practical/conceptual)?"
    
    def _extract_parameters(self, user_input: str) -> Dict:
        """Extract parameters from natural language input"""
        user_input_lower = user_input.lower()
        
        # Default parameters
        params = {
            "difficulty": "medium",
            "num_questions": 5,
            "question_type": "academic"
        }
        
        # Extract topic using multiple strategies
        topic = self._extract_topic(user_input, user_input_lower)
        
        if not topic or len(topic) < 2:
            return {
                "needs_clarification": True,
                "message": "I'd be happy to help you create questions! Could you please tell me what topic you'd like questions about? For example: 'Create 5 questions about Python programming' or 'Generate hard questions on machine learning'"
            }
        
        params["topic"] = topic.title()
        
        # Extract difficulty
        if "easy" in user_input_lower or "beginner" in user_input_lower:
            params["difficulty"] = "easy"
        elif "hard" in user_input_lower or "difficult" in user_input_lower or "challenging" in user_input_lower:
            params["difficulty"] = "hard"
        elif "expert" in user_input_lower or "advanced" in user_input_lower:
            params["difficulty"] = "expert"
        elif "medium" in user_input_lower or "intermediate" in user_input_lower:
            params["difficulty"] = "medium"
        
        # Extract number
        numbers = re.findall(r'\b(\d+)\b', user_input)
        if numbers:
            num = int(numbers[0])
            if 1 <= num <= 50:
                params["num_questions"] = num
        
        # Extract question type
        if "practical" in user_input_lower or "scenario" in user_input_lower or "real-world" in user_input_lower:
            params["question_type"] = "practical"
        elif "conceptual" in user_input_lower or "concept" in user_input_lower or "theory" in user_input_lower or "theoretical" in user_input_lower:
            params["question_type"] = "conceptual"
        
        return params
    
    def _extract_topic(self, user_input: str, user_input_lower: str) -> Optional[str]:
        """Extract topic from user input using multiple strategies"""
        # Strategy 1: Look for "about", "on", "regarding" patterns
        topic_keywords = ["about", "on", "regarding", "questions about", "quiz on", "questions on", "test on"]
        
        for keyword in topic_keywords:
            if keyword in user_input_lower:
                parts = user_input_lower.split(keyword)
                if len(parts) > 1:
                    topic_part = parts[1].strip()
                    # Clean up and extract meaningful topic
                    topic_words = topic_part.split()[:4]  # Take first 4 words
                    topic = " ".join(word for word in topic_words if word.isalpha() or word.isalnum())
                    if len(topic) >= 2:
                        return topic
        
        # Strategy 2: Look for patterns like "create X questions about Y"
        patterns = [
            r'create.*?questions.*?(?:about|on)\s+([a-zA-Z\s]+)',
            r'generate.*?questions.*?(?:about|on)\s+([a-zA-Z\s]+)',
            r'make.*?questions.*?(?:about|on)\s+([a-zA-Z\s]+)',
            r'(?:about|on)\s+([a-zA-Z\s]{2,30})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                topic = match.group(1).strip()
                # Clean up topic
                topic_words = topic.split()[:4]
                clean_topic = " ".join(word for word in topic_words if word.isalpha())
                if len(clean_topic) >= 2:
                    return clean_topic
        
        # Strategy 3: Look for common subject areas
        subjects = [
            "python", "javascript", "java", "c++", "programming", "coding",
            "machine learning", "ai", "artificial intelligence", "data science",
            "mathematics", "physics", "chemistry", "biology", "history",
            "geography", "english", "literature", "economics", "finance",
            "marketing", "business", "management", "psychology", "sociology",
            "html", "css", "react", "node", "database", "sql", "web development"
        ]
        
        for subject in subjects:
            if subject in user_input_lower:
                return subject
        
        # Strategy 4: Extract capitalized words (likely proper nouns/topics)
        words = user_input.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
        if capitalized_words:
            return " ".join(capitalized_words[:3])
        
        return None
    
    def _format_questions_response(self, result: Dict) -> str:
        """Format the generated questions into a nice response"""
        if "questions" not in result:
            return "I generated some questions but had trouble formatting them. Please try the /generate endpoint directly."
        
        questions = result["questions"]
        metadata = result.get("metadata", {})
        topic = metadata.get("topic", "the topic")
        difficulty = metadata.get("difficulty", "medium")
        question_type = metadata.get("question_type", "academic")
        
        response = f"Great! I've created {len(questions)} {difficulty}-level {question_type} questions about {topic}:\n\n"
        
        for i, q in enumerate(questions, 1):
            response += f"**Question {i}:**\n{q['question']}\n\n"
            
            for option, text in q["options"].items():
                marker = "✅" if option == q["correct_answer"] else "  "
                response += f"{marker} {option}) {text}\n"
            
            response += f"\n💡 **Explanation:** {q['explanation']}\n"
            response += "─" * 50 + "\n\n"
        
        response += "Would you like me to create more questions, modify any of these, or change the difficulty level?"
        return response

# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize conversation handler
try:
    conversation_handler = ConversationHandler()
    logger.info("Flask app initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Flask app: {e}")
    conversation_handler = None

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        "message": "Educational Question Generator API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "/health": "GET - Health check",
            "/generate": "POST - Generate questions directly",
            "/chat": "POST - Conversational interface"
        },
        "example_usage": {
            "generate": {
                "url": "/generate",
                "method": "POST",
                "body": {
                    "topic": "Python Programming",
                    "difficulty": "medium",
                    "num_questions": 5,
                    "question_type": "academic"
                }
            },
            "chat": {
                "url": "/chat",
                "method": "POST", 
                "body": {
                    "message": "Create 3 easy questions about HTML"
                }
            }
        }
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Conversational interface endpoint"""
    try:
        if not conversation_handler:
            return jsonify({"error": "Conversation handler not initialized"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "Please provide a message"}), 400
        
        response = conversation_handler.process_message(user_input)
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"error": "Failed to process chat request"}), 500

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """Direct generation endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required field
        if "topic" not in data:
            return jsonify({"error": "Missing required field: topic"}), 400
        
        result = generate_questions(data)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify({
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Generate endpoint error: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Service health check"""
    try:
        conversation_items = len(conversation_handler.conversation_history) if conversation_handler else 0
        return jsonify({
            "status": "healthy" if conversation_handler else "unhealthy",
            "model": "llama3-70b-8192",
            "conversation_items": conversation_items,
            "timestamp": datetime.now().isoformat(),
            "groq_api_configured": bool(GROQ_API_KEY),
            "version": "2.0.0"
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    logger.info("Starting Educational Question Generator Service v2.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        threaded=True
    )