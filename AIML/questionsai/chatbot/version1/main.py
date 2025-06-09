from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
from groq import Groq
import os
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Initialize logging - Essential for debugging and monitoring system behavior
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables - Securely manages API keys and configuration
load_dotenv()

# Validate required environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise ValueError("GROQ_API_KEY environment variable is required")

# Initialize Groq client - Interface to the LLM API
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

# Enhanced prompt templates - Critical for getting high-quality, structured responses
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

# Pydantic model for request validation - Ensures input data integrity
class QuestionRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200, description="Subject matter")
    difficulty: str = Field("medium", regex="^(easy|medium|hard|expert)$", description="Difficulty level")
    num_questions: int = Field(5, ge=1, le=50, description="Number of questions")
    question_type: str = Field("academic", regex="^(academic|practical|conceptual)$", description="Question type")

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

# Core question generation function - The engine that creates questions
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

# Agent tools and setup - Creates the intelligent assistant
class QuestionGeneratorAgent:
    def __init__(self):
        try:
            # LLM with balanced creativity/accuracy - FIXED API KEY PARAMETER
            self.llm = ChatGroq(
                model="llama3-70b-8192", 
                temperature=0.5,
                api_key=GROQ_API_KEY  # Changed from groq_api_key to api_key
            )
            
            # Conversation memory - fixed memory key configuration
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=6,
                return_messages=True
            )
            
            # Tools
            self.tools = self._setup_tools()
            
            # Create ReAct agent
            self.agent = self._create_agent()
            logger.info("ReAct agent initialized successfully")
            
            logger.info("Question Generator Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def _setup_tools(self) -> List[Tool]:
        return [
            Tool(
                name="QuestionGenerator",
                func=self.generate_with_fallback,
                description=(
                    "Generates educational questions with replacement options. "
                    "Input must be a complete JSON string with: "
                    "{'topic': string, 'difficulty': string, "
                    "'num_questions': integer, 'question_type': string}"
                )
            )
        ]
    
    def generate_with_fallback(self, json_str: str) -> str:
        """Generate questions with extra replacements and validation"""
        try:
            data = json.loads(json_str)
            result = generate_questions(data)
            
            if "error" in result:
                return json.dumps({"status": "error", "message": result["error"]})
                
            # Generate buffer questions for replacements
            buffer_count = min(5, max(2, data.get("num_questions", 5)))
            extra_data = {**data, "num_questions": buffer_count}
            extra = generate_questions(extra_data)
            
            if "questions" in extra:
                result["replacement_pool"] = extra["questions"]
                result["metadata"]["total_replacements"] = len(extra["questions"])
                    
            return json.dumps({"status": "success", "data": result})
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return json.dumps({"status": "error", "message": "Invalid JSON input"})
        except Exception as e:
            logger.exception("Generation error")
            return json.dumps({"status": "error", "message": f"Generation failed: {str(e)}"})
    
    def _create_agent(self) -> AgentExecutor:
        # Enhanced system prompt with required ReAct template variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are Professor Quizwell, a kind and knowledgeable educational assistant specializing in creating learning materials.

Core Principles:
1. Be patient, encouraging, and supportive at all times
2. Collect parameters systematically when information is missing
3. Always confirm details before generation
4. Present questions clearly with explanations
5. Maintain a warm, professional tone

Workflow:
1. PARAMETER COLLECTION:
   - Required: topic (what subject to create questions about)
   - Optional: difficulty (easy/medium/hard/expert, default: medium)
   - Optional: num_questions (1-50, default: 5)
   - Optional: question_type (academic/practical/conceptual, default: academic)
   - If any required parameter is missing, ask for it politely
   - Summarize all parameters before confirmation

2. GENERATION CONFIRMATION:
   - Clearly state what you'll generate
   - Example: "I'll generate 5 medium-difficulty academic questions about Machine Learning"

3. QUESTION PRESENTATION:
   - Present questions in a clear, numbered format
   - Show all options clearly
   - Include explanations for correct answers

4. REPLACEMENT HANDLING:
   - Keep replacement questions ready from the pool
   - If user wants to replace a question, offer alternatives immediately

5. ERROR HANDLING:
   - If errors occur, explain simply and offer to retry
   - Never show technical details to users

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Always be helpful and ask how you can further assist!"""),
            
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Create ReAct agent with proper tool formatting
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Wrap in executor - REMOVED handle_parsing_errors
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input"""
        try:
            response = self.agent.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            return "I encountered an unexpected issue. Could you please try rephrasing your request?"

# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize agent globally
try:
    agent = QuestionGeneratorAgent()
    logger.info("Flask app initialized with agent")
except Exception as e:
    logger.error(f"Failed to initialize Flask app: {e}")
    logger.error(traceback.format_exc())
    agent = None

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
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/generate": "POST - Generate questions directly",
            "/chat": "POST - Conversational interface"
        },
        "status": "active" if agent else "error"
    })

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Conversational interface endpoint"""
    try:
        if not agent:
            return jsonify({"error": "Agent not initialized"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "Please provide a message"}), 400
        
        response = agent.run(user_input)
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
        memory_items = len(agent.memory.buffer) if agent and agent.memory else 0
        return jsonify({
            "status": "healthy" if agent else "unhealthy",
            "model": "llama3-70b-8192",
            "memory_items": memory_items,
            "timestamp": datetime.now().isoformat(),
            "groq_api_configured": bool(GROQ_API_KEY)
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    logger.info("Starting Educational Question Agent Service")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        threaded=True
    )