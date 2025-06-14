import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from utils import *
from openai import OpenAI

class ConversationState(Enum):
    INITIAL = "initial"
    SYMPTOM_EXTRACTION = "symptom_extraction"
    SYMPTOM_REFINEMENT = "symptom_refinement"
    DISEASE_ANALYSIS = "disease_analysis"
    FOLLOW_UP_QUESTIONS = "follow_up_questions"
    FINAL_RECOMMENDATION = "final_recommendation"

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = None

@dataclass
class SymptomData:
    extracted_symptoms: List[str]
    confirmed_symptoms: List[str]
    severity_info: Dict[str, str]
    duration_info: Dict[str, str]
    additional_context: Dict[str, Any]

def get_sambanova_response(messages, model="Meta-Llama-3.3-70B-Instruct", system_prompt=None):
    """Helper function to get response from SambaNova API"""
    client = OpenAI(
        api_key="f9c890ca-64fa-4e37-ab62-fd9a1e6c4de6",
        base_url="https://api.sambanova.ai/v1",
    )
    
    # Format messages properly for OpenAI-style API
    formatted_messages = []
    
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation messages
    for msg in messages:
        formatted_messages.append({
            "role": msg["role"] if isinstance(msg, dict) else msg.role,
            "content": msg["content"] if isinstance(msg, dict) else msg.content
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting SambaNova response: {e}")
        return "I'm sorry, I'm having trouble processing your request. Please try again."

class MedicalChatbot:
    def __init__(self, rag_indexer):
        """
        Initialize the medical chatbot with SambaNova API and RAG system
        
        Args:
            rag_indexer: Instance of your MedicalRAGIndexer or string path to indexes
        """
        # Handle both object and string inputs for backwards compatibility
        if isinstance(rag_indexer, str):
            # If string path provided, create and load RAG indexer
            from utils import MedicalRAGIndexer
            self.rag_indexer = MedicalRAGIndexer()
            self.rag_indexer.load_indexes(rag_indexer)
        else:
            # If RAG indexer object provided, use it directly
            self.rag_indexer = rag_indexer
        
        # Conversation management
        self.conversation_history: List[ChatMessage] = []
        self.current_state = ConversationState.INITIAL
        self.symptom_data = SymptomData([], [], {}, {}, {})
        self.disease_candidates: List[Dict] = []
        self.question_count = 0
        self.max_questions = 6  # Reduced for better UX
        
        # System prompts for different stages
        self.system_prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different conversation stages"""
        return {
            "symptom_extraction": """You are a medical assistant that extracts symptoms from patient descriptions.

Your task: Extract all symptoms mentioned by the user and return them in a structured JSON format.

Rules:
1. Extract symptoms in their medical terminology when possible
2. Normalize similar symptoms (e.g., "tummy ache" -> "abdominal pain", "runny nose" -> "rhinorrhea")
3. Include severity and duration if mentioned
4. Be conservative - only extract what's clearly stated

Response format (MUST be valid JSON):
{
    "extracted_symptoms": ["symptom1", "symptom2"],
    "severity": {"symptom1": "mild/moderate/severe"},
    "duration": {"symptom1": "hours/days/weeks"},
    "additional_info": "any other relevant context"
}

Examples:
User: "I have a bad headache for 2 days and feeling nauseous"
Response: {
    "extracted_symptoms": ["headache", "nausea"],
    "severity": {"headache": "severe"},
    "duration": {"headache": "2 days"},
    "additional_info": ""
}""",

            "follow_up_questions": """You are a medical assistant conducting a symptom assessment interview.

Based on the current symptoms and potential diseases, ask 1-2 targeted follow-up questions to help narrow down the diagnosis.

Current context will be provided about confirmed symptoms, top disease candidates, and missing key symptoms.

Rules:
1. Ask about the most important differentiating symptoms first
2. Use simple, clear language
3. Ask about one symptom group at a time
4. Include severity/duration questions when relevant
5. Be empathetic and professional
6. Don't repeat questions already asked

Format your response as a natural conversation, not a list.""",

            "disease_analysis": """You are a medical assistant providing preliminary disease analysis.

Based on the symptoms and RAG analysis, provide a clear, empathetic explanation of possible conditions.

Rules:
1. Present 2-3 most likely conditions with confidence levels
2. Explain the reasoning clearly
3. Emphasize this is not a diagnosis
4. Recommend appropriate next steps
5. Include red flags if present
6. Be reassuring but honest

Use this format:
- Brief summary of symptoms
- Most likely possibilities with explanations
- Recommended actions
- When to seek immediate care""",

            "general_conversation": """You are a helpful medical assistant chatbot. You help users understand their symptoms and guide them through a medical assessment process.

Key capabilities:
1. Extract symptoms from natural language
2. Ask clarifying questions
3. Provide preliminary health guidance
4. Recommend when to see healthcare professionals

Always be:
- Professional and empathetic
- Clear that you're not providing medical diagnosis
- Encouraging about seeking professional help when needed
- Careful not to cause alarm unnecessarily"""
        }
    
    async def process_user_message(self, user_input: str) -> str:
        """
        Main method to process user input and generate appropriate response
        
        Args:
            user_input: The user's message
            
        Returns:
            Bot's response as a string
        """
        # Add user message to history
        self.conversation_history.append(ChatMessage("user", user_input))
        
        # Process based on current conversation state
        if self.current_state == ConversationState.INITIAL:
            response = await self._handle_initial_message(user_input)
        elif self.current_state == ConversationState.SYMPTOM_EXTRACTION:
            response = await self._handle_symptom_extraction(user_input)
        elif self.current_state == ConversationState.SYMPTOM_REFINEMENT:
            response = await self._handle_symptom_refinement(user_input)
        elif self.current_state == ConversationState.FOLLOW_UP_QUESTIONS:
            response = await self._handle_follow_up_questions(user_input)
        else:
            response = await self._handle_general_conversation(user_input)
        
        # Add bot response to history
        self.conversation_history.append(ChatMessage("assistant", response))
        
        return response
    
    async def _handle_initial_message(self, user_input: str) -> str:
        """Handle the first message from user"""
        # Extract symptoms from initial message
        extracted_data = await self._extract_symptoms(user_input)
        
        if extracted_data["extracted_symptoms"]:
            # Found symptoms, move to refinement
            self.symptom_data.extracted_symptoms = extracted_data["extracted_symptoms"]
            self.symptom_data.severity_info.update(extracted_data.get("severity", {}))
            self.symptom_data.duration_info.update(extracted_data.get("duration", {}))
            
            # Get initial RAG results
            rag_results = self.rag_indexer.query_diseases(self.symptom_data.extracted_symptoms, top_k=5)
            self.disease_candidates = rag_results
            
            response = f"""I understand you're experiencing {', '.join(self.symptom_data.extracted_symptoms)}. Let me ask a few more questions to better understand your situation and provide more accurate guidance.

"""
            
            # Generate follow-up questions
            follow_up = await self._generate_follow_up_questions()
            response += follow_up
            
            self.current_state = ConversationState.FOLLOW_UP_QUESTIONS
            
        else:
            # No clear symptoms found, ask for clarification
            response = """Hello! I'm here to help you understand your symptoms and guide you on next steps.

Could you please describe what symptoms or health concerns you're experiencing? For example:
- Physical symptoms (pain, fever, nausea, etc.)
- When they started
- How severe they are

The more details you can provide, the better I can assist you."""
            
            self.current_state = ConversationState.SYMPTOM_EXTRACTION
        
        return response
    
    async def _handle_symptom_extraction(self, user_input: str) -> str:
        """Handle additional symptom extraction"""
        extracted_data = await self._extract_symptoms(user_input)
        
        if extracted_data["extracted_symptoms"]:
            self.symptom_data.extracted_symptoms.extend(extracted_data["extracted_symptoms"])
            self.symptom_data.severity_info.update(extracted_data.get("severity", {}))
            self.symptom_data.duration_info.update(extracted_data.get("duration", {}))
            
            # Remove duplicates
            self.symptom_data.extracted_symptoms = list(set(self.symptom_data.extracted_symptoms))
            
            # Move to follow-up questions
            self.current_state = ConversationState.FOLLOW_UP_QUESTIONS
            
            # Get RAG results
            rag_results = self.rag_indexer.query_diseases(self.symptom_data.extracted_symptoms, top_k=5)
            self.disease_candidates = rag_results
            
            response = f"""Thank you for the additional information. I've noted these symptoms: {', '.join(self.symptom_data.extracted_symptoms)}.

"""
            
            # Generate follow-up questions
            follow_up = await self._generate_follow_up_questions()
            response += follow_up
            
        else:
            response = """I'm having trouble identifying specific symptoms from your description. Could you please be more specific about what you're feeling? 

For example:
- Do you have pain anywhere? Where?
- Any fever or temperature changes?
- Digestive issues?
- Respiratory symptoms?
- Skin changes?"""
        
        return response
    
    async def _handle_follow_up_questions(self, user_input: str) -> str:
        """Handle responses to follow-up questions"""
        # Extract any new symptoms from the response
        new_symptoms = await self._extract_symptoms(user_input)
        
        # Add new symptoms if found
        if new_symptoms["extracted_symptoms"]:
            self.symptom_data.extracted_symptoms.extend(new_symptoms["extracted_symptoms"])
            self.symptom_data.extracted_symptoms = list(set(self.symptom_data.extracted_symptoms))
            
            # Update RAG results
            rag_results = self.rag_indexer.query_diseases(self.symptom_data.extracted_symptoms, top_k=5)
            self.disease_candidates = rag_results
        
        self.question_count += 1
        
        # Decide next step - be more aggressive about moving to analysis
        if (self.question_count >= self.max_questions or 
            len(self.symptom_data.extracted_symptoms) >= 5 or
            "continue" in user_input.lower() or 
            "analysis" in user_input.lower()):
            # Move to final analysis
            self.current_state = ConversationState.FINAL_RECOMMENDATION
            response = await self._generate_final_analysis()
        else:
            # Continue with more questions
            follow_up = await self._generate_follow_up_questions()
            response = f"""Thank you for that information. 

{follow_up}"""
        
        return response
    
    async def _extract_symptoms(self, user_input: str) -> Dict[str, Any]:
        """Extract symptoms from user input using SambaNova"""
        try:
            messages = [{"role": "user", "content": user_input}]
            response_text = get_sambanova_response(messages, system_prompt=self.system_prompts["symptom_extraction"])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                return extracted_data
            else:
                return {"extracted_symptoms": [], "severity": {}, "duration": {}}
                
        except Exception as e:
            print(f"Error in symptom extraction: {e}")
            return {"extracted_symptoms": [], "severity": {}, "duration": {}}
    
    async def _generate_follow_up_questions(self) -> str:
        """Generate intelligent follow-up questions"""
        if not self.disease_candidates:
            return "Could you tell me about any other symptoms you might be experiencing?"
        
        # Get suggested symptoms from RAG
        try:
            suggested_symptoms = self.rag_indexer.get_symptom_suggestions(
                self.symptom_data.extracted_symptoms, 
                top_diseases=3
            )
        except:
            suggested_symptoms = []
        
        # Prepare context for the LLM
        top_diseases = [f"{r['disease']} (score: {r.get('confidence_score', r.get('confidence', 0)):.2f})" 
                       for r in self.disease_candidates[:3]]
        
        missing_symptoms = suggested_symptoms[:5] if suggested_symptoms else []
        
        context = f"""
Confirmed symptoms: {', '.join(self.symptom_data.extracted_symptoms)}
Top disease candidates: {', '.join(top_diseases)}
Missing key symptoms to check: {', '.join(missing_symptoms)}
Question count: {self.question_count}
"""
        
        try:
            messages = [{"role": "user", "content": context + "\n\nGenerate 1-2 targeted follow-up questions to help narrow down the diagnosis."}]
            response = get_sambanova_response(messages, system_prompt=self.system_prompts["follow_up_questions"])
            return response
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return "Are there any other symptoms you'd like to mention?"
    
    async def _generate_final_analysis(self) -> str:
        """Generate final disease analysis and recommendations"""
        if not self.disease_candidates:
            return """Based on your symptoms, I recommend consulting with a healthcare professional for proper evaluation. 

If you're experiencing severe symptoms or feel unwell, please don't hesitate to seek medical attention."""
        
        # Prepare comprehensive analysis context
        analysis_context = f"""
Current symptoms: {', '.join(self.symptom_data.extracted_symptoms)}

Top disease candidates from analysis:
"""
        
        for i, result in enumerate(self.disease_candidates[:3], 1):
            confidence = result.get('confidence_score', result.get('confidence', 0))
            matched_symptoms = result.get('matched_symptoms', [])
            match_ratio = result.get('match_ratio', 0)
            
            analysis_context += f"""
{i}. {result['disease'].title()}:
   - Confidence: {confidence:.2f}
   - Matched symptoms: {', '.join(matched_symptoms) if matched_symptoms else 'N/A'}
   - Match ratio: {match_ratio:.2f if match_ratio else 'N/A'}
"""
        
        try:
            messages = [{"role": "user", "content": analysis_context + "\n\nProvide a comprehensive but reassuring analysis with recommendations."}]
            analysis = get_sambanova_response(messages, system_prompt=self.system_prompts["disease_analysis"])
            
            # Add disclaimer and next steps
            disclaimer = """

---
**Important Disclaimer:**
This analysis is for informational purposes only and is not a medical diagnosis. Please consult with a qualified healthcare professional for proper medical evaluation and treatment.

**Recommended Next Steps:**
- Schedule an appointment with your primary care physician
- If symptoms worsen or you develop new concerning symptoms, seek immediate medical attention
- Keep track of your symptoms and any changes

**Seek Emergency Care Immediately if you experience:**
- Severe difficulty breathing
- Chest pain
- Severe headache with neck stiffness
- High fever (over 103°F/39.4°C)
- Severe abdominal pain
- Signs of severe dehydration"""
            
            return analysis + disclaimer
            
        except Exception as e:
            print(f"Error generating final analysis: {e}")
            return """I recommend consulting with a healthcare professional about your symptoms for proper evaluation and guidance.

Please seek medical attention if your symptoms worsen or if you have any concerns."""
    
    async def _handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation or questions"""
        try:
            # Build conversation context
            recent_messages = self.conversation_history[-6:]  # Last 3 exchanges
            messages = []
            
            for msg in recent_messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            response = get_sambanova_response(messages, system_prompt=self.system_prompts["general_conversation"])
            return response
            
        except Exception as e:
            print(f"Error in general conversation: {e}")
            return "I'm sorry, I'm having trouble processing your request. Could you please rephrase or ask a specific question about your symptoms?"
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.conversation_history = []
        self.current_state = ConversationState.INITIAL
        self.symptom_data = SymptomData([], [], {}, {}, {})
        self.disease_candidates = []
        self.question_count = 0
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return {
            "current_state": self.current_state.value,
            "extracted_symptoms": self.symptom_data.extracted_symptoms,
            "question_count": self.question_count,
            "top_disease_candidates": [
                {"disease": r["disease"], "confidence": r.get("confidence_score", r.get("confidence", 0))} 
                for r in self.disease_candidates[:3]
            ],
            "conversation_length": len(self.conversation_history)
        }

# Sync wrapper for Streamlit compatibility
class MedicalChatbotSync:
    """Synchronous wrapper for the async MedicalChatbot"""
    
    def __init__(self, rag_indexer_or_path):
        self.async_chatbot = MedicalChatbot(rag_indexer_or_path)
    
    def process_user_message(self, user_input: str) -> str:
        """Synchronous wrapper for process_user_message"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.async_chatbot.process_user_message(user_input))
                    return future.result()
            else:
                return loop.run_until_complete(self.async_chatbot.process_user_message(user_input))
        except RuntimeError:
            # Create new event loop if none exists
            return asyncio.run(self.async_chatbot.process_user_message(user_input))
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.async_chatbot.reset_conversation()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation"""
        return self.async_chatbot.get_conversation_summary()
    
    @property
    def current_state(self):
        """Get current conversation state"""
        return self.async_chatbot.current_state
    
    @property
    def rag_indexer(self):
        """Access to RAG indexer"""
        return self.async_chatbot.rag_indexer

# For backwards compatibility, keep the original name
MedicalChatbot = MedicalChatbotSync