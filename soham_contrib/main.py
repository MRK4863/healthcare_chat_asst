import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from openai import OpenAI
from utils import MedicalRAGIndexer

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

def get_sambanova_response(messages, model="Meta-Llama-3.3-70B-Instruct", temperature=0.1, top_p=0.1):
    """
    Get response from SambaNova API
    
    Args:
        messages: List of message dictionaries
        model: Model name to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        Response content as string
    """
    client = OpenAI(
        api_key="f9c890ca-64fa-4e37-ab62-fd9a1e6c4de6",
        base_url="https://api.sambanova.ai/v1",
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    
    return response.choices[0].message.content

class MedicalChatbot:
    def __init__(self, rag_indexer_path: str = "medical_rag_indexes"):
        """
        Initialize the medical chatbot with SambaNova API and RAG system
        
        Args:
            rag_indexer_path: Path to the saved RAG indexes (relative to root directory)
        """
        # Load RAG indexer from saved indexes
        self.rag_indexer = MedicalRAGIndexer()
        try:
            self.rag_indexer.load_indexes(rag_indexer_path)
            print("✅ RAG indexes loaded successfully")
        except Exception as e:
            print(f"❌ Error loading RAG indexes: {e}")
            print("Please ensure indexes are created using the utils.py script first")
            raise
        
        # Conversation management
        self.conversation_history: List[ChatMessage] = []
        self.current_state = ConversationState.INITIAL
        self.symptom_data = SymptomData([], [], {}, {}, {})
        self.disease_candidates: List[Dict] = []
        self.question_count = 0
        self.max_questions = 6
        
        # System prompts for different stages
        self.system_prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different conversation stages"""
        return {
            "symptom_extraction": """You are a medical assistant that extracts symptoms from patient descriptions.

Your task: Extract all symptoms mentioned by the user and return them in a structured JSON format.

Rules:
1. Extract symptoms in their medical terminology when possible
2. Normalize similar symptoms (e.g., "tummy ache" -> "abdominal pain")
3. Include severity and duration if mentioned
4. Be conservative - only extract what's clearly stated
5. Return ONLY valid JSON, no other text

Response format:
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

Your role: Ask 1-2 targeted follow-up questions to help narrow down the diagnosis based on the current symptoms and potential diseases.

Rules:
1. Ask about the most important differentiating symptoms from the missing symptoms list
2. Use simple, clear language that patients can understand
3. Ask about one symptom group at a time
4. Include severity/duration questions when relevant
5. Be empathetic and professional
6. Keep questions concise and focused

Format your response as a natural conversation, not a list.""",

            "disease_analysis": """You are a medical assistant providing preliminary disease analysis.

Your role: Based on the symptoms and medical analysis, provide a clear, empathetic explanation of possible conditions.

Rules:
1. Present 2-3 most likely conditions with confidence levels
2. Explain the reasoning clearly based on symptom matching
3. Emphasize this is NOT a medical diagnosis
4. Recommend appropriate next steps
5. Include red flags if present
6. Be reassuring but honest
7. Use clear, non-technical language

Structure:
- Brief summary of reported symptoms
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
- Careful not to cause alarm unnecessarily

Keep responses helpful and focused."""
        }
    
    def process_user_message(self, user_input: str) -> str:
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
        try:
            if self.current_state == ConversationState.INITIAL:
                response = self._handle_initial_message(user_input)
            elif self.current_state == ConversationState.SYMPTOM_EXTRACTION:
                response = self._handle_symptom_extraction(user_input)
            elif self.current_state == ConversationState.SYMPTOM_REFINEMENT:
                response = self._handle_symptom_refinement(user_input)
            elif self.current_state == ConversationState.FOLLOW_UP_QUESTIONS:
                response = self._handle_follow_up_questions(user_input)
            else:
                response = self._handle_general_conversation(user_input)
        except Exception as e:
            print(f"Error processing message: {e}")
            response = "I'm sorry, I encountered an error. Could you please rephrase your message?"
        
        # Add bot response to history
        self.conversation_history.append(ChatMessage("assistant", response))
        
        return response
    
    def _handle_initial_message(self, user_input: str) -> str:
        """Handle the first message from user"""
        # Extract symptoms from initial message
        extracted_data = self._extract_symptoms(user_input)
        
        if extracted_data and extracted_data.get("extracted_symptoms"):
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
            follow_up = self._generate_follow_up_questions()
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
    
    def _handle_symptom_extraction(self, user_input: str) -> str:
        """Handle additional symptom extraction"""
        extracted_data = self._extract_symptoms(user_input)
        
        if extracted_data and extracted_data.get("extracted_symptoms"):
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
            follow_up = self._generate_follow_up_questions()
            response += follow_up
            
        else:
            response = """I'm having trouble identifying specific symptoms from your description. Could you please be more specific about what you're feeling? 

For example:
- Do you have pain anywhere? Where?
- Any fever or temperature changes?
- Digestive issues like nausea or stomach pain?
- Respiratory symptoms like cough or difficulty breathing?
- Any skin changes or rashes?"""
        
        return response
    
    def _handle_follow_up_questions(self, user_input: str) -> str:
        """Handle responses to follow-up questions"""
        # Extract any new symptoms from the response
        new_symptoms = self._extract_symptoms(user_input)
        
        # Add new symptoms if found
        if new_symptoms and new_symptoms.get("extracted_symptoms"):
            self.symptom_data.extracted_symptoms.extend(new_symptoms["extracted_symptoms"])
            self.symptom_data.extracted_symptoms = list(set(self.symptom_data.extracted_symptoms))
            
            # Update RAG results
            rag_results = self.rag_indexer.query_diseases(self.symptom_data.extracted_symptoms, top_k=5)
            self.disease_candidates = rag_results
        
        self.question_count += 1
        
        # Decide next step
        if self.question_count >= self.max_questions or len(self.symptom_data.extracted_symptoms) >= 5:
            # Move to final analysis
            self.current_state = ConversationState.FINAL_RECOMMENDATION
            response = self._generate_final_analysis()
        else:
            # Continue with more questions
            follow_up = self._generate_follow_up_questions()
            response = f"""Thank you for that information. 

{follow_up}"""
        
        return response
    
    def _extract_symptoms(self, user_input: str) -> Dict[str, Any]:
        """Extract symptoms from user input using SambaNova"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompts["symptom_extraction"]},
                {"role": "user", "content": user_input}
            ]
            
            response_text = get_sambanova_response(messages, temperature=0.1, top_p=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group())
                    return extracted_data
                except json.JSONDecodeError:
                    print(f"JSON decode error. Response: {response_text}")
                    return {"extracted_symptoms": [], "severity": {}, "duration": {}}
            else:
                print(f"No JSON found in response: {response_text}")
                return {"extracted_symptoms": [], "severity": {}, "duration": {}}
                
        except Exception as e:
            print(f"Error in symptom extraction: {e}")
            return {"extracted_symptoms": [], "severity": {}, "duration": {}}
    
    def _generate_follow_up_questions(self) -> str:
        """Generate intelligent follow-up questions"""
        if not self.disease_candidates:
            return "Could you tell me about any other symptoms you might be experiencing?"
        
        try:
            # Get suggested symptoms from RAG
            suggested_symptoms = self.rag_indexer.get_symptom_suggestions(
                self.symptom_data.extracted_symptoms, 
                top_diseases=3
            )
            
            # Prepare context for the LLM
            top_diseases = [f"{r['disease']} (confidence: {r['score']:.2f})" 
                           for r in self.disease_candidates[:3]]
            
            missing_symptoms = suggested_symptoms[:5] if suggested_symptoms else []
            
            context = f"""Current confirmed symptoms: {', '.join(self.symptom_data.extracted_symptoms)}

Top disease candidates: {', '.join(top_diseases)}

Important symptoms to check: {', '.join(missing_symptoms)}

Generate 1-2 targeted follow-up questions to help differentiate between these conditions."""
            
            messages = [
                {"role": "system", "content": self.system_prompts["follow_up_questions"]},
                {"role": "user", "content": context}
            ]
            
            response = get_sambanova_response(messages, temperature=0.3, top_p=0.1)
            return response
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return "Are there any other symptoms you'd like to mention?"
    
    def _generate_final_analysis(self) -> str:
        """Generate final disease analysis and recommendations"""
        if not self.disease_candidates:
            return """Based on your symptoms, I recommend consulting with a healthcare professional for proper evaluation. 

If you're experiencing severe symptoms or feel unwell, please don't hesitate to seek medical attention."""
        
        try:
            # Prepare comprehensive analysis context
            analysis_context = f"""Patient symptoms: {', '.join(self.symptom_data.extracted_symptoms)}

Medical analysis results:
"""
            
            for i, result in enumerate(self.disease_candidates[:3], 1):
                analysis_context += f"""
{i}. {result['disease'].title()}:
   - Confidence score: {result['score']:.2f}
   - Matched symptoms: {', '.join(result['matched_symptoms'])}
   - Symptom match ratio: {result['match_ratio']:.2f}
   - Coverage ratio: {result['coverage_ratio']:.2f}
"""
            
            analysis_context += "\n\nProvide a comprehensive but reassuring analysis with clear recommendations."
            
            messages = [
                {"role": "system", "content": self.system_prompts["disease_analysis"]},
                {"role": "user", "content": analysis_context}
            ]
            
            analysis = get_sambanova_response(messages, temperature=0.2, top_p=0.1)
            
            # Add disclaimer and next steps
            disclaimer = """

---
**Important Disclaimer:**
This analysis is for informational purposes only and is NOT a medical diagnosis. Please consult with a qualified healthcare professional for proper medical evaluation and treatment.

**Recommended Next Steps:**
- Schedule an appointment with your primary care physician
- If symptoms worsen or you develop new concerning symptoms, seek immediate medical attention
- Keep track of your symptoms and any changes

**Seek Emergency Care Immediately if you experience:**
- Severe difficulty breathing
- Chest pain or pressure
- Severe headache with neck stiffness
- High fever (over 103°F/39.4°C)
- Severe abdominal pain
- Signs of severe dehydration
- Loss of consciousness or severe confusion"""
            
            return analysis + disclaimer
            
        except Exception as e:
            print(f"Error generating final analysis: {e}")
            return """Based on your symptoms, I recommend consulting with a healthcare professional for proper evaluation and guidance.

Please seek medical attention if your symptoms worsen or if you have any concerns about your health."""
    
    def _handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation or questions"""
        try:
            # Build conversation context (last 3 exchanges)
            recent_messages = []
            
            # Add system message
            recent_messages.append({
                "role": "system", 
                "content": self.system_prompts["general_conversation"]
            })
            
            # Add recent conversation history
            for msg in self.conversation_history[-6:]:
                recent_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current user input
            recent_messages.append({
                "role": "user",
                "content": user_input
            })
            
            response = get_sambanova_response(recent_messages, temperature=0.3, top_p=0.1)
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
                {"disease": r["disease"], "confidence": r["score"]} 
                for r in self.disease_candidates[:3]
            ],
            "conversation_length": len(self.conversation_history)
        }

# Demo and Testing Classes
class MedicalChatbotDemo:
    def __init__(self, rag_indexer_path: str = "medical_rag_indexes"):
        self.chatbot = MedicalChatbot(rag_indexer_path)
    
    def run_demo_conversation(self):
        """Run a demo conversation"""
        print("🤖 Medical Chatbot Demo")
        print("=" * 50)
        print("Type 'quit' to exit, 'reset' to start over, 'summary' for conversation summary")
        print()
        
        # Initial greeting
        greeting = """Hello! I'm your medical assistant chatbot. I'm here to help you understand your symptoms and guide you on next steps.

Please describe any symptoms or health concerns you're experiencing. Remember, I provide preliminary guidance only - always consult healthcare professionals for medical diagnosis and treatment."""
        
        print("🤖 Bot:", greeting)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Take care! Remember to consult healthcare professionals for any health concerns.")
                    break
                elif user_input.lower() == 'reset':
                    self.chatbot.reset_conversation()
                    print("🔄 Conversation reset. How can I help you today?")
                    continue
                elif user_input.lower() == 'summary':
                    summary = self.chatbot.get_conversation_summary()
                    print("📊 Conversation Summary:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                    continue
                
                if not user_input:
                    continue
                
                # Get bot response
                print("🤔 Thinking...")
                response = self.chatbot.process_user_message(user_input)
                print(f"\n🤖 Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'reset' to start over.")

# Production API wrapper
class ProductionChatbotAPI:
    """Production-ready API wrapper for the medical chatbot"""
    
    def __init__(self, rag_indexer_path: str = "medical_rag_indexes"):
        self.chatbots = {}  # Session management
        self.rag_indexer_path = rag_indexer_path
    
    def handle_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Handle a message for a specific session"""
        
        # Get or create chatbot for session
        if session_id not in self.chatbots:
            self.chatbots[session_id] = MedicalChatbot(self.rag_indexer_path)
        
        chatbot = self.chatbots[session_id]
        
        # Process message
        response = chatbot.process_user_message(message)
        
        # Return structured response
        return {
            "response": response,
            "session_id": session_id,
            "conversation_state": chatbot.current_state.value,
            "summary": chatbot.get_conversation_summary()
        }
    
    def end_session(self, session_id: str):
        """End a conversation session"""
        if session_id in self.chatbots:
            del self.chatbots[session_id]
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.chatbots.keys())

# Main execution
def main():
    """Main function to run the chatbot demo"""
    print("🚀 Starting Medical Chatbot System")
    print("=" * 50)
    
    try:
        # Initialize demo (assumes RAG indexes are already created)
        demo = MedicalChatbotDemo("medical_rag_indexes")
        
        # Run interactive demo
        demo.run_demo_conversation()
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\n📋 Make sure you have:")
        print("1. Created RAG indexes using utils.py")
        print("2. Saved indexes to 'medical_rag_indexes' in root directory")
        print("3. Installed required dependencies (openai package)")

def test_chatbot():
    """Test function for development"""
    try:
        chatbot = MedicalChatbot("medical_rag_indexes")
        
        # Test conversation flow
        test_messages = [
            "I have a fever and headache",
            "Yes, I also have body aches and feel tired",
            "No cough, but I lost my sense of taste"
        ]
        
        print("🧪 Testing chatbot with sample conversation:")
        print("=" * 50)
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n👤 Test message {i}: {message}")
            response = chatbot.process_user_message(message)
            print(f"🤖 Bot response: {response[:200]}...")
            print(f"📊 State: {chatbot.current_state.value}")
        
        # Print final summary
        summary = chatbot.get_conversation_summary()
        print(f"\n📋 Final Summary: {summary}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # You can run either the demo or test
    # main()  # Interactive demo
    test_chatbot()  # Automated test