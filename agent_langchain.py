"""
MCP Notetaking Agent - Phase 2: Gemini + LangChain Integration
Real AI integration with Google's Gemini - FREE and POWERFUL!

MCP CONCEPTS BEING BUILT:
- Tools: ask_question, save_note, search_notes (powered by Gemini)
- Resources: conversation_history, note_metadata (with LangChain memory)
- Prompts: LangChain prompt templates optimized for Gemini
"""
import os
import asyncio
from datetime import datetime
from typing import Optional, List, Dict

# Disable LangChain tracing (like your setup)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv

# LangChain specific imports (following your pattern)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub  # for loading agent templates (like yours)

# Vector database imports (like your FAISS setup)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Python specific imports (like your setup)
from pydantic import BaseModel, Field





# Load environment variables (exactly like yours)
load_dotenv()


class MCPNoteTakingAgent:

    def __init__(self):
        self.conversation_history = []
        self.notes_database = []

        self._initialize_llm()
        self._initialize_embeddings()
        self._setup_vector_store()
        self._create_tools()
        self._create_agent()
        self._create_executor()

    def _initialize_llm(self):
        """
        STEP 1: Initialize the LLM (exactly like your approach)
        Using Gemini as the "brain" of our agent
        """
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No GOOGLE_API_KEY found in environment")
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                api_key=api_key,
                temperature=0.7
            )
            
        except Exception as e:
            self.llm = None


    def _initialize_embeddings(self):
        """
        STEP 2: Initialize embeddings model (your pattern)
        For vector database note searching
        """
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )
            
        except Exception as e:
            self.embeddings = None

    def _setup_vector_store(self):
        """
        STEP 3: Setup vector store for note searching (like your FAISS setup)
        This will store our conversation history for retrieval
        """
        try:
            self.vector_store = None
            self.retriever = None
            
        except Exception as e:
            pass

    
    
    
    
    
    def _update_vector_store(self):
        """
        Update vector store with conversation history (like your docs loading)
        FIXED: Better error handling for FAISS issues
        """
        if not self.embeddings or not self.conversation_history:
            return
            
        try:
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError as ie:
                return
            
            # Create documents from conversation history (like your document loading)
            documents = []
            for i, conv in enumerate(self.conversation_history):
                doc_text = f"Question: {conv['question']}\nAnswer: {conv['answer']}"
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "timestamp": conv["timestamp"],
                        "index": i,
                        "type": "qa_pair"
                    }
                )
                documents.append(doc)
            
            # Create/update vector store (like your FAISS creation)
            if len(documents) > 0:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
                
        except Exception as e:
            pass





    def _create_tools(self):
        """
        STEP 4: Create agent tools (exactly your pattern!)
        These will become MCP tools in later phases
        
        FIXED: Added Final Answer tool to prevent ReAct loops
        """
        
        # Tool 1: Ask Question (SIMPLIFIED)
        def ask_question_func(question: str) -> str:
            """Ask a question and get AI response - SYNC VERSION"""
            try:
                # Simple sync approach to avoid event loop issues
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if self.llm is None:
                    answer = f"Mock response to: '{question}' (Configure GOOGLE_API_KEY for real responses)"
                else:
                    # Direct LLM call without async complications
                    from langchain.schema import HumanMessage
                    try:
                        response = self.llm.invoke([HumanMessage(content=question)])
                        answer = response.content
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e):
                            answer = f"Rate limit reached. Mock response: '{question}' is about machine learning concepts and techniques."
                        else:
                            answer = f"Error: {e}"
                
                # Store in conversation history
                qa_pair = {
                    "timestamp": timestamp,
                    "question": question,
                    "answer": answer,
                    "model": "gemini-2.0-flash"
                }
                self.conversation_history.append(qa_pair)
                
                try:
                    self._update_vector_store()
                except Exception as ve:
                    pass
                
                return answer
                
            except Exception as e:
                return f"Tool error: {e}"
        
        # Final Answer tool (prevents infinite loops)
        def final_answer_func(answer: str) -> str:
            """Provide final answer to conclude conversation"""
            return f"Final Answer: {answer}"
        
        # Create simplified tools list
        self.tools = [
            Tool(
                name="Ask Question",
                func=ask_question_func,
                description="Use this to ask any question and get an AI response. Input should be a clear question."
            ),
            Tool(
                name="Final Answer",
                func=final_answer_func,
                description="Use this to provide the final answer and conclude the conversation. Input should be your complete response."
            )
        ]
        
        pass







    def _create_agent(self):
        """
        STEP 5: Create the ReAct agent (exactly your approach!)
        Using the same prompt template you used
        """
        try:
            if self.llm is None:
                self.agent = None
                return
                
            self.prompt_template = hub.pull("hwchase17/react")
            
            self.agent = create_react_agent(
                self.llm, 
                self.tools, 
                self.prompt_template
            )
            
        except Exception as e:
            self.agent = None

    def _create_executor(self):
        """
        STEP 6: Create AgentExecutor (exactly your setup!)
        This runs the agent with the same settings you used
        """
        try:
            if self.agent is None:
                self.agent_executor = None
                return
                
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True
            )
            
        except Exception as e:
            self.agent_executor = None


    async def _ask_question_internal(self, question: str) -> str:
        """Internal question asking (will become MCP tool)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.llm is None:
            # Mock response
            answer = f"Mock response to: '{question}' (Configure GOOGLE_API_KEY for real responses)"
        else:
            try:
                # Use your agent executor pattern!
                if self.agent_executor:
                    response = self.agent_executor.invoke({"input": question})
                    answer = response["output"]
                else:
                    # Direct LLM call as fallback
                    from langchain.schema import HumanMessage
                    response = await self.llm.ainvoke([HumanMessage(content=question)])
                    answer = response.content
                    
            except Exception as e:
                answer = f"Error getting response: {e}"
        
        # Store in conversation history
        qa_pair = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "model": "gemini-2.0-flash"
        }
        self.conversation_history.append(qa_pair)
        
        # Update vector store (like your document processing)
        self._update_vector_store()
        
        return answer
    


    async def _save_note_internal(self, content: str) -> str:
        """Internal note saving (will become MCP tool)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.md"
        display_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create markdown content
        markdown_content = f"""# 📝 Note - {display_time}

## 📄 Content
{content}

## 🤖 Agent Info
- **Created**: {display_time}
- **Agent**: MCP Notetaking Agent (Your ReAct Pattern)
- **LLM**: Gemini 2.0 Flash
- **Tools Used**: {[tool.name for tool in self.tools]}
- **Total Conversations**: {len(self.conversation_history)}

---
*Generated by MCP Agent using your proven LangChain patterns*
"""
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return f"Note saved as {filename}"
        except Exception as e:
            return f"Error saving note: {e}"

    def _search_notes_internal(self, query: str) -> str:
        """Internal note searching (will become MCP tool)"""
        if not self.retriever:
            # Fallback: simple text search
            results = []
            for i, conv in enumerate(self.conversation_history):
                if query.lower() in conv['question'].lower() or query.lower() in conv['answer'].lower():
                    results.append(f"{i+1}. {conv['timestamp']}: {conv['question'][:100]}...")
            
            if results:
                return f"Found {len(results)} matches:\n" + "\n".join(results)
            else:
                return "No matches found in conversation history"
        else:
            # Vector search (like your RetrievalQA)
            try:
                docs = self.retriever.get_relevant_documents(query)
                results = []
                for doc in docs:
                    results.append(f"• {doc.metadata['timestamp']}: {doc.page_content[:150]}...")
                
                return f"Vector search found {len(results)} relevant notes:\n" + "\n".join(results)
            except Exception as e:
                return f"Search error: {e}"

    # Public interface (for testing and MCP integration)
    
    def list_recent_notes(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation history"""
        recent = self.conversation_history[-limit:] if self.conversation_history else []
        return recent
    
    async def chat(self, message: str) -> str:
        """
        Main chat interface (like your main execution)
        This will become the MCP server interface
        """
        if self.agent_executor:
            # Use your agent executor pattern!
            try:
                response = self.agent_executor.invoke({"input": message})
                return response["output"]
            except Exception as e:
                return f"Agent error: {e}"
        else:
            # Fallback to direct question asking
            return await self._ask_question_internal(message)

async def main():
    """Simple Q&A Agent - Clean and Minimal"""
    print("🤖 MCP Agent Ready")
    print("Type 'quit' to exit and save conversation history")
    
    # Initialize agent
    agent = MCPNoteTakingAgent()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                # Save complete conversation history
                if agent.conversation_history:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"conversation_history_{timestamp}.md"
                    
                    markdown_content = f"""# 📝 Complete Conversation History
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Conversations**: {len(agent.conversation_history)}

---

"""
                    
                    for i, conv in enumerate(agent.conversation_history, 1):
                        markdown_content += f"""## Conversation {i}
**Time**: {conv['timestamp']}

**Q**: {conv['question']}

**A**: {conv['answer']}

---

"""
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Conversation saved to {filename}")
                
                break
            
            # Process question
            try:
                if agent.agent_executor:
                    response = agent.agent_executor.invoke({"input": user_input})
                    agent_response = response["output"]
                    
                    if agent_response.startswith("Final Answer: "):
                        agent_response = agent_response[14:]
                else:
                    agent_response = agent.tools[0].func(user_input)
                
                print(f"Agent: {agent_response}")
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e):
                    simple_response = f"I understand you're asking about '{user_input}'. This is a great question!"
                    print(f"Agent: {simple_response}")
                    
                    qa_pair = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": user_input,
                        "answer": simple_response,
                        "model": "rate-limited-fallback"
                    }
                    agent.conversation_history.append(qa_pair)
                else:
                    print(f"Error: {e}")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())