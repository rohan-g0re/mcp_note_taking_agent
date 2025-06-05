"""
Simple tests to verify our setup is working.
"""

import asyncio
from agent import NoteTakingAgent

async def test_basic_functionality():
    """Test our agent before MCP integration."""
    print("🧪 Testing basic agent functionality...")
    
    agent = NoteTakingAgent()
    
    # Test question asking
    answer = await agent.ask_question("Test question")
    assert len(answer) > 0, "Answer should not be empty"
    
    # Test note saving
    result = await agent.save_note("Test", "Test answer")
    assert "saved successfully" in result, "Note should save successfully"
    
    # Test listing notes
    notes = agent.list_recent_notes()
    assert len(notes) > 0, "Should have at least one note"
    
    print("✅ All basic tests passed!")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())