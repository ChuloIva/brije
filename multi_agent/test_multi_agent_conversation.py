"""
Test script for multi-agent conversation with cognitive action probes
"""

import sys
from pathlib import Path

# Add liminal_backrooms to path
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "liminal_backrooms"))

from third_party.liminal_backrooms.main import ai_turn
from third_party.liminal_backrooms.config import AI_MODELS, SYSTEM_PROMPT_PAIRS
import json

def test_multi_agent_conversation():
    """
    Test a simple conversation between two AI agents:
    - AI-1: Gemma 3 4B with cognitive action probes
    - AI-2: Gemma 3 4B with cognitive action probes
    """

    print("\n" + "="*80)
    print("MULTI-AGENT CONVERSATION TEST WITH COGNITIVE ACTION PROBES")
    print("="*80)

    # Configuration
    ai_1_model = "Gemma 3 4B (with Probes)"
    ai_2_model = "Gemma 3 4B (with Probes)"

    # Use the "Cognitive Roles - Analyst vs Creative" prompt pair
    prompt_pair = "Cognitive Roles - Analyst vs Creative"
    ai_1_prompt = SYSTEM_PROMPT_PAIRS[prompt_pair]["AI_1"]
    ai_2_prompt = SYSTEM_PROMPT_PAIRS[prompt_pair]["AI_2"]

    print(f"\nAI-1 Model: {ai_1_model}")
    print(f"AI-1 System Prompt: {ai_1_prompt[:100]}...")
    print(f"\nAI-2 Model: {ai_2_model}")
    print(f"AI-2 System Prompt: {ai_2_prompt[:100]}...")

    # Initialize conversation with a starting prompt
    conversation = [
        {
            "role": "user",
            "content": "How should we approach solving climate change?"
        }
    ]

    print("\n" + "-"*80)
    print("STARTING CONVERSATION")
    print("-"*80)
    print(f"\nInitial prompt: {conversation[0]['content']}")

    # Run 3 turns (AI-1, AI-2, AI-1)
    num_turns = 3
    for turn in range(num_turns):
        ai_name = "AI-1" if turn % 2 == 0 else "AI-2"
        model = ai_1_model if turn % 2 == 0 else ai_2_model
        system_prompt = ai_1_prompt if turn % 2 == 0 else ai_2_prompt

        print("\n" + "="*80)
        print(f"TURN {turn + 1}: {ai_name}")
        print("="*80)

        # Run the AI turn
        conversation = ai_turn(
            ai_name=ai_name,
            conversation=conversation,
            model=model,
            system_prompt=system_prompt,
            gui=None
        )

        # Display the latest response
        latest = conversation[-1]
        print(f"\n{ai_name} Response:")
        print("-" * 40)
        print(latest.get('content', ''))

        # Display cognitive action predictions if available
        if 'predictions' in latest:
            print("\n" + "-" * 40)
            print("COGNITIVE ACTIONS DETECTED:")
            print("-" * 40)
            predictions = latest['predictions']
            for i, pred in enumerate(predictions[:5], 1):  # Show top 5
                action = pred.get('action', 'Unknown')
                confidence = pred.get('confidence', 0.0)
                is_active = pred.get('is_active', False)
                print(f"  {i}. {action:35s} {confidence:6.1%}  {'[ACTIVE]' if is_active else ''}")

    # Save the full conversation
    output_file = Path(__file__).parent / "test_conversation_output.json"
    with open(output_file, 'w') as f:
        json.dump(conversation, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"Full conversation saved to: {output_file}")
    print("="*80)

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_multi_agent_conversation()