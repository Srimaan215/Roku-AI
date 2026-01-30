"""Test multi-turn conversation and profile updates."""
from core.personalized_roku import PersonalizedRoku

def test_multi_turn():
    print('='*60)
    print('TEST 1: Multi-Turn Conversation')
    print('='*60)
    
    roku = PersonalizedRoku(username='Srimaan', verbose=False)
    history = []

    # Turn 1
    msg1 = "Hey Roku, I'm feeling a bit overwhelmed today."
    print(f'\nUser: {msg1}')
    resp1 = roku.chat(msg1, history=history, max_tokens=100)
    print(f'Roku: {resp1}')
    history.append({'role': 'user', 'content': msg1})
    history.append({'role': 'assistant', 'content': resp1})

    # Turn 2 - references previous context
    msg2 = "What should I prioritize given my goals?"
    print(f'\nUser: {msg2}')
    resp2 = roku.chat(msg2, history=history, max_tokens=100)
    print(f'Roku: {resp2}')
    history.append({'role': 'user', 'content': msg2})
    history.append({'role': 'assistant', 'content': resp2})

    # Turn 3 - tests if it remembers the conversation + profile
    msg3 = "Can you remind me what my current research project is about?"
    print(f'\nUser: {msg3}')
    resp3 = roku.chat(msg3, history=history, max_tokens=100)
    print(f'Roku: {resp3}')


if __name__ == "__main__":
    test_multi_turn()
