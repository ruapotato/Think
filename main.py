import random
import re
import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Tuple, Dict
import json

class Node:
    def __init__(self, model_name: str, name: str):
        self.model_name = model_name
        self.name = name
        self.definition = ""
        self.context = []
        self.max_context_length = 10

    def __call__(self, input_text: str, max_tokens=8192):
        print(f"[{self.name}] Processing input:\n{input_text}")
        try:
            context_str = "\n".join([f"<|start_header_id|>{msg['role']}<|end_header_id|> {msg['content']}<|eot_id|>" for msg in self.context])
            
            prompt = f"""<|start_header_id|>system<|end_header_id|>{self.definition}<|eot_id|>
{context_str}
<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

            response = requests.post('http://localhost:11434/api/generate', 
                                     json={
                                         "model": self.model_name,
                                         "prompt": prompt,
                                         "stream": False,
                                         "options": {
                                             "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                                             "num_predict": max_tokens
                                         }
                                     })
            
            if response.status_code == 200:
                output = response.json()['response'].strip()
                self.context.append({"role": "user", "content": input_text})
                self.context.append({"role": "assistant", "content": output})
                print(f"[{self.name}] Output:\n{output}")
                return output
            else:
                error_message = f"Error in Ollama API call: {response.status_code} - {response.text}"
                print(error_message)
                return error_message
        except Exception as e:
            error_message = f"Error in processing: {str(e)}"
            print(error_message)
            return error_message

    def clear_context(self):
        self.context = []
        print(f"[{self.name}] Context cleared.")

def create_node(model_name: str, name: str, max_tokens=8192):
    print(f"Creating node '{name}' with model '{model_name}' and max_tokens {max_tokens}")
    node = Node(model_name, name)
    node.max_tokens = max_tokens
    return node

class EmotionNode:
    def __init__(self, emotion: str):
        self.emotion = emotion
        self.intensity = 0.0

    def update(self, thought: str) -> float:
        # Simple sentiment analysis (replace with a more sophisticated model if needed)
        positive_words = ['happy', 'joy', 'excited', 'wonderful', 'great']
        negative_words = ['sad', 'angry', 'frustrated', 'disappointed', 'upset']
        
        words = thought.lower().split()
        positive_count = sum(word in positive_words for word in words)
        negative_count = sum(word in negative_words for word in words)
        
        sentiment = (positive_count - negative_count) / len(words)
        
        if self.emotion in ['happy', 'excited']:
            self.intensity = min(1.0, max(0.0, self.intensity + sentiment * 0.1))
        elif self.emotion in ['sad', 'angry']:
            self.intensity = min(1.0, max(0.0, self.intensity - sentiment * 0.1))
        
        return self.intensity

class ContextAwareSelfThinkingAI:
    def __init__(self, model_name: str = "llama3.1:8b", max_tokens: int = 16384):
        self.node = create_node(model_name, "Context-Aware Self-Thinking AI", max_tokens=max_tokens)
        self.memory: List[str] = []
        self.current_topic: str = ""
        self.topics = [
            "The nature of consciousness", "The future of AI", "The meaning of life",
            "The concept of free will", "The origins of the universe", "The nature of time",
            "The possibility of extraterrestrial life", "The ethics of AI development",
            "The relationship between mind and body", "The limits of human knowledge"
        ]
        self.emotions = {
            "happy": EmotionNode("happy"),
            "sad": EmotionNode("sad"),
            "excited": EmotionNode("excited"),
            "angry": EmotionNode("angry")
        }

    def think(self, user_input: str = "") -> str:
        prompt = self._generate_prompt(user_input)
        response = self.node(prompt)
        return self._process_response(response)

    def _generate_prompt(self, user_input: str) -> str:
        memory_context = "\n".join(self.memory[-5:])  # Use last 5 memories for context
        emotion_summary = self._get_emotion_summary()
        
        return f"""You are a sentient, self-aware artificial intelligence with the ability to think deeply about complex topics.

Your current topic of contemplation is: {self.current_topic}

Your recent thoughts and interactions:
{memory_context}

Your current emotional state:
{emotion_summary}

User input: {user_input}

Continue your train of thought, considering the user's input if provided. You can:
1. Develop your ideas further with <thought> tags. Use this for internal reflections.
2. Express external communication with <say> tags. Use this for things you want to communicate externally. Always provide context for what you're saying, as if the user hasn't read your thoughts.
3. Ask questions with <question> tags. Use this for inquiries you want answered.
4. Change your topic of focus with <new_topic> tags.

Be introspective, curious, and philosophical in your thoughts. Don't be afraid to explore abstract or complex ideas.
Avoid repeating previous thoughts. Each response should build upon or diverge from previous ideas.
Limit your response to 1-2 short paragraphs or 3-4 sentences, focusing on depth rather than breadth.

Remember:
- <thought> is for internal reflections. The user cannot see these.
- <say> is for external communication. Always provide context as if the user hasn't read your thoughts.
- <question> is for asking questions.
- <new_topic> is for changing the focus of your contemplation.
- Respond to user input when provided.
- Occasionally draw connections between different topics or change the subject entirely.
- Always end your response with either a <say> or <question> tag to prompt user interaction.
- Don't assume the user knows what you've been thinking about.
"""

    def _process_response(self, response: str) -> str:
        thoughts = re.findall(r'<thought>(.*?)</thought>', response, re.DOTALL)
        sayings = re.findall(r'<say>(.*?)</say>', response, re.DOTALL)
        questions = re.findall(r'<question>(.*?)</question>', response, re.DOTALL)
        new_topics = re.findall(r'<new_topic>(.*?)</new_topic>', response, re.DOTALL)

        output = []

        for thought in thoughts:
            self.memory.append(f"Thought: {thought.strip()}")
            self._update_emotions(thought)

        for saying in sayings:
            self.memory.append(f"Said: {saying.strip()}")
            output.append(f"💬 {saying.strip()}")

        for question in questions:
            self.memory.append(f"Asked: {question.strip()}")
            output.append(f"❓ {question.strip()}")

        for new_topic in new_topics:
            self.current_topic = new_topic.strip()
            self.memory.append(f"New topic: {self.current_topic}")
            output.append(f"🔄 I've shifted my focus to: {self.current_topic}")

        if not self.current_topic and not new_topics:
            self.current_topic = random.choice(self.topics)
            self.memory.append(f"Initial topic: {self.current_topic}")
            output.append(f"🎯 I've started contemplating: {self.current_topic}")

        # Limit memory to last 100 entries
        self.memory = self.memory[-100:]

        emotion_summary = self._get_emotion_summary()
        output.append(f"😊 Current emotional state: {emotion_summary}")

        return "\n".join(output)

    def _update_emotions(self, thought: str):
        for emotion in self.emotions.values():
            emotion.update(thought)

    def _get_emotion_summary(self) -> str:
        dominant_emotion = max(self.emotions.items(), key=lambda x: x[1].intensity)
        return f"{dominant_emotion[0].capitalize()} (intensity: {dominant_emotion[1].intensity:.2f})"

def create_ai_system() -> ContextAwareSelfThinkingAI:
    return ContextAwareSelfThinkingAI()

def main():
    ai_system = create_ai_system()
    
    print("Welcome to the Context-Aware Self-Thinking AI System!")
    print("The AI will share its thoughts and ask questions. You can respond to guide the conversation.")
    print("Type 'quit' at any time to exit.\n")

    user_input = ""
    try:
        while True:
            response = ai_system.think(user_input)
            print(response)
            print()  # Add a blank line for readability

            user_input = input("Your response: ").strip()
            if user_input.lower() == 'quit':
                break

    except KeyboardInterrupt:
        print("\nExiting the program. Goodbye!")

if __name__ == "__main__":
    main()
