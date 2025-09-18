# CodSoft Internship - Task 1
# Rule-Based Chatbot in Python

# Predefined patterns and responses
data = {
    "hi": "Hi there! I'm a friendly chatbot here to assist you 😊",
    "hello": "Hello! How can I help you today?",
    "who are you": "I'm just a chatbot created for CodSoft Internship Task 1.",
    "what is your name": "I don't have a real name, but you can call me ChatBot 🤖",
    "where are you from": "I'm from the digital world, always ready to chat!",
    "how are you": "I'm just a chatbot, but I'm here to assist you!",
    "do you have any hobbies": "I love chatting with people like you and answering questions!",
    "what did you eat today": "I don't eat food, but I can suggest you some tasty recipes 😋",
    "what is your favorite color": "I don’t have personal preferences, but I think all colors are beautiful 🌈",
    "do you listen to music": "I can’t listen to music, but I know many people enjoy it 🎵",
    "bye": "Goodbye! Take care and have a wonderful day! 👋"
}

# Function to get chatbot response
def get_response(user_input):
    user_input = user_input.lower()
    for pattern, response in data.items():
        if pattern in user_input:
            return response
    return "I'm sorry, I didn’t understand that. Can you please rephrase your sentence?"

# Main program
print("Chatbot 🤖: Hi! I'm a simple chatbot created for CodSoft Internship Task 1. Type 'bye' to exit.")

while True:
    user = input("You: ")
    response = get_response(user)
    print("Chatbot 🤖:", response)
    if "bye" in user.lower():
        break
