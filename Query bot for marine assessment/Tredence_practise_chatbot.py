
############# Using normal rule-based approach ##################
import time

# Dummy question-answer pairs
knowledge_base = {
    "Q1": "What is the sea worthiness criteria for vessels?",
    "A1": "The sea worthiness criteria for vessels include factors such as structural integrity, equipment functionality, navigation systems, and adherence to safety regulations.",
    
    "Q2": "What are the common maintenance procedures for ships?",
    "A2": "Common maintenance procedures for ships involve regular inspections, hull cleaning, engine maintenance, and safety equipment checks.",
    
    "Q3": "What are the regulations specific to vessels operating in the European waters?",
    "A3": "Vessels operating in European waters must comply with the regulations set by the European Maritime Safety Agency (EMSA) and International Maritime Organization (IMO).",
    
    "Q4": "How can we ensure compliance with safety standards for different vessel types?",
    "A4": "To ensure compliance with safety standards, surveyors must conduct thorough inspections and audits, follow industry best practices, and stay updated with the latest regulations.",
}

def chatbot():
    print("Welcome to the Marine Assessment Chatbot!")
    while True:
        print("\nPlease select a question number:")
        for i in range(1, len(knowledge_base)//2 + 1):
            question = knowledge_base[f"Q{i}"]
            print(f"{i}. {question}")

        try:
            user_input = int(input("Enter the question number (0 to exit): "))
        except ValueError:
            print("Invalid input. Please enter a valid question number.")
            continue

        if user_input == 0:
            print("Thank you for using the Marine Assessment Chatbot. Goodbye!")
            break

        if 1 <= user_input <= len(knowledge_base)//2:
            question_key = f"Q{user_input}"
            answer_key = f"A{user_input}"
            answer = knowledge_base[answer_key]
            print("\nFetching the answer...\n")
            time.sleep(1)
            print("Chatbot: " + answer)
        else:
            print("Invalid question number. Please select a valid question or enter 0 to exit.")

if __name__ == "__main__":
    chatbot()


####################### Using tranformers questions answering (BERT Large uncased) #########

from transformers import pipeline

# Dummy question-answer pairs
knowledge_base = {
    "What is the sea worthiness criteria for vessels?": "The sea worthiness criteria for vessels include factors such as structural integrity, equipment functionality, navigation systems, and adherence to safety regulations.",
    "What are the common maintenance procedures for ships?": "Common maintenance procedures for ships involve regular inspections, hull cleaning, engine maintenance, and safety equipment checks.",
    "What are the regulations specific to vessels operating in the European waters?": "Vessels operating in European waters must comply with the regulations set by the European Maritime Safety Agency (EMSA) and International Maritime Organization (IMO).",
    "How can we ensure compliance with safety standards for different vessel types?": "To ensure compliance with safety standards, surveyors must conduct thorough inspections and audits, follow industry best practices, and stay updated with the latest regulations.",
}

def chatbot():
    print("Welcome to the Marine Assessment Chatbot!")
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

    while True:
        print("\nPlease select a question number:")
        for i, question in enumerate(knowledge_base.keys(), start=1):
            print(f"{i}. {question}")

        try:
            user_input = int(input("Enter the question number (0 to exit): "))
        except ValueError:
            print("Invalid input. Please enter a valid question number.")
            continue

        if user_input == 0:
            print("Thank you for using the Marine Assessment Chatbot. Goodbye!")
            break

        if 1 <= user_input <= len(knowledge_base):
            question = list(knowledge_base.keys())[user_input - 1]
            context = list(knowledge_base.values())[user_input - 1]

            answer = qa_pipeline(question=question, context=context)
            print("\nChatbot:", answer["answer"])
        else:
            print("Invalid question number. Please select a valid question or enter 0 to exit.")

if __name__ == "__main__":
    chatbot()
