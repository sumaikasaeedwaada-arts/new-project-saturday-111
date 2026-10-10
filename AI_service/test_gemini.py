from google import genai

from AI_service.ai_service import MODEL, SYSTEM_PROMPT


def main():
    client = genai.Client()
    print("AI Chatbot started!")
    print("Type 'exit' to stop.")

    while True:
        message = input("You: ").strip()
        if message.lower() == "exit":
            print("Bot: Bye!")
            break

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=message,
                config={"system_instruction": SYSTEM_PROMPT}
            )
            print("Bot:", response.text)
        except Exception as exc:
            print("Server/API error:", exc)


if __name__ == "__main__":
    main()
