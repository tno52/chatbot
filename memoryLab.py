from langchain_ollama.chat_models import ChatOllama
from IPython.display import display, Markdown
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Initialize the language model
llm = ChatOllama(model="huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest")

# Basic prompt with memory example
print("=== Basic Memory Example ===")
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="message"),
        HumanMessage(content="Can you repeat exactly your answer?")
    ]
)

variables = {
    "message": [
        HumanMessage(content="What does IU stand for as in university?"),
        AIMessage(content="Indiana University.")
    ],
}

formatted_prompt = prompt.invoke(variables)
print("Formatted prompt:", formatted_prompt)

chain = prompt | llm
response = chain.invoke(variables)
print("Response:", response.content)

print("\n" + "="*50 + "\n")

# Advanced memory with conversation history
print("=== Advanced Memory with Conversation History ===")

# Define base prompt
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Answer all questions clearly."),
        MessagesPlaceholder(variable_name="history"),   # stores prior messages
        MessagesPlaceholder(variable_name="input"),     # current turn
    ]
)

# Create chain
chain = prompt | llm

# Define memory store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with message history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",   # matches MessagesPlaceholder
    input_messages_key="input",       # matches MessagesPlaceholder
)

# Simulate conversation session
session_id = "user123"

# First turn
print("First question:")
response1 = chain_with_history.invoke(
    {"input": [HumanMessage(content="What does IU stand for as in university?")]},
    config={"configurable": {"session_id": session_id}},
)
print("Response:", response1.content)

# Check stored messages
print("\nStored messages:")
for msg in store['user123'].messages:
    print(f"{type(msg).__name__}: {msg.content}")

# Second turn - model sees previous context
print("\nSecond question:")
response2 = chain_with_history.invoke(
    {"input": [HumanMessage(content="Can you repeat exactly your answer?")]},
    config={"configurable": {"session_id": session_id}},
)
print("Response:", response2.content)

print("\n" + "="*50 + "\n")

# Memory trimming example
print("=== Memory Trimming Example ===")

trimmer = trim_messages(
    strategy="last",
    max_tokens=2,
    token_counter=len,
    include_system=True
)

chain = (
    RunnablePassthrough.assign(
        history=lambda x: trimmer.invoke(x["history"])
    )
    | prompt
    | llm
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",
    input_messages_key="input",
)

questions = [
    "Hey, my name is James",
    "What should I eat when the weather is hot?",
    "What does IU stand for as in university?"
]

for question in questions:
    # Ask a question
    chain_with_history.invoke(
        {"input": [HumanMessage(content=question)]},
        config={"configurable": {"session_id": "user145"}},
    )
    # Then ask for name
    res = chain_with_history.invoke(
        {"input": [HumanMessage(content="What is my name?")]},
        config={"configurable": {"session_id": "user145"}},
    )
    print(f"After question: '{question}'")
    print(f"Name response: {res.content}")
    print()

print("\n" + "="*50 + "\n")

# Summary memory example
print("=== Summary Memory Example ===")

chat_history = [
    HumanMessage(content="Hey there! I'm a student at IU."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks!"),
    HumanMessage(content="Do you know anything about IU?"),
    AIMessage(content="Indiana University is in Bloomingon, Indiana")
]

summary_prompt = (
    "Distill the above chat messages into a single summary message. "
    "Include as many specific details as you can."
)
summary_message = llm.invoke(
    chat_history + [HumanMessage(content=summary_prompt)]
)

print("Summary of conversation:")
print(summary_message.content)

new_turns = [
    HumanMessage(content="What programs is IU known for?"),
    AIMessage(content="It's well known for business, music, and computer science programs."),
]

# More concrete implementation
def summarize_history_if_long(history, llm, summary_message=None, max_len=3):
    """If the history is long, summarize it into one message."""
    print(f"History length: {len(history)}")
    if len(history) > max_len:
        summary_prompt = (
            "Summarize the following conversation so far into one concise message.\n"
            "Focus on key facts, roles, and context.\n\n"
            f"{[m.content for m in history]}"
        )
        summary_message = llm.invoke([HumanMessage(content=summary_prompt)])
        # Return a new short history that keeps only summary
        print("HISTORY:", history)
        return [AIMessage(content=summary_message.content)]
    
    return history

chain = (
    RunnablePassthrough.assign(
        history=lambda x: summarize_history_if_long(x["history"], llm)
    )
    | prompt
    | llm
)

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",
    input_messages_key="input",
)

# Test the summarization
for question in questions:
    chain_with_history.invoke(
        {"input": [HumanMessage(content=question)]},
        config={"configurable": {"session_id": "user170"}},
    )
    res = chain_with_history.invoke(
        {"input": [HumanMessage(content="What is my name?")]},
        config={"configurable": {"session_id": "user170"}},
    )
    print(f"After question: '{question}'")
    print(f"Name response: {res.content}")
    print()

# Check final stored messages
print("Final stored messages for user170:")
for i in chain_with_history.get_session_history('user170').messages:
    print(f"{type(i).__name__}: {i.content}")

print("\nTODO: Change the code to combine summary + last few messages")