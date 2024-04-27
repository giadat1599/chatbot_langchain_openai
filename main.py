
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

#----- Store previous messages -------
memory = ConversationBufferMemory(

    # Key to store previous message
    memory_key="messages", 

    # Return an instance of message instead of a string
    return_messages=True,

    # Save messages to a file and load it when the app is started
    chat_memory=FileChatMessageHistory("messages.json")
)

#----- Store a summarized version of previous messages instead of all messages -------
# memory = ConversationSummaryMemory(
#     # Key to store previous message
#     memory_key="messages", 

#     # Return an instance of message instead of a string
#     return_messages=True,

#     # LLM for the memory to run its own chain 
#     llm=chat
# )

prompt = ChatPromptTemplate(
    # Memory will add an additional variable to the chain input called "messages" (We set up in the memory)
    input_variables=["content", "messages"],
    messages=[
        # Append previous messages
        MessagesPlaceholder(variable_name="messages"),

        # User message
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)


while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])