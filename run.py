from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from langchain.chains import create_tagging_chain_pydantic
from langchain.chains import LLMChain


def refine_sentence(sentence):
    re_sent = []
    text = ""
    for char in sentence:
        if char not in ['.', ',', '?', '!', ':', "'"]:
            text += char
    re_sent.append(text)
    return re_sent


sentences = ["Could you please take down my contact information?",
             "I'd like to provide my details",
             "Here are my details: Name, Phone Number, and Email.",
             "Can you record my contact information?",
             "Would you mind taking my contact details",
             "I'd appreciate it if you could save my contact",
             "Here are my contact details.",
             "Please take down my contact information.",
             "Hey, can you save my contact info?",
             "I'd like to give you my contact details",
             "Take my information",
             "Take my info",
             "Take my name, phone and email",
             "Can you take down my name, phone number, and email?",
             "Please note my contact details",
             "Can you help me by saving my contact info",
             "Can you take my contact information?",
             "store my contact details",
             "Could you store my contact details",
             "Here are my details",
             "Please take down my details",
             "take down my details",
             "save my contact information",
             "record my contact details",
             "I need to submit my contact details",
             "Please record my name, phone number, and email for contact purposes"]

r_sentences = [sentence.lower() for sentence in sentences]
re_sent = []
for sentence in r_sentences:
    new_sentence = refine_sentence(sentence)[0]
    re_sent.append(new_sentence)


class PersonalDetails(BaseModel):
    name: str = Field(
        ...,
        description="This is the name of the user.",
    )
    phone: str = Field(
        ...,
        description="This is the phone number of the user.",
    )
    email: str = Field(
        ...,
        description="This is the email of the user.",
    )

    class Config:
        orm_mode = True


def check_what_is_empty(user_personal_details):
    ask_for = []
    for field, value in user_personal_details.dict().items():
        if value in [None, "", 0]:
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for


def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details


def ask_for_info(llm, ask_for=['name', 'phone', 'email']):
    gathering_template = ChatPromptTemplate.from_template(
        """
        Your job is to ask user for their details: Name, Phone and Email. 
        You should ask user one question at a time even if you don't get all the info don't ask as a list! 
        Don't greet the user! Don't say Hi.
        Explain you need to get some info. If the ask_for list is empty thank them and ask them how you can
        help them. Ask user for the Name, Phone and Email.
        ### ask_for list: {ask_for}
        """
    )
    info_gathering_chain = LLMChain(llm=llm, prompt=gathering_template)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat


def filter_response(text_input, user_details, llm):
    chain = create_tagging_chain_pydantic(PersonalDetails, llm)
    res = chain.run(text_input)
    user_details = add_non_empty_details(user_details, res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for


def main():
    path = input("Enter path to the file: ")
    pdf_reader = PdfReader(path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    vectorstore = Chroma.from_texts(texts=chunks,
                                    embedding=OllamaEmbeddings(model="llama3", show_progress=True))

    user_details = PersonalDetails(name="", phone="", email="")
    ask_for = ["name", "phone", "email"]

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
                    You are an AI language model assistant. Your job is to answer user with the questions based upon
                    the document provided. The document text is stored in vector store. Do not answer question that
                    are not relevant to the provided document.
                """
    )

    template = """
                Answer the question based ONLY on the following: 
                context:
                {context}
                Question: {question}
                """

    llm = ChatOllama(model="llama3")
    retriever = MultiQueryRetriever.from_llm(
        vectorstore.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    prompt = ChatPromptTemplate.from_template(template=template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    try:
        while True:
            user_input = input("Ask: ")
            user_input = user_input.lower()
            user_input = refine_sentence(user_input)[0]
            if user_input in re_sent:
                while ask_for:
                    ai_chat = ask_for_info(llm, ask_for=ask_for)
                    print(ai_chat)
                    user_input = input("Ans: ")
                    user_details, ask_for = filter_response(user_input, user_details, llm)
                    ai_response = ask_for_info(llm, ask_for)
                    print(ai_response)
                print("Everything is gathered")
            else:
                response = chain.invoke(user_input)
                print("Bot: ", response)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
