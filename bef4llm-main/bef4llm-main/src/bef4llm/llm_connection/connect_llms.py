from ollama import Client
from copy import deepcopy
from pydantic import BaseModel


class ConnectLLMs():
    """
    Class to connect to an LLM via Ollama API
    """
    def __init__(self, llm_modell, sys_msg, timeout=300):
        """
        Initalizes the class
        Parameters
        ----------
        llm_modell: str
            LLM model (ollama tag is neede)
        sys_msg: str
           System message
        timeout: int
            Timeout in seconds
        """
        self.client = Client(
            # please modify the host address
            host="http://ip.port",
            timeout=timeout
        )

        self.llm = llm_modell
        self.chat_histroy = []
        self.sys_msg = sys_msg
        self.init_sys_role(sys_msg)

        print(f"LLM {self.llm} is ready. The timeout is set to {timeout}")

    def chat_with_history(self, role:str, content:str, format=None):
        """
        Allows to send a request to the LLM and returns the response.
        The conversation is saved in the history

        Parameters
        ----------
        role: str
            Role of the request
        content: str
            Content of the request
        format: dict
            Desired format of the response - not mandatory

        Returns
        -------
        response: str
            Content string of response from the LLM
        """
        self.chat_histroy.append({"role": "user", "content": content})
        if format:
            response = self.client.chat(model=self.llm,
                                        messages=self.chat_histroy,
                                        format=format,
                                        options={"temperature": 0.1, "num_ctx": 40000})  # numctx = context
        else:
            response = self.client.chat(model=self.llm,
                                    messages=self.chat_histroy,
                                    options={"temperature": 0.1, "num_ctx": 40000}) #numctx = context

        self.chat_histroy.append({"role": "assistant", "content": response.message.content})
        return response.message.content

    def chat_without_history(self, role:str, content:str, format=None):
        """
        Allows to send a request to the LLM and returns the response.
        The conversation is NOT saved in the history

        Parameters
        ----------
        role: str
            Role of the request
        content: str
            Content of the request
        format: dict
            Desired format of the response - not mandatory

        Returns
        -------
        response: dict
            Content string of response from the LLM
        """
        # history only consists of the system message
        msg = deepcopy(self.chat_histroy)
        msg.append({"role": role, "content": content})
        if format:
            response = self.client.chat(model=self.llm,
                                        messages=msg,
                                        format=format,
                                        options={"temperature": 0.1, "num_ctx": 40000})
        else:
            response = self.client.chat(model=self.llm,
                                        messages=msg,
                                        options={"temperature": 0.1, "num_ctx": 40000})
        return response.message.content

    def init_sys_role(self, msg:str):
        """
        Allows to change the system message - history is deleted

        Parameters
        ----------
        msg: str
            new system message

       """
        self.chat_histroy = []
        self.chat_histroy.append({"role": "system", "content": msg})
        response = self.client.chat(model=self.llm, messages=self.chat_histroy)
        self.chat_histroy.append({"role": "assistant", "content": response.message.content})

    def reset_chat_history(self):
        """
        deletes the chat history, without deleting the system message
        """
        if len(self.chat_histroy) > 1:
            self.chat_histroy = self.chat_histroy[:2]


