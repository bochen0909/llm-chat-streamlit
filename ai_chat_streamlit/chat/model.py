import os
from openai import OpenAI
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms.ollama import Ollama
import subprocess
from botocore.exceptions import TokenRetrievalError, UnauthorizedSSOTokenError, ClientError


def _st_message_to_langchain_message(message):
    if message["role"] == "user":
        return HumanMessage(content=message["content"])
    elif message["role"] == "assistant":
        return AIMessage(content=message["content"])
    elif message["role"] == "system":
        return SystemMessage(content=message["content"])
    else:
        raise ValueError(f"Unknown role: {message['role']}")


class OllamaModel:
    def __init__(self, model_id: str):
        self.client = Ollama(model=model_id)

    def reset(self):
        pass

    def stream(self, messages):
        return self.client.stream(
            [_st_message_to_langchain_message(m) for m in messages]
        )


class ChatGPTModel:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def reset(self):
        pass

    def stream(self, messages):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )



class ChatBedrockModel:
    def __init__(self, credentials_profile_name: str, model_id: str, sso: bool = False, region: str = None, stream: bool = True):
        self.credentials_profile_name = credentials_profile_name
        self.model_id = model_id
        self.sso = sso
        self.region = region or "us-east-1"
        self._stream_mode = stream
        self._init_client()

    def _init_client(self):
        if self.sso:
            subprocess.run(["aws", "sso", "login", "--profile", self.credentials_profile_name])
        self.client = ChatBedrock(
            credentials_profile_name=self.credentials_profile_name,
            model_id=self.model_id,
            streaming=self._stream_mode,
            region=self.region,
            model_kwargs={"temperature": 0.1, "max_tokens": 20000},
        )

    def reset(self):
        self._init_client()

    def stream(self, messages):
        msgs = [_st_message_to_langchain_message(m) for m in messages]
        if self._stream_mode:
            return self.client.stream(msgs)
        else:
            # invoke returns a single response, wrap in a generator for compatibility
            result = self.client.invoke(msgs)
            def gen():
                yield result.content
            return gen()


def create_model(_: str, model_config: dict):
    if model_config["type"] == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return ChatGPTModel(api_key=api_key, model=model_config["model_id"])
    elif model_config["type"] == "bedrock":
        sso = model_config.get("sso", False)
        region = model_config.get("region", 'us-east-1')
        stream = model_config.get("stream", True)
        return ChatBedrockModel(
            credentials_profile_name=model_config["credentials_profile_name"],
            model_id=model_config["model_id"],
            sso=sso,
            region=region,
            stream=stream,
        )
    elif model_config["type"] == "ollama":
        return OllamaModel(model_id=model_config["model_id"])
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
