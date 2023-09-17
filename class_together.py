from decouple import config
import together
from langchain import PromptTemplate, LLMChain
from typing import Any, Dict, List, Mapping, Optional
#pydantic model version>2 will not work with langchain. you should install a model version less than 2
from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
import re

together.api_key = config("TOGETHER_API_KEY")

together.Models.start("togethercomputer/llama-2-70b-chat")

class TogetherLLM(LLM):
    """Together large language models."""

    model = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key = config("TOGETHER_API_KEY")
    """Together API key"""

    temperature = 0.7
    """What sampling temperature to use."""

    max_tokens = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self):
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt,
        **kwargs: Any,
    ):
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        # Use regex substitution to remove newlines
        #cleaned_text = re.sub(r"\n", "", text)
        return text

tog_llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature=0.1,
    max_tokens=1024)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

# system_prompt = "You are an advanced assistant that excels at translation. "
# instruction = "Convert the following text from English to French:\n\n {text}"
# template = get_prompt(instruction, system_prompt)
# print(template)
system_prompt = "You are an advanced assistant who is very good at logical thinking and mathematical explanation. "
instruction = "Answer the question logically that is asked in the text : \n\n {text}"
template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt= prompt, llm = tog_llm )

text = "how many vowels are there in the days of a week?"
output = llm_chain.run(text)

print(output)

