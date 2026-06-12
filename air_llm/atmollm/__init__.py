from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx, AutoModel
    from airllm.utils import NotEnoughSpaceException, split_and_save_layers
else:
    from airllm import (
        AirLLMLlama2,
        AirLLMChatGLM,
        AirLLMQWen,
        AirLLMQWen2,
        AirLLMBaichuan,
        AirLLMInternLM,
        AirLLMMistral,
        AirLLMMixtral,
        AirLLMBaseModel,
        AutoModel,
    )
    from airllm.utils import NotEnoughSpaceException, split_and_save_layers
