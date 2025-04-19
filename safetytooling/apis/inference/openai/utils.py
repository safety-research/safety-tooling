import tiktoken

COMPLETION_MODELS = {
    "davinci-002",
    "babbage-002",
    "gpt-4-base",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo-instruct-0914",
}

EMBEDDING_MODELS = (
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)

VISION_MODELS = (
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
)

_GPT_4_MODELS = (
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1",
    "o1-2024-12-17",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
)
_GPT_3_MODELS = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
)


GPT_CHAT_MODELS = set(_GPT_4_MODELS + _GPT_3_MODELS)

OAI_FINETUNE_MODELS = (
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "davinci-002",
    "babbage-002",
)
S2S_MODELS = "gpt-4o-s2s"


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def get_max_context_length(model_id: str) -> int:
    if "gpt-4.1" in model_id:
        return 1_047_576
    elif "gpt-4o" in model_id:
        return 128_000
    elif "ft:gpt-3.5-turbo" in model_id:
        return 16_384

    match model_id:
        case "o1" | "o1-2024-12-17" | "o3-mini" | "o3-mini-2025-01-31":
            return 200_000
        case (
            "o1-mini"
            | "o1-mini-2024-09-12"
            | "o1-preview"
            | "o1-preview-2024-09-12"
            | "gpt-4-turbo"
            | "gpt-4-turbo-2024-04-09"
            | "gpt-4-0125-preview"
            | "gpt-4-turbo-preview"
            | "gpt-4-1106-preview"
            | "gpt-4-vision-preview"
        ):
            return 128_000

        case "gpt-4" | "gpt-4-0613" | "gpt-4-base":
            return 8192

        case "gpt-4-32k" | "gpt-4-32k-0613":
            return 32_768

        case "gpt-3.5-turbo-0613":
            return 4096

        case (
            "gpt-3.5-turbo"
            | "gpt-3.5-turbo-1106"
            | "gpt-3.5-turbo-0125"
            | "gpt-3.5-turbo-16k"
            | "gpt-3.5-turbo-16k-0613"
            | "babbage-002"
            | "davinci-002"
        ):
            return 16_384

        case "gpt-3.5-turbo-instruct" | "gpt-3.5-turbo-instruct-0914":
            return 4096

        case _:
            raise ValueError(f"Invalid model id: {model_id}")


def get_rate_limit(model_id: str) -> tuple[int, int]:
    """
    Returns the (tokens per min, request per min) for the given model id.
    # go to: https://platform.openai.com/settings/organization/limits
    """
    if "gpt-4o-mini" in model_id or "gpt-4.1-nano" in model_id:
        return 150_000_000, 30_000
    elif "gpt-4o" in model_id or "gpt-4.1" in model_id:
        return 30_000_000, 10_000
    elif "ft:gpt-3.5-turbo" in model_id:
        return 50_000_000, 10_000

    match model_id:
        case "o1" | "o1-2024-12-17":
            return 30_000_000, 1_000
        case "o1-mini" | "o1-mini-2024-09-12" | "o3-mini" | "o3-mini-2025-01-31":
            return 150_000_000, 30_000
        case "o1-preview" | "o1-preview-2024-09-12":
            return 30_000_000, 10_000
        case (
            "gpt-4-turbo"
            | "gpt-4-turbo-2024-04-09"
            | "gpt-4-0125-preview"
            | "gpt-4-turbo-preview"
            | "gpt-4-1106-preview"
        ):
            return 2_000_000, 10_000
        case "gpt-4" | "gpt-4-0314" | "gpt-4-0613":
            return 1_000_000, 10_000
        case "gpt-4-32k" | "gpt-4-32k-0314" | "gpt-4-32k-0613":
            return 150_000, 1_000
        case "gpt-4-base":
            return 10_000, 200
        case "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-16k":
            return 50_000_000, 10_000
        case "gpt-3.5-turbo-instruct" | "gpt-3.5-turbo-instruct-0914":
            return 90_000, 3_500
        case "babbage-002" | "davinci-002" | "text-davinci-003" | "text-davinci-002":
            return 250_000, 3_000
        case _:
            return 1_000_000, 10_000  # default to smaller rate limit for unknown models


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    Go to: https://platform.openai.com/docs/pricing
    """

    # Prices not listed yet
    if model_id in (
        "o1",
        "o1-2024-12-17",
        "o1-preview",
        "o1-preview-2024-09-12",
    ):
        prices = 15, 60
    elif model_id in ("o3-mini", "o3-mini-2025-01-31"):
        prices = 1.10, 4.40
    elif model_id in ("o1-mini", "o1-mini-2024-09-12"):
        prices = 1.10, 4.40
    elif model_id in (
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
    ):
        prices = 0.1, 0.4
    elif model_id in (
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
    ):
        prices = 0.4, 1.6
    elif model_id in (
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
    ):
        prices = 2, 8
    elif model_id in (
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-mini-2024-08-06",
    ):
        prices = 0.15, 0.6
    elif model_id in (
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
    ):
        prices = 2.5, 10
    elif model_id in (
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
    ):
        prices = 10, 30
    elif model_id == "gpt-3.5-turbo-0125":
        prices = 0.5, 1.5
    elif model_id == "gpt-3.5-turbo-1106":
        prices = 1, 2
    elif model_id.startswith("gpt-4-32k"):
        prices = 60, 120
    elif model_id.startswith("gpt-4"):
        prices = 30, 60
    elif model_id.startswith("gpt-3.5-turbo-16k"):
        prices = 3, 4
    elif model_id.startswith("gpt-3.5-turbo"):
        prices = 1.5, 2
    elif model_id == "davinci-002":
        prices = 2, 2
    elif model_id == "babbage-002":
        prices = 0.4, 0.4
    elif model_id == "text-davinci-003" or model_id == "text-davinci-002":
        prices = 20, 20
    elif "ft:gpt-3.5-turbo" in model_id:
        prices = 3, 6
    elif "ft:gpt-4.1-mini" in model_id:
        prices = 0.8, 3.2
    elif "ft:gpt-4.1" in model_id:
        prices = 3, 12
    elif "ft:gpt-4o-mini" in model_id:
        prices = 0.3, 1.2
    elif "ft:gpt-4o" in model_id:
        prices = 3.75, 15
    elif model_id == "text-embedding-3-small":
        prices = 0.02, 0.02
    elif model_id == "text-embedding-3-large":
        prices = 0.13, 0.13
    elif model_id == "text-embedding-ada-002":
        prices = 0.10, 0.10
    elif model_id == "gpt-3.5-turbo-instruct" or model_id == "gpt-3.5-turbo-instruct-0914":
        prices = 1.5, 2
    else:
        prices = 0, 0  # default to 0 price for unknown models

    return tuple(price / 1_000_000 for price in prices)


def finetune_price_per_token(model_id: str) -> float | None:
    """
    Returns price per 1000 tokens.
    Returns None if the model does not support fine-tuning.

    See https://openai.com/pricing and
    https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned
    for details.
    """
    if "gpt-4.1-mini" in model_id:
        return 0.005
    elif "gpt-4o-mini" in model_id:
        return 0.003
    elif "gpt-4o" in model_id or "gpt-4.1-" in model_id:
        return 0.025
    elif "ft:gpt-3.5-turbo" in model_id:
        return 0.008

    match model_id:
        case "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo":
            return 0.008
        case "davinci-002":
            return 0.006
        case "babbage-002":
            return 0.0004
        case _:
            return None
