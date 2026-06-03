import html
import string


def fully_unescape_html(text: str) -> str:
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = html.unescape(text)
    return text


PUNCT_TO_STRIP = string.punctuation + r"“”‘’´`–—…"


def strip_punctuation(text: str) -> str:
    return text.strip(PUNCT_TO_STRIP)
