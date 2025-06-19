# summarizer.py

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_contract(text, sentence_count=5):
    """
    Summarizes contract text using Sumy's TextRank.

    Args:
        text (str): Full contract text.
        sentence_count (int): Number of summary sentences to return.

    Returns:
        str: Summarized contract text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)

    return " ".join([str(sentence) for sentence in summary]) or "Summary could not be generated."
