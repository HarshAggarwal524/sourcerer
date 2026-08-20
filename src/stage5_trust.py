from src.config import LLM_MODEL
from src.llm import generate_llm


def check_grounding(question, context_chunks, answer, model=LLM_MODEL):
    """
    LLM-as-judge: checks whether the generated answer is fully supported
    by the source chunks, with no information added from outside them.

    Returns:
        "SUPPORTED" or "NOT_SUPPORTED"

    context_chunks:
        Either a single string or a list of strings.
    """

    if isinstance(context_chunks, list):
        context = "\n\n---\n\n".join(context_chunks)
    else:
        context = context_chunks

    prompt = (
        "You are a strict fact-checker. Your job is to determine whether an answer "
        "is fully supported by a given source passage.\n\n"
        "Rules:\n"
        "- Reply SUPPORTED only if every specific claim in the answer is directly "
        "stated in the source passage.\n"
        "- Reply NOT_SUPPORTED if the answer contains any information, inference, "
        "or detail not explicitly present in the source passage.\n"
        "- Reply NOT_SUPPORTED if the answer says there is insufficient information "
        "or that the question cannot be answered.\n"
        "- Reply with ONLY the word SUPPORTED or NOT_SUPPORTED. Nothing else.\n\n"
        f"Source passage:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )

    try:
        verdict = generate_llm(prompt, model).strip().upper()

        if verdict not in ("SUPPORTED", "NOT_SUPPORTED"):
            return "NOT_SUPPORTED"

        return verdict

    except Exception as e:
        print(f"[stage5_trust.py] Grounding check failed: {e}")
        return "NOT_SUPPORTED"


def generate_trusted_answer(
    question,
    context_chunks,
    generate_fn,
    model=LLM_MODEL
):
    """
    Generates an answer and verifies it against the source chunks.

    Returns:
        (final_answer, confidence_tag, verdict)
    """

    answer = generate_fn(question, context_chunks)

    if not answer:
        return (
            "Could not generate an answer.",
            "LOW CONFIDENCE",
            "NOT_SUPPORTED"
        )

    verdict = check_grounding(
        question,
        context_chunks,
        answer,
        model=model
    )

    if verdict == "SUPPORTED":
        return (
            answer,
            "HIGH CONFIDENCE",
            verdict
        )

    return (
        "I could not find sufficient information in the source text "
        "to answer this question confidently.",
        "LOW CONFIDENCE",
        verdict
    )