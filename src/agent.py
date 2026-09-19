from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức (kho dữ liệu rỗng)."

        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        # ponytail: numbered citations [1], [2] satisfy Source Traceability requirement without extra libraries
        context_blocks: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source") or chunk.get("id", f"chunk-{i}")
            context_blocks.append(f"[{i}] (Nguồn: {source}):\n{chunk.get('content', '')}")
        context_str = "\n\n".join(context_blocks)

        prompt = (
            "Bạn là trợ lý giải đáp thông tin dựa trên ngữ cảnh được cung cấp.\n"
            "Chỉ sử dụng thông tin trong ngữ cảnh dưới đây để trả lời câu hỏi. "
            "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời là không tìm thấy, tuyệt đối không bịa đặt.\n"
            "Khi trích dẫn thông tin, hãy ghi rõ số thứ tự nguồn [1], [2],...\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời:"
        )

        return self.llm_fn(prompt)
