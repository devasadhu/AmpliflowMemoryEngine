import os
import json
import logging
import tiktoken
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class AmpliflowMemoryEngine:
    """
    Sticky Memory Model for Ampliflow.ai
    Loads, chunks, embeds and retrieves user context
    from a structured Markdown file directory.
    """

    def __init__(self, user_id: str, base_path: str = "user_memory"):
        self.user_id = user_id
        self.user_path = os.path.join(base_path, user_id)
        self.profiles_path = os.path.join(self.user_path, "profiles")
        self.compliance_path = os.path.join(self.user_path, "compliance")
        self.telemetry_path = os.path.join(self.user_path, "telemetry")
        self.vectorstore = None
        self.embeddings = None
        self.session_state = {}
        self.TOKEN_LIMIT = 4000

        logger.info(f"Initialising AmpliflowMemoryEngine for user: {user_id}")
        self._validate_directory()

    def _validate_directory(self) -> None:
        """Check all required directories exist for this user."""
        required = [
            self.profiles_path,
            self.compliance_path,
            self.telemetry_path
        ]
        for path in required:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required directory missing: {path}"
                )
        logger.info("Directory structure validated successfully")

    def load_session_state(self) -> dict:
        """Load the user's last session state from JSON."""
        state_path = os.path.join(
            self.telemetry_path, "session_state.json"
        )
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                self.session_state = json.load(f)
            logger.info(
                f"Session state loaded. Last active: "
                f"{self.session_state.get('last_active')}"
            )
            return self.session_state
        except FileNotFoundError:
            logger.warning("No session state found. Starting fresh.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Session state file is corrupted: {e}")
            return {}

    def _load_markdown_file(self, filepath: str) -> str:
        """Read a single markdown file and return its content."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Loaded file: {filepath}")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return ""

    def _chunk_markdown(
        self, content: str, source: str
    ) -> list[Document]:
        """
        Split markdown content by headers into semantic chunks.
        Each chunk becomes one searchable unit in the vector store.
        Preserves the relationship between headers and their content.
        """
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        chunks = splitter.split_text(content)

        # Attach source metadata to every chunk
        for chunk in chunks:
            chunk.metadata["source"] = source
            chunk.metadata["user_id"] = self.user_id

        logger.info(
            f"Chunked {source} into {len(chunks)} semantic sections"
        )
        return chunks

    def build_vectorstore(self) -> None:
        """
        Load all ICP profile markdown files, chunk them,
        embed them and store in ChromaDB.
        This runs once on first login or when profiles are updated.
        """
        logger.info("Building vector store from ICP profiles...")

        # Load the free local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        all_chunks = []

        # Load and chunk every profile file
        for filename in os.listdir(self.profiles_path):
            if filename.endswith(".md"):
                filepath = os.path.join(self.profiles_path, filename)
                content = self._load_markdown_file(filepath)
                if content:
                    chunks = self._chunk_markdown(content, filename)
                    all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No profile chunks found to embed.")

        # Store embeddings in ChromaDB, persisted to disk
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(
                self.user_path, "chroma_db"
            ),
            collection_name=f"profiles_{self.user_id}"
        )

        logger.info(
            f"Vector store built with {len(all_chunks)} chunks "
            f"across {len(os.listdir(self.profiles_path))} profiles"
        )

    def retrieve_icp(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not built. Call build_vectorstore() first."
        )
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            logger.warning("No ICP match found for query.")
            return ""

        combined = "\n\n".join([r.page_content for r in results])
        source = results[0].metadata.get("source", "unknown")
        logger.info(f"ICP retrieved from: {source} ({len(results)} chunks)")
        return combined
    

    def load_brand_voice(self) -> str:
        """Always load brand voice in full — it is small and critical."""
        filepath = os.path.join(
            self.compliance_path, "brand_voice.md"
        )
        return self._load_markdown_file(filepath)

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Rough fallback: 1 token per 4 characters
            return len(text) // 4

    def check_token_budget(
        self,
        system_prompt: str,
        icp_context: str,
        brand_voice: str,
        user_query: str
    ) -> dict:
        """
        Check if assembled context fits within token limit.
        Returns a dict with token counts and whether we are safe.
        """
        counts = {
            "system_prompt": self.count_tokens(system_prompt),
            "icp_context": self.count_tokens(icp_context),
            "brand_voice": self.count_tokens(brand_voice),
            "user_query": self.count_tokens(user_query),
        }
        counts["total"] = sum(counts.values())
        counts["limit"] = self.TOKEN_LIMIT
        counts["safe"] = counts["total"] <= self.TOKEN_LIMIT
        counts["remaining"] = self.TOKEN_LIMIT - counts["total"]

        logger.info(
            f"Token budget: {counts['total']}/{self.TOKEN_LIMIT} "
            f"({'OK' if counts['safe'] else 'EXCEEDED'})"
        )
        return counts

    def assemble_context(self, user_query: str) -> dict:
        """
        Master method — called when user submits a generation request.
        Retrieves the right ICP, loads brand voice,
        checks token budget and returns the assembled prompt payload.
        """
        logger.info(f"Assembling context for query: {user_query}")

        system_prompt = (
            "You are an expert content writer for Ampliflow.ai. "
            "Use the provided ICP profile and brand voice guide "
            "to generate content that is precisely targeted, "
            "on brand and platform appropriate."
        )

        # Retrieve most relevant ICP for this query
        icp_context = self.retrieve_icp(user_query, k=3)

        # Always load brand voice
        brand_voice = self.load_brand_voice()

        # Check token budget
        budget = self.check_token_budget(
            system_prompt, icp_context, brand_voice, user_query
        )

        # If over budget, trim brand voice to essentials only
        if not budget["safe"]:
            logger.warning(
                "Token limit exceeded. Trimming brand voice..."
            )
            # Keep only the first 500 characters of brand voice
            brand_voice = brand_voice[:500] + "\n[truncated]"
            budget = self.check_token_budget(
                system_prompt, icp_context, brand_voice, user_query
            )

        return {
            "system_prompt": system_prompt,
            "icp_context": icp_context,
            "brand_voice": brand_voice,
            "user_query": user_query,
            "token_budget": budget
        }


# ── Entry point for testing ────────────────────────────────────
if __name__ == "__main__":
    engine = AmpliflowMemoryEngine(user_id="user_102")
    engine.load_session_state()
    engine.build_vectorstore()

    test_query = (
        "Write a LinkedIn post announcing our new AI assistant feature"
    )
    result = engine.assemble_context(test_query)

    print("\n" + "="*60)
    print("ASSEMBLED CONTEXT")
    print("="*60)
    print(f"\nQUERY: {result['user_query']}")
    print(f"\nTOKEN BUDGET: {result['token_budget']}")
    print(f"\nICP RETRIEVED:\n{result['icp_context']}")
    print(f"\nBRAND VOICE (first 300 chars):\n"
          f"{result['brand_voice'][:300]}...")