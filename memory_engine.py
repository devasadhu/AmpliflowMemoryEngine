import os
import json
import logging
import tiktoken
from datetime import datetime, timezone
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# ── Logging setup ─────────────────────────────────────────────────────────────
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

    # ── Directory validation ───────────────────────────────────────────────────

    def _validate_directory(self) -> None:
        """Check all required directories exist for this user."""
        required = [self.profiles_path, self.compliance_path, self.telemetry_path]
        for path in required:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required directory missing: {path}")
        logger.info("Directory structure validated successfully")

    # ── Session state ──────────────────────────────────────────────────────────

    def load_session_state(self) -> dict:
        """Load the user's last session state from JSON."""
        state_path = os.path.join(self.telemetry_path, "session_state.json")
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

    def save_session_state(
        self, last_icp: str = "", last_query: str = ""
    ) -> None:
        """Write updated session state back to disk after a generation."""
        state_path = os.path.join(self.telemetry_path, "session_state.json")
        self.session_state.update({
            "last_active": datetime.now(timezone.utc).isoformat(),
            "last_icp_loaded": last_icp,
            "last_query": last_query,
        })
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(self.session_state, f, indent=2)
            logger.info(f"Session state saved. Last ICP: {last_icp}")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    # ── File loading & chunking ────────────────────────────────────────────────

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

    def _chunk_markdown(self, content: str, source: str) -> list[Document]:
        """
        Split markdown content by headers into semantic chunks.
        Each chunk becomes one searchable unit in the vector store.
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
        for chunk in chunks:
            chunk.metadata["source"] = source
            chunk.metadata["user_id"] = self.user_id
        logger.info(f"Chunked {source} into {len(chunks)} semantic sections")
        return chunks

    # ── Vector store ───────────────────────────────────────────────────────────

    def build_vectorstore(self) -> None:
        """
        Load all ICP profile markdown files, chunk, embed and store in ChromaDB.
        On repeat runs, loads from disk instead of rebuilding — saves startup time.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        chroma_path = os.path.join(self.user_path, "chroma_db")
        collection_name = f"profiles_{self.user_id}"

        # Load from disk if already built
        if os.path.exists(chroma_path) and os.listdir(chroma_path):
            logger.info("Loading vector store from disk (cached)...")
            self.vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            count = self.vectorstore._collection.count()
            logger.info(f"Vector store loaded: {count} chunks from disk")
            return

        logger.info("Building vector store from ICP profiles...")
        all_chunks = []

        for filename in os.listdir(self.profiles_path):
            if filename.endswith(".md"):
                filepath = os.path.join(self.profiles_path, filename)
                content = self._load_markdown_file(filepath)
                if content:
                    chunks = self._chunk_markdown(content, filename)
                    all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No profile chunks found to embed.")

        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=chroma_path,
            collection_name=collection_name
        )
        logger.info(
            f"Vector store built with {len(all_chunks)} chunks "
            f"across {len(os.listdir(self.profiles_path))} profiles"
        )

    def retrieve_icp(self, query: str, k: int = 3) -> tuple[str, str]:
        """
        Retrieve the k most relevant ICP chunks for a query.
        Returns (combined_text, source_filename).
        """
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not built. Call build_vectorstore() first."
            )
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            logger.warning("No ICP match found for query.")
            return "", ""

        combined = "\n\n".join([r.page_content for r in results])
        source = results[0].metadata.get("source", "unknown")
        logger.info(f"ICP retrieved from: {source} ({len(results)} chunks)")
        return combined, source

    # ── Compliance & brand voice ───────────────────────────────────────────────

    def load_brand_voice(self) -> str:
        """Always load brand voice in full — it is small and critical."""
        filepath = os.path.join(self.compliance_path, "brand_voice.md")
        return self._load_markdown_file(filepath)

    def load_compliance_rules(self, icp_source: str) -> str:
        """
        Load only the compliance section relevant to the retrieved ICP type.
        Maps ICP filename keywords to compliance section headers.
        Falls back to full file if no match found.
        """
        filepath = os.path.join(self.compliance_path, "compliance_rules.md")
        full_content = self._load_markdown_file(filepath)
        if not full_content:
            return ""

        keyword_map = {
            "health":        "health",
            "nonprofit":     "nonprofit",
            "charity":       "nonprofit",
            "ecommerce":     "ecommerce",
            "retail":        "ecommerce",
            "local_service": "local",
            "local":         "local",
            "tech":          "tech",
            "ai":            "tech",
            "education":     "education",
            "training":      "education",
            "industrial":    "industrial",
            "manufacturing": "industrial",
            "community":     "community",
            "culture":       "community",
        }

        icp_lower = icp_source.lower()
        matched_keyword = None
        for fragment, section_key in keyword_map.items():
            if fragment in icp_lower:
                matched_keyword = section_key
                break

        if not matched_keyword:
            logger.info("No specific compliance section matched. Loading full rules.")
            return full_content

        # Extract only the matching ## section
        lines = full_content.splitlines()
        capture = False
        section_lines = []

        for line in lines:
            if line.startswith("##") and matched_keyword in line.lower():
                capture = True
            elif line.startswith("##") and capture:
                break
            if capture:
                section_lines.append(line)

        if section_lines:
            logger.info(
                f"Compliance section loaded for: {matched_keyword} "
                f"({len(section_lines)} lines)"
            )
            return "\n".join(section_lines)

        logger.warning(
            f"Section '{matched_keyword}' not found in compliance_rules.md. "
            "Loading full file."
        )
        return full_content

    # ── Token budget ───────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text) // 4

    def check_token_budget(
        self,
        system_prompt: str,
        icp_context: str,
        brand_voice: str,
        compliance: str,
        user_query: str
    ) -> dict:
        """
        Check if assembled context fits within token limit.
        Returns a dict with per-component token counts and budget status.
        """
        counts = {
            "system_prompt": self.count_tokens(system_prompt),
            "icp_context":   self.count_tokens(icp_context),
            "brand_voice":   self.count_tokens(brand_voice),
            "compliance":    self.count_tokens(compliance),
            "user_query":    self.count_tokens(user_query),
        }
        counts["total"]     = sum(counts.values())
        counts["limit"]     = self.TOKEN_LIMIT
        counts["safe"]      = counts["total"] <= self.TOKEN_LIMIT
        counts["remaining"] = self.TOKEN_LIMIT - counts["total"]

        logger.info(
            f"Token budget: {counts['total']}/{self.TOKEN_LIMIT} "
            f"({'OK' if counts['safe'] else 'EXCEEDED'})"
        )
        return counts

    # ── Master assembly ────────────────────────────────────────────────────────

    def assemble_context(self, user_query: str) -> dict:
        """
        Master method — called when user submits a generation request.
        Retrieves the right ICP, loads brand voice and selective compliance,
        checks token budget and returns the assembled prompt payload.
        """
        logger.info(f"Assembling context for query: {user_query}")

        system_prompt = (
            "You are an expert content writer for Ampliflow.ai. "
            "Use the provided ICP profile, brand voice guide and compliance "
            "rules to generate content that is precisely targeted, "
            "on brand and platform appropriate."
        )

        icp_context, icp_source = self.retrieve_icp(user_query, k=3)
        brand_voice = self.load_brand_voice()
        compliance = self.load_compliance_rules(icp_source)

        budget = self.check_token_budget(
            system_prompt, icp_context, brand_voice, compliance, user_query
        )

        # If over budget, trim compliance first, then brand voice
        if not budget["safe"]:
            logger.warning("Token limit exceeded. Trimming compliance rules...")
            compliance = compliance[:300] + "\n[truncated]"
            budget = self.check_token_budget(
                system_prompt, icp_context, brand_voice, compliance, user_query
            )

        if not budget["safe"]:
            logger.warning("Still over budget. Trimming brand voice...")
            brand_voice = brand_voice[:500] + "\n[truncated]"
            budget = self.check_token_budget(
                system_prompt, icp_context, brand_voice, compliance, user_query
            )

        # Write session state back to disk
        self.save_session_state(last_icp=icp_source, last_query=user_query)

        return {
            "system_prompt": system_prompt,
            "icp_context":   icp_context,
            "icp_source":    icp_source,
            "brand_voice":   brand_voice,
            "compliance":    compliance,
            "user_query":    user_query,
            "token_budget":  budget,
        }


# ── Entry point for testing ────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = AmpliflowMemoryEngine(user_id="user_102")
    engine.load_session_state()
    engine.build_vectorstore()

    test_query = (
        "Write a LinkedIn post announcing our new AI assistant feature"
    )
    result = engine.assemble_context(test_query)

    print("\n" + "=" * 60)
    print("ASSEMBLED CONTEXT")
    print("=" * 60)
    print(f"\nQUERY:      {result['user_query']}")
    print(f"ICP SOURCE: {result['icp_source']}")
    print(f"\nTOKEN BUDGET: {result['token_budget']}")
    print(f"\nICP RETRIEVED (first 400 chars):\n{result['icp_context'][:400]}...")
    print(f"\nCOMPLIANCE LOADED (first 200 chars):\n{result['compliance'][:200]}...")
    print(f"\nBRAND VOICE (first 200 chars):\n{result['brand_voice'][:200]}...")