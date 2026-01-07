import types
import pytest
from embeddings.ollama_embed import OllamaEmbeddings

@pytest.mark.asyncio
async def test_cosine_and_mock_embed(monkeypatch):
    emb = OllamaEmbeddings("http://localhost:11434")

    async def fake_post(self, texts):
        # Return two simple vectors for two texts
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(OllamaEmbeddings, "embed", fake_post)

    vecs = await emb.embed(["a", "b"])  # type: ignore[arg-type]
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]

    sim = OllamaEmbeddings.cosine(vecs[0], vecs[0])
    assert sim == pytest.approx(1.0)
    sim2 = OllamaEmbeddings.cosine(vecs[0], vecs[1])
    assert sim2 == pytest.approx(0.0, abs=1e-9)
