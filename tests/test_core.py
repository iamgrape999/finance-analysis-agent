import tempfile
import unittest
from unittest.mock import patch

import app
import Finance_Analysis_Agent_V581 as agent


class MemoryIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_memory_root = agent.MEMORY_ROOT
        agent.MEMORY_ROOT = self.tempdir.name

    def tearDown(self):
        agent.MEMORY_ROOT = self.old_memory_root
        self.tempdir.cleanup()

    def test_thread_summaries_are_user_scoped(self):
        hanli_adapter = agent.MemoryAdapter(memory=None, THREAD_ID="THREAD_SHARED", USER_ID="hanli")
        hanli_adapter.add_turn("user", "hanli-thread-message")
        hanli_adapter.add_turn("assistant", "reply")

        cathy_adapter = agent.MemoryAdapter(memory=None, THREAD_ID="THREAD_SHARED", USER_ID="cathy")
        cathy_adapter.add_turn("user", "cathy-thread-message")
        cathy_adapter.add_turn("assistant", "reply")

        hanli_summaries = agent.list_thread_summaries(user_id="hanli")
        cathy_summaries = agent.list_thread_summaries(user_id="cathy")

        self.assertEqual(len(hanli_summaries), 1)
        self.assertEqual(len(cathy_summaries), 1)
        self.assertEqual(hanli_summaries[0]["thread_id"], "THREAD_SHARED")
        self.assertEqual(cathy_summaries[0]["thread_id"], "THREAD_SHARED")
        self.assertIn("hanli-thread-message", hanli_summaries[0]["preview"])
        self.assertIn("cathy-thread-message", cathy_summaries[0]["preview"])
        self.assertNotIn("cathy-thread-message", hanli_summaries[0]["preview"])
        self.assertNotIn("hanli-thread-message", cathy_summaries[0]["preview"])

    def test_global_facts_are_user_scoped(self):
        agent.upsert_global_fact("customer_segment", "SME", user_id="hanli")
        agent.upsert_global_fact("customer_segment", "Enterprise", user_id="cathy")

        hanli_facts = agent.load_global_facts(user_id="hanli")
        cathy_facts = agent.load_global_facts(user_id="cathy")

        self.assertEqual(hanli_facts["customer_segment"]["value"], "SME")
        self.assertEqual(cathy_facts["customer_segment"]["value"], "Enterprise")


class ProviderOverrideTests(unittest.TestCase):
    def test_chat_impl_passes_request_scoped_overrides(self):
        req = app.ChatRequest(
            thread_id="THREAD_A",
            message="hello",
            response_mode="stable",
            provider_order="mistral,openrouter",
            model_overrides={"mistral": "mistral-small-latest", "openrouter": "meta-llama/test"},
        )

        with patch.object(app, "chat_once_detailed") as mock_chat:
            mock_chat.return_value = {
                "reply": "ok",
                "meta": {
                    "provider": "mistral",
                    "model": "mistral-small-latest",
                    "usage": {},
                    "provider_attempts": ["mistral"],
                    "failover_errors": [],
                    "provider_trace": [],
                    "context_sizes": {},
                    "web_search": {},
                    "continue_rounds": 0,
                },
            }

            response = app._chat_impl(req, user_id="hanli", enforce_min_tokens=False)

        self.assertEqual(response.reply, "ok")
        _, kwargs = mock_chat.call_args
        self.assertEqual(kwargs["provider_order_override"], "mistral,openrouter")
        self.assertEqual(
            kwargs["provider_models_override"],
            {"mistral": "mistral-small-latest", "openrouter": "meta-llama/test"},
        )
        self.assertEqual(kwargs["user_id"], "hanli")


class SearchClassifierTests(unittest.TestCase):
    def setUp(self):
        self.old_enabled = agent.WEB_SEARCH_ENABLED
        self.old_tavily = agent.TAVILY_API_KEY
        self.old_serper = agent.SERPER_API_KEY
        agent.WEB_SEARCH_ENABLED = True
        agent.TAVILY_API_KEY = "test-key"
        agent.SERPER_API_KEY = ""

    def tearDown(self):
        agent.WEB_SEARCH_ENABLED = self.old_enabled
        agent.TAVILY_API_KEY = self.old_tavily
        agent.SERPER_API_KEY = self.old_serper

    def test_explicit_search_request_is_forced(self):
        decision = agent.search_intent_classifier("請幫我查一下最新的 NVIDIA NIM 模型")
        self.assertTrue(decision["must_search"])
        self.assertEqual(decision["priority"], "explicit")

    def test_blocked_pattern_beats_latest_keyword(self):
        decision = agent.search_intent_classifier("解釋什麼是最新科技")
        self.assertFalse(decision["must_search"])
        self.assertEqual(decision["priority"], "blocked")

    def test_priority_one_keyword_requires_search(self):
        decision = agent.search_intent_classifier("今天台灣股市收盤多少？")
        self.assertTrue(decision["must_search"])
        self.assertEqual(decision["priority"], "p1")


class MaskingTests(unittest.TestCase):
    def test_masks_common_pii_patterns(self):
        text = (
            "Email: hanli.chang@example.com "
            "phone 0912345678 "
            "id A123456789 "
            "card 4111-1111-1111-1111"
        )
        masked = agent.mask_pii(text)
        self.assertIn("[EMAIL]", masked)
        self.assertIn("[MOBILE_TW]", masked)
        self.assertIn("[TW_ID]", masked)
        self.assertIn("[CARD]", masked)
        self.assertNotIn("example.com", masked)
        self.assertNotIn("0912345678", masked)
        self.assertNotIn("A123456789", masked)
        self.assertNotIn("4111-1111-1111-1111", masked)


if __name__ == "__main__":
    unittest.main()
