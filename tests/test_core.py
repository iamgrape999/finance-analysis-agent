import asyncio
import tempfile
import unittest
from unittest.mock import patch

import app
import Finance_Analysis_Agent_V581 as agent
from fastapi.testclient import TestClient


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

    def test_global_facts_are_truncated_to_safe_length(self):
        long_value = "A" * (agent.GLOBAL_FACT_VALUE_MAX_CHARS + 50)
        agent.upsert_global_fact("notes", long_value, user_id="hanli")
        facts = agent.load_global_facts(user_id="hanli")
        self.assertEqual(len(facts["notes"]["value"]), agent.GLOBAL_FACT_VALUE_MAX_CHARS)

    def test_thread_summaries_use_lightweight_paths(self):
        adapter = agent.MemoryAdapter(memory=None, THREAD_ID="THREAD_LIGHT", USER_ID="hanli")
        adapter.add_turn("user", "preview-text")
        adapter.add_turn("assistant", "reply-text")
        summaries = agent.list_thread_summaries(user_id="hanli")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["message_count"], 2)
        self.assertIn("preview-text", summaries[0]["preview"])


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
            kwargs["model_overrides"],
            {"mistral": "mistral-small-latest", "openrouter": "meta-llama/test"},
        )
        self.assertEqual(kwargs["user_id"], "hanli")


class BasicAuthApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.app)
        self.old_app_password = app.APP_PASSWORD
        self.old_app_users_raw = app.APP_USERS_RAW
        app.APP_PASSWORD = "shared-secret"
        app.APP_USERS_RAW = ""

    def tearDown(self):
        app.APP_PASSWORD = self.old_app_password
        app.APP_USERS_RAW = self.old_app_users_raw

    def test_health_accepts_shared_password(self):
        response = self.client.get(
            "/api/health",
            headers={
                "X-User-Id": "hanli",
                "X-App-Password": "shared-secret",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["user_id"], "hanli")
        self.assertTrue(data["password_required"])

    def test_health_rejects_wrong_shared_password(self):
        health = self.client.get(
            "/api/health",
            headers={
                "X-User-Id": "hanli",
                "X-App-Password": "wrong-secret",
            },
        )
        self.assertEqual(health.status_code, 401)
        self.assertIn("invalid app password", health.json()["detail"].lower())

    def test_health_rejects_unknown_user_in_per_user_mode(self):
        app.APP_PASSWORD = ""
        app.APP_USERS_RAW = "hanli=aaa,cathy=bbb"
        health = self.client.get(
            "/api/health",
            headers={
                "X-User-Id": "iris",
                "X-App-Password": "aaa",
            },
        )
        self.assertEqual(health.status_code, 401)
        self.assertIn("invalid user id or password", health.json()["detail"].lower())


class StartupWarningTests(unittest.TestCase):
    def test_startup_warns_when_auth_is_open(self):
        old_app_password = app.APP_PASSWORD
        old_app_users_raw = app.APP_USERS_RAW
        try:
            app.APP_PASSWORD = ""
            app.APP_USERS_RAW = ""
            with patch("warnings.warn") as mock_warn:
                asyncio.run(app._startup_warning())
        finally:
            app.APP_PASSWORD = old_app_password
            app.APP_USERS_RAW = old_app_users_raw
        mock_warn.assert_called()
        self.assertIn("accessible without authentication", mock_warn.call_args[0][0])


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


class FireworksModelStatusTests(unittest.TestCase):
    def test_fireworks_model_status_marks_catalog_and_serverless(self):
        fake_models = [
            {
                "name": "accounts/fireworks/models/minimax-m2p7",
                "display_name": "MiniMax M2.7",
                "online_in_catalog": True,
                "serverless_supported": True,
            },
            {
                "name": "accounts/fireworks/models/legacy-model",
                "display_name": "Legacy Model",
                "online_in_catalog": True,
                "serverless_supported": False,
            },
        ]
        with patch.object(agent, "list_fireworks_catalog_models", return_value=fake_models):
            ok_status = agent.fireworks_model_status("accounts/fireworks/models/minimax-m2p7")
            legacy_status = agent.fireworks_model_status("accounts/fireworks/models/legacy-model")
            missing_status = agent.fireworks_model_status("accounts/fireworks/models/missing")

        self.assertTrue(ok_status["online_in_catalog"])
        self.assertTrue(ok_status["serverless_supported"])
        self.assertEqual(ok_status["status_label"], "ok")
        self.assertTrue(legacy_status["online_in_catalog"])
        self.assertFalse(legacy_status["serverless_supported"])
        self.assertEqual(legacy_status["status_label"], "catalog_online_not_serverless")
        self.assertFalse(missing_status["online_in_catalog"])
        self.assertFalse(missing_status["serverless_supported"])
        self.assertEqual(missing_status["status_label"], "not_in_account_list")


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
