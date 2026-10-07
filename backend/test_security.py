"""
upay ActivateAI — Security Test Suite.

Verifies RBAC enforcement (API key gating), CORS restriction, SQLite
persistence for campaign approvals, and graceful rejection of
unauthorized requests with HTTP 401.
"""

import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.main import app, ACTIVATE_AI_API_KEY

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VALID_HEADERS = {"X-API-Key": ACTIVATE_AI_API_KEY}
INVALID_HEADERS = {"X-API-Key": "wrong-key-12345"}


# =========================================================================
# RBAC — unauthenticated requests must be rejected with 401
# =========================================================================

class TestRBACUnauthorized:
    """Protected endpoints MUST reject requests without a valid API key."""

    def test_simulate_no_key_returns_401(self):
        """POST /api/simulate-mau-growth without API key → 401."""
        r = client.post(
            "/api/simulate-mau-growth",
            json={"budget_bdt": 10000},
        )
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"

    def test_simulate_wrong_key_returns_401(self):
        """POST /api/simulate-mau-growth with wrong API key → 401."""
        r = client.post(
            "/api/simulate-mau-growth",
            json={"budget_bdt": 10000},
            headers=INVALID_HEADERS,
        )
        assert r.status_code == 401

    def test_customer_no_key_returns_401(self):
        """GET /api/customer/{id} without API key → 401."""
        r = client.get("/api/customer/UPAY_100000")
        assert r.status_code == 401

    def test_customer_wrong_key_returns_401(self):
        """GET /api/customer/{id} with wrong API key → 401."""
        r = client.get("/api/customer/UPAY_100000", headers=INVALID_HEADERS)
        assert r.status_code == 401

    def test_approve_campaign_no_key_returns_401(self):
        """POST /api/approve-campaign without API key → 401."""
        r = client.post(
            "/api/approve-campaign",
            json={
                "admin_user": "Attacker",
                "allocated_budget": 99999,
            },
        )
        assert r.status_code == 401

    def test_approve_campaign_wrong_key_returns_401(self):
        """POST /api/approve-campaign with wrong API key → 401."""
        r = client.post(
            "/api/approve-campaign",
            json={
                "admin_user": "Attacker",
                "allocated_budget": 99999,
            },
            headers=INVALID_HEADERS,
        )
        assert r.status_code == 401

    def test_approvals_list_no_key_returns_401(self):
        """GET /api/approvals without API key → 401."""
        r = client.get("/api/approvals")
        assert r.status_code == 401

    def test_approvals_list_wrong_key_returns_401(self):
        """GET /api/approvals with wrong API key → 401."""
        r = client.get("/api/approvals", headers=INVALID_HEADERS)
        assert r.status_code == 401


# =========================================================================
# RBAC — authenticated requests must succeed
# =========================================================================

class TestRBACAuthorized:
    """Protected endpoints MUST accept requests with a valid API key."""

    def test_simulate_valid_key_returns_200(self):
        """POST /api/simulate-mau-growth with valid key → 200."""
        r = client.post(
            "/api/simulate-mau-growth",
            json={"budget_bdt": 10000},
            headers=VALID_HEADERS,
        )
        assert r.status_code == 200
        data = r.json()
        assert "activate_ai" in data
        assert "mass_blast_baseline" in data

    def test_customer_valid_key_returns_200_or_404(self):
        """GET /api/customer/{id} with valid key → 200 or 404 (not 401)."""
        r = client.get("/api/customer/UPAY_100000", headers=VALID_HEADERS)
        # Might be 404 if this customer isn't in the test set, but never 401
        assert r.status_code in (200, 404)

    def test_approvals_valid_key_returns_200(self):
        """GET /api/approvals with valid key → 200."""
        r = client.get("/api/approvals", headers=VALID_HEADERS)
        assert r.status_code == 200
        assert "approvals" in r.json()


# =========================================================================
# Public endpoints must remain open (no regression)
# =========================================================================

class TestPublicEndpoints:
    """Health and overview must NOT require authentication."""

    def test_health_no_key(self):
        """GET /api/health is public."""
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_overview_no_key(self):
        """GET /api/overview is public."""
        r = client.get("/api/overview")
        assert r.status_code == 200
        assert "funnel" in r.json()

    def test_customers_list_no_key(self):
        """GET /api/customers is public (read-only listing)."""
        r = client.get("/api/customers?page=1&limit=5")
        assert r.status_code == 200
        assert "customers" in r.json()


# =========================================================================
# CORS restriction — no wildcard
# =========================================================================

class TestCORSRestriction:
    """CORS must NOT include '*' when credentials are enabled."""

    def test_allowed_origin_gets_cors_header(self):
        """Request from allowed origin gets Access-Control-Allow-Origin."""
        r = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_disallowed_origin_gets_no_cors_header(self):
        """Request from disallowed origin must NOT get a permissive CORS header."""
        r = client.get(
            "/api/health",
            headers={"Origin": "http://evil.example.com"},
        )
        acao = r.headers.get("access-control-allow-origin", "")
        assert acao != "*", "Wildcard CORS is not allowed with credentials"
        assert "evil.example.com" not in acao


# =========================================================================
# SQLite persistence — campaign approvals survive round-trips
# =========================================================================

class TestSQLitePersistence:
    """Campaign approvals must be stored in SQLite, not in-memory."""

    def test_approve_and_retrieve(self):
        """Approve a campaign and verify it appears in the ledger."""
        # Create approval
        r = client.post(
            "/api/approve-campaign",
            json={
                "admin_user": "Security Test Reviewer",
                "allocated_budget": 7500,
                "notes": "pytest security test",
            },
            headers=VALID_HEADERS,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["approved"] is True
        assert data["admin_user"] == "Security Test Reviewer"
        assert data["allocated_budget"] == 7500
        assert "campaign_id" in data
        assert "timestamp" in data

        campaign_id = data["campaign_id"]

        # Verify it's in the ledger
        r2 = client.get("/api/approvals", headers=VALID_HEADERS)
        assert r2.status_code == 200
        approvals = r2.json()["approvals"]
        ids = [a["campaign_id"] for a in approvals]
        assert campaign_id in ids

    def test_approval_requires_admin_user(self):
        """Empty admin_user must be rejected by Pydantic validation."""
        r = client.post(
            "/api/approve-campaign",
            json={
                "admin_user": "",
                "allocated_budget": 10000,
            },
            headers=VALID_HEADERS,
        )
        assert r.status_code == 422  # Pydantic validation error

    def test_approval_requires_positive_budget(self):
        """Zero or negative budget must be rejected."""
        r = client.post(
            "/api/approve-campaign",
            json={
                "admin_user": "Tester",
                "allocated_budget": 0,
            },
            headers=VALID_HEADERS,
        )
        assert r.status_code == 422


# =========================================================================
# Database module unit tests
# =========================================================================

class TestDatabaseModule:
    """Direct tests for backend/database.py functions."""

    def test_init_and_insert(self, tmp_path):
        """insert_approval + list_approvals round-trip with temp DB."""
        from backend.database import override_db_path, init_db, insert_approval, list_approvals

        db_file = tmp_path / "test_approvals.db"
        override_db_path(db_file)
        init_db()

        record = insert_approval(
            campaign_id="TEST_001",
            admin_user="Unit Tester",
            allocated_budget=5000.0,
            notes="unit test",
        )
        assert record["approved"] is True
        assert record["campaign_id"] == "TEST_001"

        rows = list_approvals()
        assert len(rows) >= 1
        assert any(r["campaign_id"] == "TEST_001" for r in rows)

    def test_duplicate_campaign_id_rejected(self, tmp_path):
        """Inserting the same campaign_id twice must raise an error."""
        import sqlite3
        from backend.database import override_db_path, init_db, insert_approval

        db_file = tmp_path / "test_dup.db"
        override_db_path(db_file)
        init_db()

        insert_approval("DUP_001", "Tester", 1000.0)
        with pytest.raises(sqlite3.IntegrityError):
            insert_approval("DUP_001", "Tester", 2000.0)
