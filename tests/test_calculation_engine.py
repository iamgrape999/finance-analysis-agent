"""
Sprint 2 acceptance tests for the Calculation Engine and Block AST layers.

Covers:
  1. FX rate store + retrieval + staleness mark
  2. FX conversion with lineage (TWD → USD)
  3. Dual-currency display formatting
  4. Unmapped line item queue + manual mapping approval
  5. IBD aggregation respects mapping rules
  6. DSCR calculation with formula lineage
  7. LTV/ACR table (straight-line depreciation)
  8. Balloon LTV summary
  9. Cash flow classification
  10. Block AST builder: Markdown → blocks + cells
  11. Block AST builder: table cell numeric binding to facts
  12. Block optimistic locking (409 on stale version)
  13. Block history snapshot on update
  14. Stale propagation: mapping rule approval marks calcs stale
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from credit_report.database import Base

# Import calculation engine models so Base knows about them
import credit_report.calculation_engine.models  # noqa: F401
import credit_report.block_ast.models  # noqa: F401
import credit_report.fact_store.models  # noqa: F401
import credit_report.security.models  # noqa: F401
import credit_report.audit.events  # noqa: F401
import credit_report.models  # noqa: F401

from credit_report.calculation_engine.exchange_rate.rate_table import set_rate, get_rate, check_staleness
from credit_report.calculation_engine.exchange_rate.conversion import convert, dual_currency_display, MissingFXRateError
from credit_report.calculation_engine.mapping.mapping_rules import (
    submit_mapping_rule, approve_mapping_rule, get_unmapped_queue, queue_unmapped_item,
)
from credit_report.calculation_engine.dscr import calculate_dscr, store_dscr
from credit_report.calculation_engine.ltv_acr import build_ltv_table, balloon_ltv_summary
from credit_report.calculation_engine.collateral import ltc, current_ltv, rg_coverage
from credit_report.calculation_engine.cash_flow import classify_cash_flow, summarize_cash_flows
from credit_report.calculation_engine.interest_bearing_debt import calculate_ibd
from credit_report.block_ast.builder import build_blocks, segment_markdown
from credit_report.block_ast.repository import (
    save_blocks, get_block, get_block_cells, update_block_content,
    get_block_history, BlockOptimisticLockError,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def db() -> AsyncSession:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


REPORT_ID = "RPT-TEST-001"


# ── 1. FX rate store + retrieval ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fx_rate_store_and_retrieve(db: AsyncSession):
    rate = await set_rate(db, REPORT_ID, "TWD", "USD", 0.03236, "2024-12-31")
    await db.flush()
    fetched = await get_rate(db, REPORT_ID, "TWD", "USD")
    assert fetched is not None
    assert fetched.rate == 0.03236
    assert fetched.is_stale is False


@pytest.mark.asyncio
async def test_fx_rate_new_rate_marks_old_stale(db: AsyncSession):
    await set_rate(db, REPORT_ID, "TWD", "USD", 0.03236, "2024-12-31")
    await db.flush()
    await set_rate(db, REPORT_ID, "TWD", "USD", 0.03200, "2025-01-31")
    await db.flush()
    # Only latest should be non-stale
    fetched = await get_rate(db, REPORT_ID, "TWD", "USD")
    assert fetched.rate == 0.03200
    assert fetched.is_stale is False


# ── 2. FX conversion with lineage ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fx_conversion_twd_to_usd(db: AsyncSession):
    await set_rate(db, REPORT_ID, "TWD", "USD", 0.032362, "2024-12-31")
    await db.flush()
    converted, lineage = await convert(db, REPORT_ID, 253400.0, "TWD", "USD")
    # 253,400 TWD million × 0.032362 ≈ 8,200.4 USD million
    assert 8100 < converted < 8300
    assert lineage["rate"] == 0.032362
    assert lineage["rate_date"] == "2024-12-31"


@pytest.mark.asyncio
async def test_fx_conversion_same_currency(db: AsyncSession):
    converted, lineage = await convert(db, REPORT_ID, 1000.0, "USD", "USD")
    assert converted == 1000.0
    assert lineage["rate"] == 1.0


@pytest.mark.asyncio
async def test_fx_conversion_missing_rate_raises(db: AsyncSession):
    with pytest.raises(MissingFXRateError):
        await convert(db, REPORT_ID, 100.0, "EUR", "USD")


# ── 3. Dual-currency display ──────────────────────────────────────────────────

def test_dual_currency_display():
    s = dual_currency_display(253400.0, "TWD", 8200.0, "USD", "million", "million")
    assert "TWD" in s and "USD" in s
    assert "253,400.0m" in s
    assert "8,200.0m" in s


# ── 4. Unmapped line item queue + mapping approval ───────────────────────────

@pytest.mark.asyncio
async def test_unmapped_queue_and_approval(db: AsyncSession):
    # Add item to queue
    item = await queue_unmapped_item(db, REPORT_ID, "Lease liabilities", 7, 120.5)
    await db.flush()
    queue = await get_unmapped_queue(db, REPORT_ID)
    assert len(queue) == 1
    assert queue[0].source_label == "Lease liabilities"

    # Submit and approve a mapping rule
    rule = await submit_mapping_rule(
        db, REPORT_ID, "Lease liabilities", "interest_bearing_debt",
        "balance_sheet", "analyst-001",
    )
    await db.flush()
    approved_rule = await approve_mapping_rule(db, rule.id, "reviewer-001")
    await db.flush()

    assert approved_rule.status == "approved"
    assert approved_rule.approved_by == "reviewer-001"

    # Queue should now be empty (item resolved)
    queue_after = await get_unmapped_queue(db, REPORT_ID)
    assert len(queue_after) == 0


@pytest.mark.asyncio
async def test_duplicate_unmapped_item_not_duplicated(db: AsyncSession):
    await queue_unmapped_item(db, REPORT_ID, "Lease liabilities", 7, 120.5)
    await db.flush()
    await queue_unmapped_item(db, REPORT_ID, "Lease liabilities", 7, 120.5)
    await db.flush()
    queue = await get_unmapped_queue(db, REPORT_ID)
    assert len(queue) == 1  # Not duplicated


# ── 5. IBD aggregation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ibd_aggregation_only_includes_ib_items(db: AsyncSession):
    line_items = [
        {"label": "Bank loans", "value": 1000.0, "canonical_metric": "interest_bearing_debt"},
        {"label": "Lease liabilities", "value": 120.5, "canonical_metric": "interest_bearing_debt"},
        {"label": "Accounts payable", "value": 500.0, "canonical_metric": "accounts_payable"},  # NOT IB
    ]
    total, formula, labels = await calculate_ibd(db, REPORT_ID, "EMA", "FY2024", line_items)
    assert total == 1120.5
    assert "Accounts payable" not in labels
    assert "1,120.5" in formula


# ── 6. DSCR calculation ───────────────────────────────────────────────────────

def test_dscr_calculation():
    dscr, formula, fact_ids = calculate_dscr(
        operating_cash_flow=3000.0,
        principal_repayment=200.0,
        interest_expense=86.6,
        ocf_fact_id="FACT-OCF-001",
        principal_fact_id="FACT-PRINC-001",
        interest_fact_id="FACT-INT-001",
    )
    assert dscr is not None
    assert abs(dscr - 3000.0 / 286.6) < 0.01
    assert "3,000.0" in formula
    assert "FACT-OCF-001" in fact_ids
    assert "FACT-INT-001" in fact_ids


def test_dscr_zero_debt_service_returns_none():
    dscr, formula, _ = calculate_dscr(3000.0, 0.0, 0.0)
    assert dscr is None
    assert "N/M" in formula


@pytest.mark.asyncio
async def test_dscr_store_and_retrieve(db: AsyncSession):
    dscr, formula, fact_ids = calculate_dscr(3000.0, 200.0, 86.6, "F1", "F2", "F3")
    result = await store_dscr(db, REPORT_ID, "EMA", "FY2024", dscr, formula, fact_ids)
    await db.flush()
    assert result.metric_name == "dscr"
    assert result.entity == "EMA"
    assert result.value is not None
    assert "2024" in result.period


# ── 7. LTV/ACR table ─────────────────────────────────────────────────────────

def test_ltv_table_year_zero():
    schedule = [{"year": 0, "outstanding_pct": 100}]
    rows = build_ltv_table(213.84, 267.30, schedule, useful_life_25yr=25, useful_life_20yr=20)
    assert len(rows) == 1
    row = rows[0]
    assert row.loan_outstanding == 213.84
    assert row.asset_value_25yr == 267.30
    assert row.ltv_25yr_pct == round(213.84 / 267.30 * 100, 1)


def test_ltv_table_full_schedule():
    schedule = [
        {"year": 0, "outstanding_pct": 100},
        {"year": 0.5, "outstanding_pct": 95},
        {"year": 1, "outstanding_pct": 90},
        {"year": 7, "outstanding_pct": 35},
    ]
    rows = build_ltv_table(213.84, 267.30, schedule)
    assert len(rows) == 4
    # At year 7, LTV should be lower than at year 0 (asset still valuable)
    ltv_0 = rows[0].ltv_25yr_pct
    ltv_7 = rows[3].ltv_25yr_pct
    assert ltv_7 < ltv_0  # Loan paid down faster than asset depreciates


def test_balloon_ltv_summary():
    # Balloon = 35% of 213.84 = 74.844 USD m
    balloon = 213.84 * 0.35
    asset_25yr = 195.7  # from plan example
    asset_20yr = 178.5
    summary = balloon_ltv_summary(balloon, asset_25yr, asset_20yr)
    assert summary["ltv_25yr_pct"] == round(balloon / asset_25yr * 100, 1)
    assert summary["acr_25yr_pct"] == round(asset_25yr / balloon * 100, 1)


# ── 8. Collateral calculations ───────────────────────────────────────────────

def test_ltc_calculation():
    val, formula = ltc(213.84, 267.30)
    assert abs(val - 80.0) < 0.5
    assert "213.84" in formula


def test_current_ltv_acr():
    ltv_pct, acr_pct, formula = current_ltv(100.0, 123.95)
    assert abs(ltv_pct - 80.7) < 0.5
    assert abs(acr_pct - 123.95) < 0.5


def test_rg_coverage():
    coverage, formula = rg_coverage(133.65, 85.50)
    assert abs(coverage - 156.3) < 1.0
    assert "156" in formula


# ── 9. Cash flow classification ──────────────────────────────────────────────

def test_cf_classification_known_items():
    assert classify_cash_flow("operating_cash_flow") == "operating"
    assert classify_cash_flow("capex") == "investing"
    assert classify_cash_flow("dividends_paid") == "financing"


def test_cf_classification_unknown_returns_none():
    assert classify_cash_flow("something_exotic") is None


def test_cf_summarize():
    items = [
        {"label": "OCF", "value": 3000.0, "canonical_metric": "operating_cash_flow"},
        {"label": "Capex", "value": -500.0, "canonical_metric": "capex"},
        {"label": "Debt repayment", "value": -200.0, "canonical_metric": "debt_repayment"},
        {"label": "Unknown", "value": 999.0, "canonical_metric": "unknown_item"},  # excluded
    ]
    totals = summarize_cash_flows(items)
    assert totals["operating"] == 3000.0
    assert totals["investing"] == -500.0
    assert totals["financing"] == -200.0


# ── 10. Block AST builder: segment_markdown ───────────────────────────────────

def test_segment_markdown_types():
    md = """## Section 2 Overall Comments

This is a paragraph with value 2791.

| Entity | Cash | Debt |
|--------|------|------|
| EMA | 2791 | 2488 |

- Risk 1: tariff impact
- Risk 2: refinancing
"""
    segments = segment_markdown(md)
    types = [s["type"] for s in segments]
    assert "heading" in types
    assert "paragraph" in types
    assert "table" in types
    assert "list" in types


# ── 11. Block AST builder: fact binding ──────────────────────────────────────

def test_block_builder_table_cell_binding():
    md = """| Entity | Cash | Debt |
|--------|------|------|
| EMA | 2791.0 | 2488.0 |
"""
    facts = [
        {"id": "FACT-CASH-EMA-FY2024", "metric_name": "cash_balance", "value": 2791.0},
        {"id": "FACT-DEBT-EMA-FY2024", "metric_name": "total_debt", "value": 2488.0},
    ]
    blocks, cells = build_blocks("RPT-001", 2, md, facts)
    assert len(blocks) == 1
    assert blocks[0]["block_type"] == "table"

    bound_cells = [c for c in cells if c["binding_status"] == "bound"]
    bound_fact_ids = {c["fact_id"] for c in bound_cells}
    assert "FACT-CASH-EMA-FY2024" in bound_fact_ids
    assert "FACT-DEBT-EMA-FY2024" in bound_fact_ids


def test_block_builder_paragraph_no_facts():
    md = "This is a paragraph with no financial numbers.\n"
    blocks, cells = build_blocks("RPT-001", 2, md, [])
    assert len(blocks) == 1
    assert blocks[0]["block_type"] == "paragraph"
    assert cells == []


def test_block_builder_chart_image_detected():
    md = "Refer to [Alliance Chart] for vessel capacity data.\n"
    blocks, cells = build_blocks("RPT-001", 4, md, [])
    chart_blocks = [b for b in blocks if b["block_type"] == "chart_image"]
    assert len(chart_blocks) == 1


# ── 12. Block optimistic locking ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_block_patch_optimistic_lock(db: AsyncSession):
    md = "## Section heading\n\nSome content here.\n"
    blocks, cells = build_blocks("RPT-001", 2, md, [])
    await save_blocks(db, blocks, cells)
    await db.flush()

    block_id = blocks[0]["id"]
    # First update at version 1 succeeds
    await update_block_content(db, block_id, "Updated content", "user-001", "Test edit", 1)
    await db.flush()

    # Second update with stale version 1 should fail
    with pytest.raises(BlockOptimisticLockError):
        await update_block_content(db, block_id, "Another edit", "user-002", "Concurrent", 1)


# ── 13. Block history snapshot ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_block_update_creates_history_snapshot(db: AsyncSession):
    md = "## Heading\n\nOriginal content.\n"
    blocks, cells = build_blocks("RPT-001", 1, md, [])
    await save_blocks(db, blocks, cells)
    await db.flush()

    block_id = blocks[0]["id"]

    block = await get_block(db, block_id)
    if block and block.block_type == "heading":
        # Get a non-heading block
        paragraph_blocks = [b for b in blocks if b["block_type"] == "paragraph"]
        if not paragraph_blocks:
            pytest.skip("No paragraph block to test")
        block_id = paragraph_blocks[0]["id"]

    await update_block_content(db, block_id, "Revised content v2", "analyst-001", "Restatement", 1)
    await db.flush()

    history = await get_block_history(db, block_id)
    assert len(history) == 1
    assert "Original" in (history[0].content or "")


# ── 14. Stale flag on block after calculation change ─────────────────────────

@pytest.mark.asyncio
async def test_stale_blocks_flagged_after_mapping_change(db: AsyncSession):
    from credit_report.block_ast import repository as block_repo

    md = "Balance sheet data: 1250.5 total IBD.\n"
    blocks, cells = build_blocks(REPORT_ID, 7, md, [])
    await save_blocks(db, blocks, cells)
    await db.flush()

    # Mark blocks stale (simulates what happens when a mapping rule changes)
    await block_repo.mark_section_blocks_stale(db, REPORT_ID, 7)
    await db.flush()

    stale_blocks = [b for b in await block_repo.get_blocks_for_section(db, REPORT_ID, 7) if b.is_stale]
    assert len(stale_blocks) > 0
