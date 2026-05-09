from __future__ import annotations

from fastapi import APIRouter

from credit_report.api import auth, conflicts, facts, reports, audit

router = APIRouter(prefix="/api/credit-report")

router.include_router(auth.router)
router.include_router(reports.router)
router.include_router(facts.router)
router.include_router(conflicts.router)
router.include_router(audit.router)
