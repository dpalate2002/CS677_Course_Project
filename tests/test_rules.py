"""Two kinds of tests.

1. Unit tests: each rule fires on a hand-made bad record and not on a good one
   (validates the rules themselves, including the date-granularity trap).
2. Data assertions: each rule run against the real 2025 file. A failing data
   assertion is a *finding* (a documented defect), not a bug in the code.
   Run: pytest tests -v  (or  pytest tests -m "not data" for unit tests only)
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
import pytest
import dq

GOOD = dict(service_request_id=1, service_request_parent_id=None, sap_notification_number=None,
            date_requested="2025-03-01 15:30:00.000", case_age_days=0, case_record_type="TSW",
            service_name="Pothole", service_name_detail=None, date_closed="2025-03-01",
            status="Closed", lat=32.72, lng=-117.16, street_address="1 Main St",
            zipcode=92101.0, council_district=3.0, comm_plan_code=1.0, comm_plan_name="DOWNTOWN",
            park_name=None, case_origin="Mobile", referred=None, iamfloc=None, floc=None,
            public_description="hole")


def frame(**over):
    rows = [GOOD, {**GOOD, "service_request_id": 2, **over}]
    return dq.prepare(pd.DataFrame(rows))


BAD_CASES = {
    "C1_required_nonnull": dict(service_name=None),
    "C2_location_present": dict(lat=None),
    "C3_district_present": dict(council_district=None),
    "C4_zip_present": dict(zipcode=None),
    "U1_id_unique": dict(service_request_id=1),
    "V1_coords_in_city": dict(lat=-117.16, lng=32.72),          # swapped
    "V2_district_domain": dict(council_district=11.0),
    "V3_zip_format": dict(zipcode=94103.0),
    "V4_status_domain": dict(status="Open"),
    "K1_close_after_request": dict(date_closed="2025-02-27", case_age_days=-2),
    "K2_age_matches_dates": dict(case_age_days=40),
    "T1_closed_in_2025": dict(date_closed="2026-01-02", case_age_days=307),
    "T2_requested_after_launch": dict(date_requested="2014-03-01 10:00:00.000",
                                      case_age_days=4018),
    "T3_age_nonnegative": dict(date_closed="2025-02-27", case_age_days=-2),
}


@pytest.mark.parametrize("rule", list(BAD_CASES))
def test_rule_fires_on_bad_record(rule):
    v = dq.RULES[rule][2](frame(**BAD_CASES[rule])).fillna(False)
    assert bool(v.iloc[1]), f"{rule} missed its bad record"


@pytest.mark.parametrize("rule", [r for r in dq.RULES if r != "U1_id_unique"])
def test_rule_passes_good_record(rule):
    v = dq.RULES[rule][2](frame()).fillna(False)
    assert not v.any(), f"{rule} false alarm on a clean record"


def test_same_day_closure_is_not_a_violation():
    """The format trap: closed the same day, later in the day than midnight."""
    f = frame(date_requested="2025-03-01 23:59:00.000", date_closed="2025-03-01")
    assert not dq.RULES["K1_close_after_request"][2](f).any()
    assert dq.naive_timestamp_violations(f) == 2   # naive comparison wrongly flags both


# ------------------------------------------------------------ data assertions
@pytest.fixture(scope="module")
def data():
    return dq.load()


@pytest.mark.data
@pytest.mark.parametrize("rule", list(dq.RULES))
def test_dataset_satisfies(data, rule):
    n = int(dq.RULES[rule][2](data).fillna(False).sum())
    assert n == 0, f"{rule} ({dq.RULES[rule][0]}): {n} violating records"
