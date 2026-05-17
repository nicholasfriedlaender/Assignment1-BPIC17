"""Slot schema and binning helpers for BPIC2017. Experimental, moved away from binning as accuracy decreased"""
from __future__ import annotations
import math
import pandas as pd


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def bin_duration(seconds) -> str:
    if _is_nan(seconds):
        return "start"
    s = float(seconds)
    if s < 60:          return "instant"
    if s < 3_600:       return "minutes"
    if s < 28_800:      return "hours"
    if s < 86_400:      return "half-day"
    if s < 604_800:     return "days"
    if s < 2_419_200:   return "weeks"
    return "months"

def bin_amount(v) -> str:
    if _is_nan(v):
        return "NaN"
    x = float(v)
    if x < 5_000:    return "very_low"
    if x < 10_000:   return "low"
    if x < 20_000:   return "medium"
    if x < 50_000:   return "high"
    return "very_high"

def bin_monthly_cost(v) -> str:
    if _is_nan(v):
        return "NaN"
    x = float(v)
    if x < 100:  return "very_low"
    if x < 200:  return "low"
    if x < 400:  return "medium"
    if x < 700:  return "high"
    return "very_high"

def bin_terms(v) -> str:
    if _is_nan(v):
        return "NaN"
    x = float(v)
    if x < 24:   return "short"
    if x < 60:   return "medium"
    if x < 120:  return "long"
    return "very_long"

def bin_credit(v) -> str:
    if _is_nan(v):
        return "NaN"
    x = float(v)
    if x == 0:   return "not_scored"
    if x < 700:  return "low"
    if x < 900:  return "medium"
    return "high"

def fmt_str(v) -> str:
    if _is_nan(v):
        return "NaN"
    s = str(v).strip()
    return s if s else "NaN"

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def fmt_weekday(v) -> str:
    if _is_nan(v):
        return "NaN"
    try:
        return _WEEKDAYS[pd.Timestamp(v).weekday()]
    except Exception:
        return "NaN"



OUTCOME_LABELS = ["A_Denied", "A_Cancelled", "A_Pending"]


#   (slot_key, transform_fn, source_column)
EVENT_SLOTS = [
    ("A",   fmt_str,          "concept:name"),
    ("R",   fmt_str,          "org:resource"),
    ("Dur", bin_duration,     "duration_sec"),
    ("WD",  fmt_weekday,      "time:timestamp"),
    ("LG",  fmt_str,          "case:LoanGoal"),
    ("AT",  fmt_str,          "case:ApplicationType"),
    ("Amt", bin_amount,       "case:RequestedAmount"),
    ("FW",  bin_amount,       "FirstWithdrawalAmount"),
    ("NT",  bin_terms,        "NumberOfTerms"),
    ("MC",  bin_monthly_cost, "MonthlyCost"),
    ("CS",  bin_credit,       "CreditScore"),
    ("OA",  bin_amount,       "OfferedAmount"),
]
