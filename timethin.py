#!/usr/bin/env python3
"""
CLI tool which filters a list of strings containing datetimes,
based on intervals in which those datetimes occur.
Designed for thinning of file-based backups with a nontrivial retention policy.
"""
import datetime as dt
import logging
import re
import sys
from abc import ABC
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List

from _strptime import TimeRE

logger = logging.getLogger(__name__)

GLOBAL_DOC = __doc__
NOW = dt.datetime.now()
DEFAULT_PATTERN = "%Y-%m-%dT%H:%M:%s"


def none_iter(obj):
    if obj is None:
        return
    yield from obj


def lines_from_filelike(path: str):
    if path == "-":
        yield from sys.stdin
    else:
        with open(path) as f:
            yield from f


@contextmanager
def writeable_filelike(path: str, mode="w"):
    if mode not in set("wax"):
        raise ValueError(f"Unknown mode {mode}")

    if path == "-":
        yield sys.stdout
    else:
        with open(path, mode) as f:
            yield f


def yield_strs(strs, from_files):
    yield from none_iter(strs)
    for fpath in none_iter(from_files):
        yield from lines_from_filelike(fpath)


def sanitise_strs(strs: Iterable[str]):
    for s in strs:
        stripped = s.strip()
        if stripped:
            yield stripped


@dataclass
class Interval:
    start: dt.datetime
    stop: dt.datetime

    @property
    def interval(self) -> dt.timedelta:
        return self.stop - self.start

    def __contains__(self, other: dt.datetime):
        return self.start <= other < self.stop

    def subdivide(self, delta: dt.timedelta):
        start = self.start
        stop = start + delta
        cls = type(self)
        while stop < self.stop:
            yield cls(start, stop)
            start = stop
            stop = start + delta
        yield cls(start, stop)


@dataclass
class Policy:
    retention: dt.timedelta
    rate: dt.timedelta

    @classmethod
    def from_str(cls, s):
        logger.debug("Parsing policy '%s'", s)
        ret_rate = s.strip().split("=>")
        if len(ret_rate) == 1:
            ret = ret_rate[0]
            rate = ret_rate[0]
        elif len(ret_rate) == 2:
            ret, rate = ret_rate
        else:
            raise ValueError("Could not parse policy: " + s)
        return cls(parse_duration(ret), parse_duration(rate))

    def matches(self, datetimes: Iterable[dt.datetime], now=None):
        if now is None:
            now = NOW
        interval = Interval(now - self.retention, now)
        subs = interval.subdivide(self.rate)
        datetimes = sorted(datetimes)
        out = []
        for sub in subs:
            this_out = []
            idx_to_remove = []
            for idx, datetime in enumerate(datetimes):
                if datetime in sub:
                    this_out.append(datetime)
                    idx_to_remove.append(idx)
            out.append(this_out)
            for idx in reversed(idx_to_remove):
                datetimes.pop(idx)
        return out


@dataclass
class MultiPolicy:
    policies: List[Policy]

    @classmethod
    def from_str(cls, s):
        logger.debug("Parsing multipolicy '%s'", s)
        ps = s.strip().split(",")
        return cls([Policy.from_str(p) for p in ps])

    def matches(self, datetimes: List[dt.datetime], now=None):
        if now is None:
            now = NOW
        return [p.matches(datetimes, now) for p in self.policies]


durations = {}
for s in ["second", "sec", "s"]:
    durations[s] = dt.timedelta(seconds=1)
for s in ["minute", "min"]:
    durations[s] = dt.timedelta(minutes=1)
for s in ["hour", "h"]:
    durations[s] = dt.timedelta(hours=1)
for s in ["day", "d"]:
    durations[s] = dt.timedelta(days=1)
for s in ["week", "w"]:
    durations[s] = dt.timedelta(weeks=1)
for s in ["month", "mon", "m"]:
    durations[s] = dt.timedelta(days=31)
for s in ["year", "y"]:
    durations[s] = dt.timedelta(days=366)

duration_str = "|".join(k for k in durations)
duration_re = re.compile(r"^(?P<number>\d*)(?P<duration>" + duration_str + r")$")


def parse_duration(s):
    """
    second|sec|s
    minute|min
    hour|h
    day|d
    week|w
    month|mon|m
    year|y
    month standardised to 31 days
    year standardised to 366 days
    """
    m = duration_re.fullmatch(s.lower())
    if m is None:
        raise ValueError("Invalid duration: " + s)

    groups = m.groupdict()
    number = int(groups.get("number") or 1)
    duration = groups["duration"]
    return number * durations[duration]


def re_for_pattern(pattern: str):
    re_pattern = pattern
    for k, v in TimeRE().items():
        re_pattern = re_pattern.replace("%" + k, v)
    logger.debug("compiling regex '%s'", re_pattern)
    return re.compile(re_pattern)


def _main(
    pattern: str,
    policy: MultiPolicy,
    arg_strs: List[str],
    from_file_strs: List[str],
    invert: bool,
    now: dt.datetime,
):
    regex = re_for_pattern(pattern)

    dd = defaultdict(list)
    strs = set()

    for s in sanitise_strs(yield_strs(arg_strs, from_file_strs)):
        all_matches = [m.group() for m in regex.finditer(s)]
        if not all_matches:
            logger.info("No matching datetime: skipping %s", s)
            continue
        if len(all_matches) > 1:
            logger.warning(
                "Taking last of %s multiple possible datetimes: %s",
                len(all_matches),
                s,
            )

        m = all_matches.pop()
        dd[dt.datetime.strptime(m, pattern)].append(s)
        strs.add(s)

    matches = policy.matches(sorted(dd), now=now)
    dt_to_keep = set()
    for outer in matches:
        for inner in outer:
            if inner:
                # take earliest in each window
                dt_to_keep.add(inner[0])

    strs_to_keep = set()
    for d in dt_to_keep:
        strs_to_keep.update(dd[d])

    if invert:
        return sorted(strs - strs_to_keep)
    else:
        return sorted(strs_to_keep)


def main(args=None):
    parser = ArgumentParser(description=GLOBAL_DOC)
    parser.add_argument(
        "policy",
        type=MultiPolicy.from_str,
        help="datetimes to keep given as 'retA=>intA,retB=>intB' etc; see znapzendzetup docs https://github.com/oetiker/znapzend/blob/master/doc/znapzendzetup.pod#create",
    )
    parser.add_argument("strings", nargs="*", help="any number of strings to filter")
    parser.add_argument(
        "--pattern",
        "-p",
        default=DEFAULT_PATTERN,
        help=f"Datetime pattern; see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes ; default second-precision timezone-unaware ISO-8601, i.e. '{DEFAULT_PATTERN.replace('%', '%%')}'",
    )
    parser.add_argument(
        "--in-file",
        "-i",
        help="read newline-separated strings from file; - for stdin",
        action="append",
    )
    parser.add_argument(
        "--out-file",
        "-o",
        help="write result to file; - for stdout (default)",
        default="-",
    )
    parser.add_argument(
        "--invert",
        "-i",
        action="store_true",
        help="print files which should be removed, not kept",
    )
    parser.add_argument(
        "--now",
        "-n",
        type=dt.datetime.fromisoformat,
        default=NOW,
        help=(
            "effective time to use in ISO-8601 format; "
            "default now i.e. " + NOW.isoformat()
        ),
    )
    parser.add_argument("--verbose", "-v", action="count", help="increase verbosity")
    parsed = parser.parse_args(args)
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        parsed.verbose, logging.DEBUG
    )
    logging.basicConfig(level=level)

    with writeable_filelike(parsed.out_file) as f:
        for line in _main(
            parsed.pattern,
            parsed.policy,
            parsed.strings,
            parsed.from_file,
            parsed.invert,
            parsed.now,
        ):
            print(line, file=f)


if __name__ == "__main__":
    main()
