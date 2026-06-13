from datetime import datetime, timezone

from market_state_news_tensor.data.universe import filter_universe_asof
from market_state_news_tensor.schemas.labels import FeatureRow
from market_state_news_tensor.schemas.universe import UniverseMembership


def dt(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=timezone.utc)


def row(symbol: str) -> FeatureRow:
    return FeatureRow(symbol=symbol, decision_time=dt(10), available_at=dt(10))


def membership(
    *,
    symbol: str = "A",
    start: datetime | None = None,
    end: datetime | None = None,
    tradable: bool = True,
    observable: bool = True,
    eligible: bool = True,
) -> UniverseMembership:
    return UniverseMembership(
        symbol=symbol,
        start=start or dt(1),
        end=end or dt(31),
        tradable=tradable,
        observable=observable,
        eligible=eligible,
        reason=None,
    )


def test_symbol_included_during_active_membership() -> None:
    filtered = filter_universe_asof([row("A")], [membership()], dt(10))

    assert [item.symbol for item in filtered] == ["A"]


def test_symbol_excluded_before_start_date() -> None:
    filtered = filter_universe_asof([row("A")], [membership(start=dt(11))], dt(10))

    assert filtered == []


def test_symbol_excluded_after_end_date() -> None:
    filtered = filter_universe_asof([row("A")], [membership(end=dt(9))], dt(10))

    assert filtered == []


def test_symbol_excluded_when_tradable_false() -> None:
    filtered = filter_universe_asof([row("A")], [membership(tradable=False)], dt(10))

    assert filtered == []


def test_symbol_excluded_when_observable_false() -> None:
    filtered = filter_universe_asof([row("A")], [membership(observable=False)], dt(10))

    assert filtered == []


def test_symbol_excluded_when_eligible_false() -> None:
    filtered = filter_universe_asof([row("A")], [membership(eligible=False)], dt(10))

    assert filtered == []
