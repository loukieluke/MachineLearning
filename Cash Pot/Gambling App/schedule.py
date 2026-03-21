"""
Cash Pot draw schedule (Jamaica).
Draws: 8:30 AM | 10:30 AM | 1:00 PM | 3:00 PM | 5:00 PM | 8:25 PM
Every day except Christmas Day and Good Friday.
"""
from datetime import datetime, date, time, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

JAMAICA = ZoneInfo("America/Jamaica")

# Game name -> time (24h) for the draw
GAME_TO_TIME = {
    "EARLYBIRD": time(8, 30),
    "MORNING": time(10, 30),
    "MIDDAY": time(13, 0),
    "MIDAFTERNOON": time(15, 0),
    "DRIVETIME": time(17, 0),
    "EVENING": time(20, 25),
}

# Chronological order for "next draw" logic
DRAW_TIMES_ORDERED = [
    (time(8, 30), "EARLYBIRD"),
    (time(10, 30), "MORNING"),
    (time(13, 0), "MIDDAY"),
    (time(15, 0), "MIDAFTERNOON"),
    (time(17, 0), "DRIVETIME"),
    (time(20, 25), "EVENING"),
]


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_blackout_date(d: date) -> bool:
    """True if Cash Pot has no draws on this date (Christmas, Good Friday)."""
    if d.month == 12 and d.day == 25:
        return True
    easter_sunday = _easter(d.year)
    good_friday = easter_sunday - timedelta(days=2)
    return d == good_friday


def get_next_draw_datetime() -> datetime:
    """Next draw datetime in Jamaica time (timezone-aware)."""
    now = datetime.now(JAMAICA)  # aware datetime
    today = now.date()

    # Find next slot today
    for t, game in DRAW_TIMES_ORDERED:
        if is_blackout_date(today):
            break
        # Make draw datetime aware in Jamaica timezone
        draw_dt = datetime.combine(today, t).replace(tzinfo=JAMAICA)
        if now < draw_dt:
            return draw_dt

    # Next valid day
    day = today
    for _ in range(366):
        day += timedelta(days=1)
        if is_blackout_date(day):
            continue
        first_draw = datetime.combine(day, DRAW_TIMES_ORDERED[0][0]).replace(tzinfo=JAMAICA)
        return first_draw

    # Fallback: today at first draw time
    return datetime.combine(today, DRAW_TIMES_ORDERED[0][0]).replace(tzinfo=JAMAICA)


def format_target_draw_at(dt: datetime) -> str:
    """Store as ISO-like string for DB (date and time)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def target_draw_at_from_date_and_game(draw_date: str, game: str):
    """Build target_draw_at string from draw date and game (e.g. for linking). Returns None if game unknown."""
    draw_date = (draw_date or "").strip()
    if len(draw_date) < 10:
        return None
    # Normalize to YYYY-MM-DD (HTML date input sends this; CSV might vary)
    date_str = draw_date[:10]
    if date_str[4] == "-" and date_str[7] == "-":
        pass  # already YYYY-MM-DD
    else:
        try:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                try:
                    d = datetime.strptime(draw_date[:10], fmt)
                    date_str = d.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                return None
        except Exception:
            return None
    game_upper = (game or "").strip().upper()
    if game_upper not in GAME_TO_TIME:
        return None
    t = GAME_TO_TIME[game_upper]
    return f"{date_str} {t.hour:02d}:{t.minute:02d}:00"
