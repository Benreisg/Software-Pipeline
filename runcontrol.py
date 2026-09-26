"""
runcontrol.py — pause or stop a running experiment from the console
========================================================================
A long paid run should not have to be killed with Ctrl-C, which loses the
aggregation and leaves a half-written run directory behind.

    Enter                     pause — takes effect before the next API call
    Enter again               resume
    end + Enter               stop now, dropping the call in flight
    stop / quit / q + Enter   the same, for whichever word comes to hand
    pause / resume + Enter    the explicit forms of the Enter toggle
    ? + Enter                 print the list again
    Ctrl-C                    the same as `end` (see `stop_on_sigint`)

Pausing acts *between* calls: it takes effect before the next request, so a
call on the wire is never interrupted by it.

**Stopping does not wait.** The request in flight is dropped and the loop gets
control back at once, rather than sitting out a call that can take minutes on a
reasoning model. The vendor may still complete and bill that request; the run
keeps no reply for it and records the row as abandoned, so a stopped run's
totals are a lower bound on what the account was charged. Everything generated
before it is kept, and the run still scores, reports and writes its manifest.

Only one thread ever touches stdin — this one. The experiment loop just reads
two flags (`stopped`, and blocking inside `wait_if_paused`), so there is no
competing `input()` anywhere.
"""
from __future__ import annotations

import contextlib
import signal
import sys
import threading
import time
from typing import Optional

STOP_WORD = "end"
# Every word that ends the run. One of them is the documented one; the others
# are what a person actually types when a run is burning money and the printed
# word has scrolled off the screen.
STOP_WORDS = {STOP_WORD, "stop", "quit", "exit", "q"}
PAUSE_WORDS = {"pause", "p"}
RESUME_WORDS = {"resume", "continue", "go", "r", "c"}
HELP_WORDS = {"?", "h", "help"}
_POLL_SECONDS = 0.15


class RunControl:
    """Keyboard pause/stop for the generation loop.

    The keyboard half is disabled automatically when stdin is not a terminal
    (piped, redirected or a scheduled run), where there is nobody to press a key
    and `readline()` would return EOF immediately. The stop flag itself keeps
    working there: `request_stop` is reachable without a keyboard, and Ctrl-C
    uses it.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and _stdin_is_interactive()
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._listen, name="runcontrol",
                                        daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop reacting to keys. The listener thread is a daemon blocked on
        stdin; it cannot be interrupted portably, so it is simply told to
        ignore whatever arrives next and left to die with the process."""
        self._finished.set()

    # ── state the loop reads ─────────────────────────────────────────────
    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def request_stop(self, reason: str = "") -> None:
        """Ask the run to stop, from anywhere: the listener thread, a signal
        handler, or a caller that already knows the run is pointless.

        Idempotent, and the first reason is the one printed — Ctrl-C during a
        keyboard stop must not tell the story twice.
        """
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._paused.clear()              # release a paused loop
        if reason:
            print(f"\n  {reason}\n")

    def wait_if_paused(self) -> bool:
        """Block while paused. Returns False when the run should stop.

        Called before each API call, so pausing never interrupts a request that
        is already on the wire. The stop flag is honoured even with the keyboard
        listener disabled: Ctrl-C has to work on a piped run too.
        """
        while self._paused.is_set() and not self._stopped.is_set():
            time.sleep(_POLL_SECONDS)
        return not self._stopped.is_set()

    # ── what the console says about all this ─────────────────────────────
    def hint(self) -> str:
        """The one line a run prints about its own controls.

        Without it the mechanism may as well not exist: nothing else on screen
        says a running experiment can be paused, and the only key a person
        reaches for is Ctrl-C.
        """
        if not self.enabled:
            return ("  Controls: Ctrl-C stops at once, dropping the call in flight "
                    "— everything generated before it is kept.")
        return (f"  Controls: [Enter] pauses before the next call, [Enter] again "
                f"resumes · '{STOP_WORD}' + [Enter] stops at once, dropping the call in "
                f"flight · Ctrl-C does the same · '?' lists this again.")

    # ── listener thread ──────────────────────────────────────────────────
    def _listen(self) -> None:
        while not self._stopped.is_set() and not self._finished.is_set():
            try:
                line = sys.stdin.readline()
            except (ValueError, OSError):     # stdin closed under us
                return
            if line == "":                    # EOF
                return
            if self._finished.is_set():
                return

            if not self.handle(line):
                return

    def handle(self, line: str) -> bool:
        """React to one line of input. Returns False when there is nothing left
        to listen for, which is only ever after a stop.

        Split out of the listener thread so the command vocabulary can be
        tested without a terminal.
        """
        command = line.strip().lower()
        if command in STOP_WORDS:
            self.request_stop(f"'{command}' received — stopping now. The call in "
                              f"flight is dropped, not waited for; the vendor may still "
                              f"bill it. Everything finished before it is kept.")
            return False
        if command in HELP_WORDS:
            print(f"\n{self.hint()}\n")
            return True
        # Enter alone is the toggle it always was; the words are for the reader
        # who does not remember that and types what they mean.
        if command in PAUSE_WORDS or (command == "" and not self._paused.is_set()):
            self._paused.set()
            print(f"\n  Paused before the next API call. [Enter] or 'resume' to "
                  f"continue, '{STOP_WORD}' to stop.\n")
            return True
        if command in RESUME_WORDS or command == "":
            self._paused.clear()
            print("\n  Resumed.\n")
            return True
        print(f"\n  Unknown input {command!r} — [Enter] pauses/resumes, "
              f"'{STOP_WORD}' stops, '?' lists the commands.\n")
        return True


@contextlib.contextmanager
def stop_on_sigint(control: Optional[RunControl]):
    """Make Ctrl-C ask the run to stop instead of killing it.

    The first Ctrl-C sets exactly the flag `end` sets: the call in flight is
    dropped, and the run stops and still scores, reports and writes its manifest
    as a partial run — where a plain KeyboardInterrupt would have thrown all of
    that away.

    The default handler is put back at once, so a **second** Ctrl-C aborts the
    hard way. A run wedged in a network read must stay killable, and a person
    pressing Ctrl-C twice means it.
    """
    if control is None:
        yield
        return
    try:
        previous = signal.getsignal(signal.SIGINT)
    except (ValueError, AttributeError, OSError):
        yield                                 # no signal handling available here
        return

    def _first_interrupt(_signum, _frame):
        signal.signal(signal.SIGINT, previous)
        control.request_stop(
            "Ctrl-C — stopping now. The call in flight is dropped, not waited for; "
            "everything finished before it is kept, scored and reported. Ctrl-C again "
            "to abort outright.")

    try:
        signal.signal(signal.SIGINT, _first_interrupt)
    except (ValueError, OSError):             # signals only work on the main thread
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGINT, previous)


def _stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False
