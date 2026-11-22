"""Radio-carried rendezvous agreement; independent of final-map ACKs.

Every acceptance retains its origin timestamp. A committed plan binds the
entire team to one observed meeting point, a window and an observed fallback.
This is cooperative consistency checking, not cryptographic authentication.
"""
import hashlib
import math


def digest35(fields):
    return hashlib.sha256(repr(tuple(fields)).encode('ascii')).hexdigest()


def point35(value):
    return (isinstance(value, tuple) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


class RendezvousBook35:
    def __init__(self, source, count):
        self.source, self.count = int(source), int(count)
        self.proposal = None
        self.accepts = {}
        self.commit = None
        self.seen_commits = {}
        self.rejections = 0

    def propose(self, clock, meeting, fallback):
        if self.source != 0 or self.proposal is not None:
            return False
        fields = (0, 1, int(clock), tuple(map(float, meeting)),
                  tuple(map(float, fallback)), .88, .93, .965)
        self.proposal = (*fields, digest35(fields))
        return True

    def valid_plan(self, p, clock):
        return (isinstance(p, tuple) and len(p) == 9 and p[0] == 0
                and type(p[1]) is int and p[1] > 0
                and type(p[2]) is int and 0 <= p[2] <= clock
                and point35(p[3]) and point35(p[4])
                and all(type(v) in (int, float) and math.isfinite(v) for v in p[5:8])
                and 0 < p[5] < p[6] < p[7] < .975 and p[8] == digest35(p[:8]))

    def valid_ack(self, a, clock):
        return (self.proposal is not None and isinstance(a, tuple) and len(a) == 3
                and type(a[0]) is int and 0 <= a[0] < self.count
                and type(a[1]) is int and self.proposal[2] <= a[1] <= clock
                and a[2] == self.proposal[8])

    def accept_local(self, clock, observed_route):
        if self.proposal is None or not observed_route:
            return False
        self.accepts.setdefault(self.source, (self.source, int(clock), self.proposal[8]))
        return True

    def ingest(self, packet, clock):
        if not isinstance(packet, tuple) or len(packet) != 4:
            self.rejections += 1
            return
        p, accepts, commit, seen = packet
        if p is not None:
            if not self.valid_plan(p, clock):
                self.rejections += 1
                return
            # One frozen agreement per flight. Never move a disconnected peer's
            # rendezvous by unilateral re-proposal or by replaying another plan.
            if self.proposal is None:
                self.proposal = p
            elif p != self.proposal:
                self.rejections += 1
                return
        if isinstance(accepts, tuple) and len(accepts) <= self.count:
            for a in accepts:
                if self.valid_ack(a, clock):
                    if a[0] == self.source and a != getattr(self, '_local_accept', None):
                        self.rejections += 1
                        continue
                    old = self.accepts.get(a[0])
                    if old is None or a[1] < old[1]:
                        self.accepts[a[0]] = a
        if commit is not None:
            valid = (isinstance(commit, tuple) and len(commit) == 4 and commit[0] == 0
                     and type(commit[1]) is int and 0 <= commit[1] <= clock
                     and self.proposal is not None and commit[2] == self.proposal[8]
                     and isinstance(commit[3], tuple) and len(commit[3]) == self.count
                     and all(self.valid_ack(a, clock) and a[1] <= commit[1] for a in commit[3])
                     and {a[0] for a in commit[3]} == set(range(self.count)))
            # Each participant must really have accepted locally; a remote
            # receipt purporting to be our acceptance cannot activate the plan.
            own = next((a for a in commit[3] if a[0] == self.source), None) if valid else None
            if valid and own == self.accepts.get(self.source) and getattr(self, '_local_accept', None) == own:
                if self.commit is None or commit == self.commit:
                    self.commit = commit
            elif not valid:
                self.rejections += 1
        if isinstance(seen, tuple) and len(seen) <= self.count:
            for a in seen:
                if self.valid_ack(a, clock) and self.commit is not None and a[1] >= self.commit[1]:
                    self.seen_commits.setdefault(a[0], a)
        self.advance(clock)

    def accept(self, clock, observed_route):
        if self.accept_local(clock, observed_route):
            self._local_accept = (self.source, int(clock), self.proposal[8]) if not hasattr(self, '_local_accept') else self._local_accept
            self.accepts[self.source] = self._local_accept
            self.advance(clock)
            return True
        return False

    def advance(self, clock):
        if (self.source == 0 and self.commit is None and self.proposal is not None
                and len(self.accepts) == self.count and hasattr(self, '_local_accept')):
            self.commit = (0, int(clock), self.proposal[8],
                           tuple(self.accepts[i] for i in range(self.count)))
        if self.commit is not None:
            self.seen_commits.setdefault(self.source, (self.source, int(clock), self.proposal[8]))

    def packet(self):
        return (self.proposal, tuple(self.accepts[i] for i in sorted(self.accepts)),
                self.commit, tuple(self.seen_commits[i] for i in sorted(self.seen_commits)))

    def status(self):
        p = self.proposal
        return dict(plan_id=None if p is None else p[8], proposer=0,
                    proposal_created_elapsed=None if p is None else p[2],
                    meeting=None if p is None else list(p[3]),
                    fallback=None if p is None else list(p[4]),
                    arrival_progress=None if p is None else p[5],
                    window_end_progress=None if p is None else p[6],
                    finish_progress=None if p is None else p[7],
                    accepted_sources=sorted(self.accepts), committed=self.commit is not None,
                    committed_seen_sources=sorted(self.seen_commits),
                    local_accepted=hasattr(self, '_local_accept'), rejections=self.rejections)
