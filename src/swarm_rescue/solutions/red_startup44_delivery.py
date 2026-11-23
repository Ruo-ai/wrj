"""Source delivery bookkeeping. No map, seed, simulator or true-pose inputs."""
import math

from swarm_rescue.solutions.red_linked_passage43 import valid_receipt


def delivered(receipt, source, snapshot, digest, clock, sealed=False):
    """An exact byte receipt survives a broken link; a newer payload does not.

    Live pipeline tolerance is retained only while the receipt is recent.
    Sequence numbers alone never prove that different bytes were delivered.
    """
    if snapshot is None or not valid_receipt(receipt, 5, clock):
        return False
    row = next((r for r in receipt[1] if r[0] == source), None)
    if row is None or row[1] > snapshot[1]:
        return False
    if row[2] == digest:
        return not sealed or row[1] == snapshot[1]
    return (not sealed and clock-receipt[0] <= 120
            and 0 <= snapshot[1]-row[1] <= 90 and clock-row[1] <= 120)


def point(value):
    return (isinstance(value, tuple) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) and abs(x) < 1e6 for x in value))


def valid_ticket(ticket, clock, number=5):
    # peer id, root offer time, root position, peer position
    return (isinstance(ticket, tuple) and len(ticket) == 4
            and type(ticket[0]) is int and 0 < ticket[0] < number
            and type(ticket[1]) is int and 0 <= ticket[1] <= clock
            and point(ticket[2]) and point(ticket[3]))


def valid_status(status, clock, number=5):
    if not isinstance(status, tuple) or len(status) != 8:
        return False
    own, step, position, sealed, mode, versions, role, target = status
    return (type(own) is int and 0 <= own < number and type(step) is int
            and 0 <= step <= clock and (position is None or point(position))
            and type(sealed) is bool and mode in ('explore', 'return', 'exchange')
            and role in ('explore', 'courier', 'receiver', 'custody_wait', 'exchange')
            and isinstance(versions, tuple) and valid_receipt((step, versions), number, clock)
            and (target is None or point(target)))


def replace_fusion_extra(message, extra):
    """Replace the existing relay slot, not add a second full-map payload."""
    if not isinstance(message, tuple) or len(message) < 2:
        return message
    if message[0] == 'fusion27-v1':
        return (message[0], message[1], extra, *message[3:])
    if isinstance(message[1], tuple):
        return (message[0], replace_fusion_extra(message[1], extra), *message[2:])
    return message


def relay_source(snapshots, receipt, digest, step):
    if not snapshots:
        return None
    sources = sorted(snapshots)
    rows = dict((r[0], r) for r in receipt[1]) if receipt else {}
    pending = [i for i in sources if i not in rows or rows[i][1] != snapshots[i][1]
               or rows[i][2] != digest(i)]
    # Every third slot is an unconditional fair relay for peer-to-peer ACKs.
    pool = sources if step % 3 == 0 or not pending else pending
    return pool[(step//3) % len(pool)]


def elect_carrier(statuses, ticket, missing):
    """Only robots already in delivery participate; ongoing search is protected."""
    candidates = []
    for status in statuses:
        own, _, _, sealed, mode, versions, _, _ = status
        if own == 0 or not (sealed or mode != 'explore'):
            continue
        count = sum(i in missing for i, _, _ in versions)
        if count:
            # Prefer the owner of the mutually confirmed contact footprint.
            candidates.append((own != ticket[0], -count, own))
    return min(candidates)[2] if candidates else None
