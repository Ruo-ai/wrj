"""Bounded, observation-only map transport and rendered-map agreement.

Connectivity evidence is generated only from an actual direct receipt. Relays
preserve its origin clock. A version report or remote ACK is never a map payload.
The cooperative protocol detects inconsistent data; it is not authentication
against a malicious participant impersonating another source.
"""
from collections import Counter
import hashlib

import numpy as np


def hash36(value):
    return hashlib.sha256(repr(value).encode('ascii')).hexdigest()


def digest36(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class MeshBook36:
    TAG = 'mesh36-v1'

    def __init__(self, source, count=5, edge_ttl=90, report_ttl=180):
        if type(source) is not int or type(count) is not int or not 1 <= count <= 16 or not 0 <= source < count:
            raise ValueError('Finite source identity/count required')
        if type(edge_ttl) is not int or type(report_ttl) is not int or min(edge_ttl, report_ttl) <= 0:
            raise ValueError('Positive finite integer TTL required')
        self.source, self.count = source, count
        self.edge_ttl, self.report_ttl = edge_ttl, report_ttl
        self.edges = {}  # (observer, directly heard peer) -> original clock
        self.reports = {}  # holder -> (holder, origin clock, source versions)
        self.acks = {}  # origin -> (origin, epoch, input set hash, rendered hash, clock)
        self.local_ack = None
        self.final_pixels = None
        self._final_key = None
        self._sent = {}
        self._send_serial = 0
        self.stats = Counter()

    def _id(self, value):
        return type(value) is int and 0 <= value < self.count

    @staticmethod
    def _clock(value):
        return type(value) is int and value >= 0

    def _fresh(self, origin, now, ttl):
        return self._clock(origin) and 0 <= now-origin <= ttl

    def _prune(self, clock):
        self.edges = {k:t for k,t in self.edges.items() if self._fresh(t,clock,self.edge_ttl)}
        self.reports = {i:r for i,r in self.reports.items() if self._fresh(r[1],clock,self.report_ttl)}

    def observe_direct(self, peer, clock):
        """Call only with sender from the native received_messages envelope."""
        if not self._id(peer) or peer == self.source or not self._clock(clock):
            self.stats['invalid_direct_receipt'] += 1
            return False
        self.edges[(self.source, peer)] = clock
        self.stats['direct_receipts'] += 1
        return True

    def _own_report(self, clock, store):
        versions = tuple(store.snapshots[i][1] if i in store.snapshots else -1 for i in range(self.count))
        row = (self.source, clock, versions)
        self.reports[self.source] = row
        return row

    def _accept_snapshot(self, snapshot, clock, store):
        if snapshot is None:
            return False
        if (not isinstance(snapshot, tuple) or len(snapshot) != 3 or not self._id(snapshot[0])
                or not self._clock(snapshot[1]) or snapshot[1] > clock
                or not isinstance(snapshot[2], bytes) or len(snapshot[2]) != 2*store.bytes_per_mask):
            self.stats['invalid_snapshot'] += 1
            return False
        source, version, blob = snapshot
        previous = store.snapshots.get(source)
        if source == self.source:
            self.stats['own_snapshot_echo_ignored'] += 1
            return False
        if previous is not None and version <= previous[1]:
            label = ('duplicate_snapshot' if version == previous[1] and blob == previous[2]
                     else 'conflicting_snapshot' if version == previous[1] else 'old_snapshot')
            self.stats[label] += 1
            return False
        accepted = bool(store.accept(snapshot))
        self.stats['accepted_snapshot' if accepted else 'store_rejected_snapshot'] += 1
        return accepted

    def ingest(self, sender, packet, clock, store):
        """Ingest one actually received packet; never use reported pose as RX."""
        if (not self._id(sender) or sender == self.source or not self._clock(clock)
                or not isinstance(packet, tuple) or len(packet) != 6 or packet[0] != self.TAG):
            self.stats['invalid_packet'] += 1
            return False
        _, edges, reports, snapshot, manifests, acks = packet
        bounded = ((edges,self.count*(self.count-1)), (reports,self.count),
                   (manifests,self.count), (acks,self.count))
        if any(not isinstance(rows,tuple) or len(rows)>limit for rows,limit in bounded):
            self.stats['oversized_or_invalid_packet'] += 1
            return False
        self._prune(clock)
        self.observe_direct(sender,clock)
        for row in edges:
            valid = (isinstance(row,tuple) and len(row)==3 and self._id(row[0]) and self._id(row[1])
                     and row[0]!=row[1] and self._fresh(row[2],clock,self.edge_ttl))
            if not valid:
                self.stats['invalid_or_expired_edge'] += 1
                continue
            observer, peer, origin = row
            if observer == self.source:
                # Even a plausible relayed row cannot invent our own contact.
                if self.edges.get((observer,peer)) != origin:
                    self.stats['own_edge_claim_ignored'] += 1
                continue
            key = (observer,peer)
            if origin > self.edges.get(key,-1):
                self.edges[key] = origin
        for row in reports:
            valid = (isinstance(row,tuple) and len(row)==3 and self._id(row[0])
                and self._fresh(row[1],clock,self.report_ttl) and isinstance(row[2],tuple)
                and len(row[2])==self.count and all(type(v) is int and -1<=v<=row[1] for v in row[2]))
            if not valid:
                self.stats['invalid_or_expired_report'] += 1
                continue
            if row[0] == self.source:
                continue
            old = self.reports.get(row[0])
            if old is None or row[1]>old[1]:
                self.reports[row[0]] = row
        # Manifests are small; forward all five while sending at most one blob.
        for row in manifests:
            valid = (isinstance(row,tuple) and len(row)==5 and self._id(row[0])
                and self._clock(row[1]) and 0<row[1]<=clock and self._clock(row[2]) and row[2]<=row[1]
                and digest36(row[3]) and type(row[4]) is int and row[4]>=0)
            if not valid:
                self.stats['invalid_manifest'] += 1
            elif row[0] != self.source:
                store.accept_manifest(row)
        accepted = self._accept_snapshot(snapshot,clock,store)
        for row in acks:
            valid = (isinstance(row,tuple) and len(row)==5 and self._id(row[0]) and self._clock(row[1])
                and 0<row[1]<=clock and digest36(row[2]) and digest36(row[3]) and self._clock(row[4])
                and row[1]<=row[4]<=clock and store.epoch is not None and row[1]==store.epoch)
            if not valid:
                self.stats['invalid_render_ack'] += 1
                continue
            if row[0] == self.source:
                if row != self.local_ack:
                    self.stats['forged_own_render_ack_ignored'] += 1
                continue
            previous = self.acks.get(row[0])
            if previous is None:
                self.acks[row[0]] = row
            elif previous != row:
                self.stats['conflicting_render_ack'] += 1
        self._own_report(clock,store)
        return accepted

    def direct_neighbors(self, clock):
        self._prune(clock)
        return sorted(peer for (observer,peer) in self.edges if observer==self.source)

    def select_payload(self, clock, store):
        self._prune(clock)
        neighbors = self.direct_neighbors(clock)
        if not store.snapshots:
            return None
        def priority(source):
            version = store.snapshots[source][1]
            missing = sum(peer not in self.reports or self.reports[peer][2][source]<version for peer in neighbors)
            return (-missing,self._sent.get(source,-1),source)
        source = min(store.snapshots,key=priority)
        self._send_serial += 1
        self._sent[source] = self._send_serial
        self.stats['payload_packets'] += 1
        return store.snapshots[source]

    def packet(self, clock, store, include_payload=True):
        if not self._clock(clock):
            raise ValueError('Native nonnegative integer clock required')
        self._prune(clock)
        self._own_report(clock,store)
        manifests = tuple(store.manifests[i] for i in sorted(store.manifests))
        acks = tuple(self.acks[i] for i in sorted(self.acks)
                     if store.epoch is None or self.acks[i][1]==store.epoch)
        return (self.TAG,tuple((a,b,t) for (a,b),t in sorted(self.edges.items())),
                tuple(self.reports[i] for i in sorted(self.reports)),
                self.select_payload(clock,store) if include_payload else None,manifests,acks)

    def confirm_render(self, store, pixels, clock):
        """Bind an actual complete local set to one immutable final image.

        The controller supplies its actual renderer output exactly once and
        submits ``final_pixels``. The same sealed set reuses that image/hash;
        later pixel arguments do not silently replace the confirmed image.
        """
        if not self._clock(clock):
            raise ValueError('Native nonnegative integer clock required')
        state = store.delivery()
        if not state['final_complete']:
            self.stats['render_confirmation_incomplete'] += 1
            return None
        parts = tuple((i,state['received_versions'][str(i)],state['received_digests'][str(i)])
                      for i in range(self.count))
        key = (state['epoch_elapsed'],parts)
        if self._final_key == key:
            self.stats['render_confirmation_cache_hit'] += 1
            return self.local_ack
        if self.local_ack is not None:
            raise ValueError('Already confirmed immutable final set changed')
        check_epoch = state['epoch_elapsed']
        if not self._clock(check_epoch) or not 0<check_epoch<=clock:
            raise ValueError('Cannot confirm a future or invalid final epoch')
        # Metadata and store.delivery alone cannot create a rendered ACK.
        for i,version,digest in parts:
            snapshot,manifest = store.snapshots[i],store.manifests[i]
            if (snapshot[0]!=i or snapshot[1]!=version or len(snapshot[2])!=2*store.bytes_per_mask
                    or hashlib.sha256(snapshot[2]).hexdigest()!=digest
                    or tuple(manifest)!=(i,check_epoch,version,digest,0)):
                raise ValueError('Final source bytes do not match immutable manifest')
        candidate = np.asarray(pixels)
        if candidate.ndim != 2 or not candidate.size or not np.isfinite(candidate).all():
            raise ValueError('Actual finite two-dimensional renderer output required')
        rendered = np.asarray(candidate,dtype=bool).copy(order='C')
        rendered.setflags(write=False)
        image_digest = hashlib.sha256(rendered.tobytes()).hexdigest()
        self.stats['render_hash_computed'] += 1
        self.final_pixels = rendered
        self._final_key = key
        self.local_ack = (self.source,check_epoch,hash36(parts),image_digest,clock)
        self.acks[self.source] = self.local_ack
        return self.local_ack

    @staticmethod
    def _reachable(start, links):
        visited,queue={start},[start]
        while queue:
            source=queue.pop()
            for a,b in links:
                if a==source and b not in visited:
                    visited.add(b);queue.append(b)
        return visited

    def status(self, clock, store):
        self._prune(clock)
        self._own_report(clock,store)
        # A received-edge(observer,peer) represents data flow peer -> observer.
        directed={(peer,observer) for observer,peer in self.edges}
        mutual={edge for edge in directed if (edge[1],edge[0]) in directed}
        component=self._reachable(self.source,mutual)
        outgoing=self._reachable(self.source,directed)
        incoming=self._reachable(self.source,{(b,a) for a,b in directed})
        strong=outgoing&incoming
        local=store.delivery()
        matching=[]
        if self.local_ack is not None and local['final_complete']:
            matching=sorted(i for i,ack in self.acks.items() if ack[1:4]==self.local_ack[1:4])
        neighbors=self.direct_neighbors(clock)
        needs={str(peer):[i for i,s in store.snapshots.items()
               if peer not in self.reports or self.reports[peer][2][i]<s[1]] for peer in neighbors}
        return dict(source=self.source,direct_neighbors=neighbors,
            directed_edges=[list(e) for e in sorted(directed)],
            fresh_bidirectional_edges=[list(e) for e in sorted(mutual) if e[0]<e[1]],
            edge_origin_clocks=[list((*edge,t)) for edge,t in sorted(self.edges.items())],
            component=sorted(component),strong_component=sorted(strong),
            all_bidirectionally_connected=len(component)==self.count,
            all_strongly_connected=len(strong)==self.count,
            can_receive_from=sorted(incoming),can_send_to=sorted(outgoing),
            missing_sources=sorted(map(int,local['missing'])),source_maps_complete=local['final_complete'],
            received_versions=dict(local['received_versions']),neighbors_needing_sources=needs,
            version_reports={str(i):list(r) for i,r in self.reports.items()},
            local_render_confirmed=self.local_ack is not None,
            input_set_hash=None if self.local_ack is None else self.local_ack[2],
            render_hash=None if self.local_ack is None else self.local_ack[3],
            render_ack_sources=matching,missing_render_ack_sources=[i for i in range(self.count) if i not in matching],
            render_complete=local['final_complete'] and self.local_ack is not None and len(matching)==self.count,
            reported_render_acks={str(i):list(a) for i,a in self.acks.items()},stats=dict(self.stats))
