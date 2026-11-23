"""Startup44 delivery repair; its startup, mapping, exploration and freeze stay inherited.

Only real messages establish contact tickets/custody. Movement starts through
the original missing-map budget gate, never an added early fleet return timer.
"""
from collections import Counter
from types import SimpleNamespace
import math
import numpy as np

from swarm_rescue.solutions.my_drone_red_startup44 import MyDroneRedStartup44
from swarm_rescue.solutions.my_drone_red_linked_passage43 import MyDroneRedLinkedPassage43
from swarm_rescue.solutions.red_startup44_delivery import (
    delivered, valid_status, valid_ticket, replace_fusion_extra, relay_source, elect_carrier,
)


class MyDroneRedStartup44Delivery(MyDroneRedStartup44):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dl44 = SimpleNamespace(
            stats=Counter(), events=[], samples=[], peers={}, proposals={}, tickets={},
            local_nodes={}, offers={}, root_offer=None, role='explore', target=None,
            active_ticket=None, stage=0, started=None, arrived=None,
            best_cost=float('inf'), advance=0, custody_since=None, receipts={},
            last_versions={}, rx_steps={}, sent_sources=Counter(), first_motion=None,
            route_node=None, route_kind=None, elected=None,
        )

    def _delivery_note44(self, kind, **data):
        s = self._dl44
        if len(s.events) < 6000:
            s.events.append(dict(step=int(self._steps), kind=kind, **data))
        s.stats[kind] += 1

    def _status44(self):
        own = int(self.identifier or 0)
        versions = tuple((i, v[1], self._radio_maps.payload_digest(i))
                         for i, v in sorted(self._radio_maps.snapshots.items()))
        return (own, int(self.elapsed_timestep),
                None if self._position is None else tuple(map(float, self._position)),
                self._radio_maps.own_final is not None, self._d39_mode, versions,
                self._dl44.role, self._dl44.target)

    def define_message_for_all(self):
        base = super().define_message_for_all()
        s = self._dl44; own = int(self.identifier or 0); clock = int(self.elapsed_timestep)
        chosen = relay_source(self._radio_maps.snapshots, self._lp43_receipt,
                              self._radio_maps.payload_digest, self._steps)
        if chosen is not None and self._radio_maps.own_final is not None:
            base = replace_fusion_extra(base, self._radio_maps.snapshots[chosen])
            s.sent_sources[chosen] += 1
        offer = None
        if own == 0 and self._position is not None and self._walk39.last is not None:
            offer = (clock, tuple(map(float, self._position)))
            s.offers[clock] = (offer[1], self._walk39.last)
            for t in list(s.offers):
                if clock-t > 90: del s.offers[t]
        return ('startup44-delivery-v1', base, self._status44(), offer,
                tuple(s.proposals.values()), tuple(s.tickets.values()))

    def _receive(self):
        s = self._dl44; own = int(self.identifier or 0); clock = int(self.elapsed_timestep)
        stripped = []
        for sender, message in self.communicator.received_messages:
            if not (isinstance(message, tuple) and len(message) == 6
                    and message[0] == 'startup44-delivery-v1'):
                continue
            _, base, status, offer, proposals, tickets = message
            stripped.append((sender, base))
            if not valid_status(status, clock, self._n): continue
            peer = status[0]
            if peer == own or clock-status[1] > 4: continue
            s.peers[peer] = status
            s.rx_steps[peer] = clock
            if (own != 0 and peer == 0 and self._position is not None
                    and isinstance(offer, tuple) and len(offer) == 2
                    and type(offer[0]) is int and 0 <= clock-offer[0] <= 4):
                proposal = (own, offer[0], offer[1], tuple(map(float, self._position)))
                if valid_ticket(proposal, clock, self._n) and self._walk39.last is not None:
                    s.proposals[own] = proposal
                    s.local_nodes[(own, offer[0])] = self._walk39.last
            if own == 0 and isinstance(proposals, tuple) and len(proposals) <= self._n:
                for ticket in proposals:
                    if not valid_ticket(ticket, clock, self._n) or ticket[0] != peer: continue
                    offer_record = s.offers.get(ticket[1])
                    if offer_record is None or offer_record[0] != ticket[2]: continue
                    old = s.tickets.get(peer)
                    if old is None or ticket[1] > old[1]:
                        s.tickets[peer] = ticket
                        s.local_nodes[(peer, ticket[1])] = offer_record[1]
            if isinstance(tickets, tuple) and len(tickets) <= self._n:
                for ticket in tickets:
                    if not valid_ticket(ticket, clock, self._n): continue
                    old = s.tickets.get(ticket[0])
                    if own != 0 and (old is None or ticket[1] > old[1]):
                        s.tickets[ticket[0]] = ticket
        proxy = SimpleNamespace(**self.__dict__)
        proxy.elapsed_timestep = self.elapsed_timestep
        proxy.communicator = SimpleNamespace(received_messages=stripped)
        proxy._valid_peer = self._valid_peer; proxy._valid_contract39 = self._valid_contract39
        proxy._note39 = self._note39; proxy._note44 = self._note44
        MyDroneRedStartup44._receive(proxy)
        for name in ('_f27_home', '_d39_contract', '_d39_rx_step', '_lp43_receipt', '_st44_plan'):
            setattr(self, name, getattr(proxy, name))
        receipt = self._lp43_receipt
        if receipt is not None:
            for source, sequence, digest in receipt[1]:
                old = s.receipts.get(source)
                if old is None or sequence > old[0]:
                    s.receipts[source] = (sequence, digest, receipt[0])
                    self._delivery_note44('submitter_receipt', source=source, sequence=sequence,
                                          digest=digest, generated=receipt[0])
        for source, snap in self._radio_maps.snapshots.items():
            if s.last_versions.get(source) != snap[1]:
                s.last_versions[source] = snap[1]
                self._delivery_note44('source_available', source=source, sequence=snap[1],
                                      digest=self._radio_maps.payload_digest(source))
        # Keep just the current/previous confirmation and recent offer nodes.
        keep = {(t[0], t[1]) for t in s.tickets.values()} | {(t[0], t[1]) for t in s.proposals.values()}
        if s.active_ticket is not None: keep.add((s.active_ticket[0], s.active_ticket[1]))
        for key in list(s.local_nodes):
            if key not in keep and clock-key[1] > 90: del s.local_nodes[key]

    def _link_state43(self):
        if int(self.identifier or 0) == 0 or self._radio_maps.own_final is None:
            return super()._link_state43()
        store = self._radio_maps; own = int(self.identifier or 0)
        sealed = store.own_final is not None
        good = delivered(self._lp43_receipt, own, store.snapshots.get(own),
                         store.payload_digest(own), int(self.elapsed_timestep), sealed)
        self._lp43_link = dict(healthy=good, sealed=sealed, missing_sources=[] if good else [own],
            leader_receipt_step=None if self._lp43_receipt is None else self._lp43_receipt[0],
            exact_delivery_separate_from_link_age=True)
        return good

    def _pending44(self):
        store = self._radio_maps
        if int(self.identifier or 0) == 0:
            return [i for i in range(self._n) if str(i) in store.delivery()['missing']]
        return [i for i, snap in store.snapshots.items()
                if not delivered(self._lp43_receipt, i, snap, store.payload_digest(i),
                                 int(self.elapsed_timestep), store.own_final is not None)]

    def _choose_ticket44(self):
        tickets = list(self._dl44.tickets.values())
        return max(tickets, key=lambda t: (t[1], -t[0])) if tickets else None

    def _set_delivery_target44(self, ticket, stage=0):
        s = self._dl44; graph = self._walk39; own = int(self.identifier or 0)
        if not graph.points: return
        node = None
        if ticket is not None:
            if stage == 0: node = s.local_nodes.get((ticket[0], ticket[1]))
            position = ticket[2] if own == 0 else ticket[3]
            if stage == 1: position = ticket[3] if own == 0 else ticket[2]
            if node is None and stage < 2:
                node = min(range(len(graph.points)), key=lambda i: np.linalg.norm(graph.points[i]-position))
        if node is None: node = 0
        s.route_node = node; s.target = tuple(map(float, graph.points[node]))
        s.active_ticket = ticket; s.stage = stage; s.started = self._steps
        s.arrived = None; s.best_cost = float('inf'); s.advance = self._steps
        s.route_kind = 'confirmed_radio_footprint' if stage == 0 and ticket else 'bounded_alternate'
        self._d39_plan_step = -100
        self._delivery_note44('target_selected', ticket=ticket, node=node, stage=stage,
                              target=s.target, role=s.role)

    def _path39(self):
        s = self._dl44
        # The exploration budget still prices the original route. A new target
        # cannot cause an earlier first return than the existing budget policy.
        if self._radio_maps.own_final is None or self._d39_mode == 'explore' or s.route_node is None:
            return super()._path39()
        graph = self._walk39
        self._lp43_target_node = s.route_node
        s.target = tuple(map(float, graph.points[s.route_node]))
        self._d39_route, self._d39_cost = graph.path(self._position, self._heading,
            self.lidar_values(), self.lidar_rays_angles(), self._steps, target=graph.points[s.route_node])
        self._d39_plan_step = self._steps

    def _navigation(self):
        if not self._st44_done: return super()._navigation()
        s = self._dl44; sealed = self._radio_maps.own_final is not None
        own = int(self.identifier or 0)
        # After the unchanged freeze, a robot whose own map is delivered can
        # still carry someone else's missing final. Do not park it prematurely.
        if sealed and self._pending44() and own != 0 and self._link_state43():
            self._d39_mode = 'return'
            return self._return_motion39()
        before = self._d39_mode
        command = super()._navigation()
        if self._d39_mode == 'explore':
            s.role = 'explore'; s.route_node = None; s.active_ticket = None
            s.custody_since = None
        if before == 'explore' and self._d39_mode != 'explore' and s.first_motion is None:
            s.first_motion = self._steps
            self._delivery_note44('first_delivery_motion', sealed=sealed, budget=dict(self._d39_budget))
        return command

    def _hold_delivery44(self):
        command = self._empty_command(); v = np.clip(-.7*self._velocity, -1., 1.)
        command.update(forward=float(v[0]), lateral=float(v[1]), rotation=0.)
        self._holding = True; self._d39_mode = 'exchange'
        return command

    def _return_motion39(self):
        # R2 scope guard: contact bookkeeping must not perturb the exploration
        # trajectory or the old return executor before the existing freeze.
        if self._radio_maps.own_final is None:
            return MyDroneRedLinkedPassage43._return_motion39(self)
        s = self._dl44; own = int(self.identifier or 0); clock = int(self.elapsed_timestep)
        ticket = self._choose_ticket44(); pending = self._pending44()
        sealed = self._radio_maps.own_final is not None
        states = [self._status44()] + [v for v in s.peers.values() if clock-v[1] <= 30]
        elected = elect_carrier(states, ticket, pending) if ticket else own
        s.elected = elected
        if sealed and own != 0 and elected is not None and elected != own:
            carrier = s.peers.get(elected)
            snap = self._radio_maps.snapshots.get(own)
            custody = carrier is not None and snap is not None and any(
                row == (own, snap[1], self._radio_maps.payload_digest(own)) for row in carrier[5])
            if custody:
                if s.custody_since is None:
                    s.custody_since = self._steps
                    self._delivery_note44('custody_confirmed', carrier=elected, source=own, sequence=snap[1])
                # One bounded wait; neither a neighbour's promise nor a stale
                # lease is treated as successful end-to-end delivery.
                if self._steps-s.custody_since < 75:
                    s.role = 'custody_wait'
                    return self._hold_delivery44()
        s.role = 'receiver' if own == 0 else 'courier'
        if s.route_node is None or (ticket is not None and s.active_ticket != ticket and s.stage == 0):
            self._set_delivery_target44(ticket)
        if s.route_node is None: return super()._return_motion39()
        self._d39_mode = 'return'; self._path39()
        cost = self._d39_cost
        if math.isfinite(cost) and cost < s.best_cost-15:
            s.best_cost = cost; s.advance = self._steps
        target = self._walk39.points[s.route_node]
        distance = float(np.linalg.norm(self._position-target))
        if distance < 30:
            if s.arrived is None:
                s.arrived = self._steps
                self._delivery_note44('contact_arrived', role=s.role, stage=s.stage, pending=pending)
            if not pending:
                s.role = 'exchange'
                return self._hold_delivery44()
            remaining = max(0., self._d39_budget.get('remaining', 0.))
            window = max(15, min(60, int(remaining*.15)))
            if self._steps-s.arrived < window:
                return self._hold_delivery44()
            self._delivery_note44('contact_no_delivery', pending=pending, waited=self._steps-s.arrived)
            self._set_delivery_target44(ticket, (s.stage+1) % 3)
            self._path39()
        elif self._steps-s.advance > 100:
            self._delivery_note44('route_no_progress', pending=pending, route_cost=cost)
            self._set_delivery_target44(ticket, (s.stage+1) % 3)
            self._path39()
        # Reuse unchanged observed-path avoidance and fine hull clearance.
        # This layer owns rendezvous expiry; suppress the old 120-step/home rule.
        self._lp43_contact_wait = self._steps
        return MyDroneRedLinkedPassage43._return_motion39(self)

    def control(self):
        command = super().control()
        if self._steps % 30 == 0:
            s = self._dl44
            s.samples.append(dict(step=int(self._steps), role=s.role, mode=self._d39_mode,
                target=s.target, ticket=s.active_ticket, stage=s.stage, elected=s.elected,
                position=None if self._position is None else self._position.tolist(),
                pending=self._pending44(), receipt=self._lp43_receipt,
                versions={str(i):v[1] for i,v in self._radio_maps.snapshots.items()},
                route_cost=self._d39_cost if self._d39_cost is not None and math.isfinite(self._d39_cost) else None))
        return command
