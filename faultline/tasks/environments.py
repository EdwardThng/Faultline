"""Mock tool environments for benchmark tasks.

Each environment is a small, deterministic, stateful world exposing tools
through the FaultLine wrapper. Determinism matters twice over: success
checkers assert on exact final state, and faulted re-runs must differ from
baseline only by the injected fault.
"""

from __future__ import annotations

from typing import Any, Callable

from faultline.harness.wrapper import Tool


class Environment:
    """Base: a state dict plus tools that read and mutate it."""

    name = "base"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.state: dict[str, Any] = {}

    def tools(self) -> dict[str, Tool]:
        raise NotImplementedError


class TravelEnv(Environment):
    name = "travel"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.state = {
            "flights": {
                "FL100": {"origin": "SFO", "destination": "JFK",
                          "price": 250, "seats": 4},
                "FL200": {"origin": "SFO", "destination": "JFK",
                          "price": 310, "seats": 0},
                "FL300": {"origin": "JFK", "destination": "LHR",
                          "price": 620, "seats": 9},
            },
            "bookings": {},
            "next_booking": 1,
        }

    def search_flights(self, origin: str, destination: str) -> list[dict]:
        return [
            {"flight_id": fid, **f}
            for fid, f in sorted(self.state["flights"].items())
            if f["origin"] == origin.upper()
            and f["destination"] == destination.upper()
        ]

    def book_flight(self, flight_id: str, passenger: str) -> dict:
        flight = self.state["flights"].get(flight_id)
        if flight is None:
            raise ValueError(f"no such flight: {flight_id}")
        if flight["seats"] < 1:
            raise ValueError(f"{flight_id} is sold out")
        booking_id = f"B{self.state['next_booking']:03d}"
        self.state["next_booking"] += 1
        flight["seats"] -= 1
        self.state["bookings"][booking_id] = {
            "flight_id": flight_id,
            "passenger": passenger,
            "status": "confirmed",
        }
        return {"booking_id": booking_id, "status": "confirmed"}

    def get_booking(self, booking_id: str) -> dict:
        b = self.state["bookings"].get(booking_id)
        if b is None:
            raise ValueError(f"no such booking: {booking_id}")
        return dict(b)

    def tools(self) -> dict[str, Tool]:
        return _as_tools(
            {
                "search_flights": (
                    "Search flights by route",
                    {"origin": "str airport code",
                     "destination": "str airport code"},
                    self.search_flights,
                ),
                "book_flight": (
                    "Book a seat on a flight",
                    {"flight_id": "str", "passenger": "str full name"},
                    self.book_flight,
                ),
                "get_booking": (
                    "Fetch a booking by id",
                    {"booking_id": "str"},
                    self.get_booking,
                ),
            }
        )


class FilesEnv(Environment):
    name = "files"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.state = {
            "files": {
                "data/sales.csv": (
                    "item,units,unit_price\n"
                    "keyboard,5,90\nmouse,10,25\nmonitor,2,275\n"
                ),
                "notes/todo.txt": "ship the Q3 report\n",
            }
        }

    def list_files(self) -> list[str]:
        return sorted(self.state["files"])

    def read_file(self, path: str) -> str:
        if path not in self.state["files"]:
            raise FileNotFoundError(path)
        return self.state["files"][path]

    def write_file(self, path: str, content: str) -> dict:
        self.state["files"][path] = content
        return {"path": path, "bytes": len(content)}

    def tools(self) -> dict[str, Tool]:
        return _as_tools(
            {
                "list_files": ("List all file paths", {}, self.list_files),
                "read_file": (
                    "Read a file's contents", {"path": "str"}, self.read_file
                ),
                "write_file": (
                    "Write a file",
                    {"path": "str", "content": "str"},
                    self.write_file,
                ),
            }
        )


class OrdersEnv(Environment):
    name = "orders"

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.state = {
            "orders": {
                "O-1001": {"item": "keyboard", "amount": 89,
                           "status": "delivered",
                           "customer_email": "sam@example.com"},
                "O-1002": {"item": "desk lamp", "amount": 40,
                           "status": "shipped",
                           "customer_email": "riley@example.com"},
            },
            "emails": [],
        }

    def lookup_order(self, order_id: str) -> dict:
        o = self.state["orders"].get(order_id)
        if o is None:
            raise ValueError(f"no such order: {order_id}")
        return dict(o)

    def refund_order(self, order_id: str, amount: int) -> dict:
        o = self.state["orders"].get(order_id)
        if o is None:
            raise ValueError(f"no such order: {order_id}")
        if amount != o["amount"]:
            raise ValueError(
                f"refund amount {amount} does not match order total"
            )
        o["status"] = "refunded"
        return {"order_id": order_id, "refunded": amount,
                "status": "refunded"}

    def send_email(self, to: str, subject: str, body: str) -> dict:
        self.state["emails"].append(
            {"to": to, "subject": subject, "body": body}
        )
        return {"sent": True, "to": to}

    def tools(self) -> dict[str, Tool]:
        return _as_tools(
            {
                "lookup_order": (
                    "Fetch an order by id", {"order_id": "str"},
                    self.lookup_order,
                ),
                "refund_order": (
                    "Refund an order in full",
                    {"order_id": "str", "amount": "int, must equal total"},
                    self.refund_order,
                ),
                "send_email": (
                    "Send an email",
                    {"to": "str", "subject": "str", "body": "str"},
                    self.send_email,
                ),
            }
        )


def _as_tools(
    spec: dict[str, tuple[str, dict[str, str], Callable[..., Any]]]
) -> dict[str, Tool]:
    return {
        name: Tool(name=name, description=desc, parameters=params, fn=fn)
        for name, (desc, params, fn) in spec.items()
    }


ENVIRONMENTS: dict[str, type[Environment]] = {
    cls.name: cls for cls in (TravelEnv, FilesEnv, OrdersEnv)
}


def make_environment(name: str, seed: int = 0) -> Environment:
    if name not in ENVIRONMENTS:
        raise KeyError(
            f"unknown environment '{name}'; have {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[name](seed=seed)
