import asyncio
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from tqdm.auto import tqdm


class RateLimitProgressMonitor:
    """
    Multi-progress monitor to visualize per-model rate-limit usage.

    For each model we maintain three bars:
    - Requests  
    - Input tokens per minute
    - Output tokens per minute
    - Total in/out tokens in the current session

    Notes:
    - For OpenAI we know the per-minute caps via the Resource objects. We use those
      to set bar totals and compute remaining/used values.
    - Token usage split (in vs out) is tracked from actual responses over a rolling
      60-second window using deques. This provides an accurate view of where the
      token cap is being spent.
    """

    def __init__(self, disable: bool = False) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._disable: bool = disable
        # model_id -> (request_bar, in_bar, out_bar)
        self._bars: Dict[str, Tuple[Optional[tqdm], Optional[tqdm], Optional[tqdm]]] = {}
        # model_id -> caps
        self._caps: Dict[str, Dict[str, Optional[int]]] = {}
        # model_id -> resource refs for OpenAI (may be None for other providers)
        self._resources: Dict[str, Dict[str, Any]] = {}
        # rolling windows for last-minute accounting
        self._events_requests: Dict[str, Deque[float]] = {}
        self._events_in_tokens: Dict[str, Deque[Tuple[float, int]]] = {}
        self._events_out_tokens: Dict[str, Deque[Tuple[float, int]]] = {}
        # cumulative totals since registration
        self._total_requests: Dict[str, int] = {}
        self._total_in_tokens: Dict[str, int] = {}
        self._total_out_tokens: Dict[str, int] = {}
        # totals start time (since last reset)
        self._totals_start_time: float = time.time()
        # position bookkeeping for tqdm multi-bars
        self._next_position: int = 0

    def _ensure_deques(self, model_id: str) -> None:
        if model_id not in self._events_requests:
            self._events_requests[model_id] = deque()
        if model_id not in self._events_in_tokens:
            self._events_in_tokens[model_id] = deque()
        if model_id not in self._events_out_tokens:
            self._events_out_tokens[model_id] = deque()
        if model_id not in self._total_requests:
            self._total_requests[model_id] = 0
        if model_id not in self._total_in_tokens:
            self._total_in_tokens[model_id] = 0
        if model_id not in self._total_out_tokens:
            self._total_out_tokens[model_id] = 0

    @staticmethod
    def _prune_old(ts_deque: Deque, window_seconds: int = 60) -> None:
        now = time.time()
        while ts_deque and (now - (ts_deque[0][0] if isinstance(ts_deque[0], tuple) else ts_deque[0])) >= window_seconds:
            ts_deque.popleft()

    def register_openai_model(self, model_id: str, request_resource: Any, token_resource: Any) -> None:
        """
        Register an OpenAI model with known rate-limit resources.

        request_resource/token_resource are instances of the OpenAI Resource class with
        attributes: refresh_rate (cap per minute) and value (remaining budget that replenishes).
        """
        self._ensure_deques(model_id)

        request_cap = int(request_resource.refresh_rate)
        token_cap = int(token_resource.refresh_rate)
        self._caps[model_id] = {"request_cap": request_cap, "token_cap": token_cap}
        self._resources[model_id] = {
            "request": request_resource,
            "token": token_resource,
        }

        if self._disable:
            # still track metrics but don't render bars
            self._bars[model_id] = (None, None, None)
            return

        if model_id in self._bars and all(bar is not None for bar in self._bars[model_id]):
            # already registered
            return

        # Allocate three lines for this model
        req_bar = tqdm(
            total=request_cap,
            position=self._next_position,
            leave=True,
            unit="req",
            desc=f"{model_id} | requests",
            dynamic_ncols=True,
        )
        in_bar = tqdm(
            total=token_cap,
            position=self._next_position + 1,
            leave=True,
            unit="tok",
            desc=f"{model_id} | in tokens",
            dynamic_ncols=True,
        )
        out_bar = tqdm(
            total=token_cap,
            position=self._next_position + 2,
            leave=True,
            unit="tok",
            desc=f"{model_id} | out tokens",
            dynamic_ncols=True,
        )

        self._bars[model_id] = (req_bar, in_bar, out_bar)
        self._next_position += 3

    def register_generic_model(self, model_id: str, show_token_bars: bool = False) -> None:
        """
        Register a model without known caps.
        - If show_token_bars is False, only a request bar is created (no token bars displayed).
        - If show_token_bars is True, request + token bars are created and grow with observed usage.
        """
        self._ensure_deques(model_id)
        # caps unknown
        self._caps[model_id] = {"request_cap": None, "token_cap": None}
        self._resources[model_id] = {}

        if self._disable:
            self._bars[model_id] = (None, None, None)
            return

        if model_id in self._bars and all(bar is not None for bar in self._bars[model_id]):
            return

        req_bar = tqdm(
            total=1,
            position=self._next_position,
            leave=True,
            unit="req",
            desc=f"{model_id} | requests",
            dynamic_ncols=True,
        )
        if show_token_bars:
            in_bar = tqdm(
                total=1,
                position=self._next_position + 1,
                leave=True,
                unit="tok",
                desc=f"{model_id} | in tokens",
                dynamic_ncols=True,
            )
            out_bar = tqdm(
                total=1,
                position=self._next_position + 2,
                leave=True,
                unit="tok",
                desc=f"{model_id} | out tokens",
                dynamic_ncols=True,
            )
            self._bars[model_id] = (req_bar, in_bar, out_bar)
            self._next_position += 3
        else:
            self._bars[model_id] = (req_bar, None, None)
            self._next_position += 1

    async def update_openai_usage(
        self,
        model_id: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        request_increment: int = 1,
    ) -> None:
        """
        Update rolling usage and refresh progress bars for the given model.
        """
        async with self._lock:
            self._ensure_deques(model_id)
            now = time.time()
            # push new events
            for _ in range(request_increment):
                self._events_requests[model_id].append(now)
            self._total_requests[model_id] += int(request_increment)
            if input_tokens is not None:
                self._events_in_tokens[model_id].append((now, int(input_tokens)))
                self._total_in_tokens[model_id] += int(input_tokens)
            if output_tokens is not None:
                self._events_out_tokens[model_id].append((now, int(output_tokens)))
                self._total_out_tokens[model_id] += int(output_tokens)

            # prune
            self._prune_old(self._events_requests[model_id])
            self._prune_old(self._events_in_tokens[model_id])
            self._prune_old(self._events_out_tokens[model_id])

            # compute window sums
            req_count_window = len(self._events_requests[model_id])
            in_tok_window = sum(v for _, v in self._events_in_tokens[model_id])
            out_tok_window = sum(v for _, v in self._events_out_tokens[model_id])

            # update progress bars if present
            req_bar, in_bar, out_bar = self._bars.get(model_id, (None, None, None))
            caps = self._caps.get(model_id, {"request_cap": None, "token_cap": None})
            request_cap = caps.get("request_cap")
            token_cap = caps.get("token_cap")

            # If we have resource refs, compute used from resource directly for requests
            resources = self._resources.get(model_id)
            if resources is not None:
                try:
                    request_resource = resources.get("request")
                    # trigger replenish and compute used
                    used_req = int(request_resource.refresh_rate - request_resource.value)
                    req_count_window = max(req_count_window, used_req)
                except Exception:
                    pass

            if req_bar is not None:
                if request_cap is not None:
                    req_bar.total = int(request_cap)
                    req_bar.n = int(min(req_count_window, request_cap))
                else:
                    # scale dynamically
                    req_bar.total = max(int(req_bar.total or 1), req_count_window or 1)
                    req_bar.n = int(req_count_window)
                # show totals in postfix
                try:
                    req_bar.set_postfix(total=self._total_requests.get(model_id, 0), window=req_count_window)
                except Exception:
                    pass
                req_bar.refresh()

            if in_bar is not None:
                if token_cap is not None:
                    in_bar.total = int(token_cap)
                    in_bar.n = int(min(in_tok_window, token_cap))
                else:
                    in_bar.total = max(int(in_bar.total or 1), in_tok_window or 1)
                    in_bar.n = int(in_tok_window)
                try:
                    in_bar.set_postfix(total=self._total_in_tokens.get(model_id, 0), window=in_tok_window)
                except Exception:
                    pass
                in_bar.refresh()

            if out_bar is not None:
                if token_cap is not None:
                    out_bar.total = int(token_cap)
                    out_bar.n = int(min(out_tok_window, token_cap))
                else:
                    out_bar.total = max(int(out_bar.total or 1), out_tok_window or 1)
                    out_bar.n = int(out_tok_window)
                try:
                    out_bar.set_postfix(total=self._total_out_tokens.get(model_id, 0), window=out_tok_window)
                except Exception:
                    pass
                out_bar.refresh()

    def close(self) -> None:
        if self._disable:
            return
        for bars in self._bars.values():
            for bar in bars:
                if bar is not None:
                    try:
                        bar.close()
                    except Exception:
                        pass

    def reset_totals(self, model_id: Optional[str] = None) -> None:
        """
        Reset cumulative totals for a specific model or for all models.
        Does not affect rolling 60s windows or bar positions.
        """
        if model_id is None:
            self._total_requests = {k: 0 for k in self._total_requests.keys()}
            self._total_in_tokens = {k: 0 for k in self._total_in_tokens.keys()}
            self._total_out_tokens = {k: 0 for k in self._total_out_tokens.keys()}
        else:
            self._total_requests[model_id] = 0
            self._total_in_tokens[model_id] = 0
            self._total_out_tokens[model_id] = 0
        self._totals_start_time = time.time()

    def get_snapshot(self, model_id: str) -> Dict[str, Any]:
        """
        Return a snapshot of rolling-window usage and caps for a model.

        Keys:
        - request_cap, token_cap: caps per minute if known
        - req_count_window, in_tok_window, out_tok_window: usage in the last minute
        - total_requests, total_in_tokens, total_out_tokens: cumulative totals since registration
        """
        self._ensure_deques(model_id)
        # prune before snapshot
        self._prune_old(self._events_requests[model_id])
        self._prune_old(self._events_in_tokens[model_id])
        self._prune_old(self._events_out_tokens[model_id])

        caps = self._caps.get(model_id, {"request_cap": None, "token_cap": None})
        return {
            "request_cap": caps.get("request_cap"),
            "token_cap": caps.get("token_cap"),
            "req_count_window": len(self._events_requests[model_id]),
            "in_tok_window": sum(v for _, v in self._events_in_tokens[model_id]),
            "out_tok_window": sum(v for _, v in self._events_out_tokens[model_id]),
            "total_requests": self._total_requests.get(model_id, 0),
            "total_in_tokens": self._total_in_tokens.get(model_id, 0),
            "total_out_tokens": self._total_out_tokens.get(model_id, 0),
            "totals_since": self._totals_start_time,
        }


