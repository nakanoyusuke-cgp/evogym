from typing import Callable, Optional


class StateHandler(object):
    def __init__(self, debug_mode=False):
        self._state_callbacks = {}
        self._state = None
        self._state_step = 0
        self._debug_mode = debug_mode

    def add_state(self, name: str, callback: Callable[[float], Optional[str]]) -> None:
        if name not in self._state_callbacks.keys():
            # self._state_callbacks += {name: callback}
            self._state_callbacks[name] = callback
        else:
            print("error, the state handler still has state named" + name)

    def set_state(self, name: str) -> None:
        if name in self._state_callbacks.keys():
            if self._debug_mode:
                print("[DEBUG_LOG] state changed:", name)
            self._state_step = 0
            self._state = name
        else:
            print("error (set state), the state handler has not state named" + name)

    def step(self) -> None:
        next_state = self._state_callbacks[self._state](self._state_step)
        if next_state is not None:
            self.set_state(next_state)
        else:
            self._state_step += 1

    @property
    def state_step(self):
        return self._state_step

    @property
    def state(self):
        return self._state
