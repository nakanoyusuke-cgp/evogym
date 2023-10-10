from evogym.viewer import EvoViewer
from evogym.simulator_cpp import VisualProcessor, Viewer


class VisLineViewer(EvoViewer):
    def __init__(self, vis_proc: VisualProcessor, **kwargs):
        super().__init__(**kwargs)

        self.vis_proc = vis_proc

    def _init_viewer(self):
        if not self._has_init_viewer:
            self._viewer = Viewer(self._sim)
            self._has_init_viewer = True
            self._viewer.set_vis_proc(self.vis_proc)
